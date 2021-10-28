import numpy as np
import ngmix
import galsim
import joblib

import pytest

from ..metacal import metacal_op_shears, DEFAULT_SHEARS, get_gauss_reconv_psf
from ngmix.gaussmom import GaussMom

FITTER = GaussMom(1.2)


def _make_sim(*, seed, g1, g2, s2n):
    rng = np.random.RandomState(seed=seed)

    gal = galsim.Exponential(half_light_radius=0.5).shear(g1=g1, g2=g2)
    psf = galsim.Gaussian(fwhm=0.9)
    obj = galsim.Convolve([gal, psf])
    dim = 53
    cen = (dim-1)/2
    dither = rng.uniform(size=2, low=-0.5, high=0.5)
    scale = 0.263

    im = obj.drawImage(nx=53, ny=53, offset=dither, scale=scale).array
    nse = np.sqrt(np.sum(im**2)) / s2n
    im += rng.normal(size=im.shape, scale=nse)

    psf_im = psf.drawImage(nx=53, ny=53, scale=scale).array

    jac = ngmix.DiagonalJacobian(
        scale=scale, row=cen+dither[1], col=cen+dither[0]
    )
    psf_jac = ngmix.DiagonalJacobian(
        scale=scale, row=cen, col=cen
    )

    obs = ngmix.Observation(
        image=im,
        weight=np.ones_like(im) / nse**2,
        jacobian=jac,
        psf=ngmix.Observation(
            image=psf_im,
            jacobian=psf_jac,
        ),
        noise=rng.normal(size=im.shape, scale=nse),
    )
    return obs


def _run_fit_mcal_res(mcal_res):
    psf_res = FITTER.go(mcal_res["noshear"].psf)
    if psf_res["flags"] != 0:
        return None

    dt = [
        ("mcal_flags", "i4"),
        ("mcal_g", "f8", (2,)),
        ("mcal_T_ratio", "f8"),
        ("mcal_s2n", "f8"),
        ("shear", "U7"),
    ]
    vals = []
    for shear, obs in mcal_res.items():
        res = FITTER.go(obs)
        vals.append(
            (res["flags"], res["e"], res["T"]/psf_res["T"], res["s2n"], shear)
        )
    d = np.array(vals, dtype=dt)
    if np.any(d["mcal_flags"] != 0):
        return None
    else:
        return d


def _run_single_sim_pair(seed, s2n):
    obs_plus = _make_sim(seed=seed, g1=0.02, g2=0.0, s2n=s2n)
    reconv_psf = get_gauss_reconv_psf(obs_plus)
    mcal_res = metacal_op_shears(obs_plus, reconv_psf)
    res_p = _run_fit_mcal_res(mcal_res)
    if res_p is None or np.any(res_p["mcal_flags"] != 0):
        return None

    obs_minus = _make_sim(seed=seed, g1=-0.02, g2=0.0, s2n=s2n)
    reconv_psf = get_gauss_reconv_psf(obs_minus)
    mcal_res = metacal_op_shears(obs_minus, reconv_psf)
    res_m = _run_fit_mcal_res(mcal_res)
    if res_m is None or np.any(res_m["mcal_flags"] != 0):
        return None

    return res_p, res_m


def _msk_it(*, d, s2n_cut, size_cut, shear):
    return (
        (d["shear"] == shear) &
        (d["mcal_flags"] == 0) &
        (d["mcal_s2n"] > s2n_cut) &
        (d["mcal_T_ratio"] > size_cut)
    )


def _measure_g1g2R(*, d, s2n_cut, size_cut):
    msks = {}
    for shear in DEFAULT_SHEARS:
        msks[shear] = _msk_it(
            d=d, s2n_cut=s2n_cut, size_cut=size_cut, shear=shear)

    g1_1p = np.mean(d['mcal_g'][msks['1p'], 0])
    g1_1m = np.mean(d['mcal_g'][msks['1m'], 0])
    g2_2p = np.mean(d['mcal_g'][msks['2p'], 1])
    g2_2m = np.mean(d['mcal_g'][msks['2m'], 1])
    R11 = (g1_1p - g1_1m) / 2 / 0.01
    R22 = (g2_2p - g2_2m) / 2 / 0.01

    g1 = np.mean(d['mcal_g'][msks['noshear'], 0])
    g2 = np.mean(d['mcal_g'][msks['noshear'], 1])

    return g1, g2, R11, R22


def _measure_m_c(res_p, res_m):
    g1p, g2p, R11p, R22p = _measure_g1g2R(d=res_p, s2n_cut=10, size_cut=1.2)
    g1m, g2m, R11m, R22m = _measure_g1g2R(d=res_m, s2n_cut=10, size_cut=1.2)

    m = (g1p - g1m)/(R11p + R11m)/0.02 - 1
    c = (g2p + g2m)/(R22p + R22m)
    return m, c


def _measure_m_c_bootstrap(res_p, res_m, seed, nboot=100):
    rng = np.random.RandomState(seed=seed)
    marr = []
    carr = []
    for _ in range(nboot):
        inds = rng.choice(len(res_p), size=len(res_p), replace=True)
        _res_p = np.hstack([res_p[i] for i in inds])
        _res_m = np.hstack([res_m[i] for i in inds])
        m, c = _measure_m_c(_res_p, _res_m)
        marr.append(m)
        carr.append(c)

    m, c = _measure_m_c(np.hstack(res_p), np.hstack(res_m))
    return m, np.std(marr), c, np.std(carr)


def test_metacal():
    nsims = 50

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    jobs = [
        joblib.delayed(_run_single_sim_pair)(seed, 1e8)
        for seed in seeds
    ]
    outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)
    res_p = []
    res_m = []
    for res in outputs:
        if res is not None:
            res_p.append(res[0])
            res_m.append(res[1])

    seed = rng.randint(size=nsims, low=1, high=2**29)
    m, merr, c, cerr = _measure_m_c_bootstrap(res_p, res_m, seed, nboot=100)

    print("m: %f +/- %f [1e-3, 3-sigma]" % (m/1e-3, 3*merr/1e-3), flush=True)
    print("c: %f +/- %f [1e-5, 3-sigma]" % (c/1e-5, 3*cerr/1e-5), flush=True)

    assert np.abs(m) < max(5e-4, 3*merr), (m, merr)
    assert np.abs(c) < 4.0*cerr, (c, cerr)


@pytest.mark.slow
def test_metacal_slow():
    nsims = 100_000

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    jobs = [
        joblib.delayed(_run_single_sim_pair)(seed, s2n=20)
        for seed in seeds
    ]
    outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)
    res_p = []
    res_m = []
    for res in outputs:
        if res is not None:
            res_p.append(res[0])
            res_m.append(res[1])

    seed = rng.randint(size=nsims, low=1, high=2**29)
    m, merr, c, cerr = _measure_m_c_bootstrap(res_p, res_m, seed, nboot=100)

    print("m: %f +/- %f [1e-3, 3-sigma]" % (m/1e-3, 3*merr/1e-3), flush=True)
    print("c: %f +/- %f [1e-5, 3-sigma]" % (c/1e-5, 3*cerr/1e-5), flush=True)

    assert np.abs(m) < max(5e-4, 3*merr), (m, merr)
    assert np.abs(c) < 4.0*cerr, (c, cerr)
