import numpy as np
import ngmix
import galsim
import joblib
import multiprocessing

import pytest

from ..metacal import metacal_op_shears
from ..utils import (
    estimate_m_and_c, measure_mcal_shear_quants, fit_mcal_res_gauss_mom,
)


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


def _run_single_sim_pair(seed, s2n):
    obs_plus = _make_sim(seed=seed, g1=0.02, g2=0.0, s2n=s2n)
    mcal_res = metacal_op_shears(obs_plus)
    res_p = fit_mcal_res_gauss_mom(mcal_res)
    if res_p is None or np.any(res_p["flags"] != 0):
        return None
    res_p = measure_mcal_shear_quants(res_p)
    if res_p is None:
        return None

    obs_minus = _make_sim(seed=seed, g1=-0.02, g2=0.0, s2n=s2n)
    mcal_res = metacal_op_shears(obs_minus)
    res_m = fit_mcal_res_gauss_mom(mcal_res)
    if res_m is None or np.any(res_m["flags"] != 0):
        return None
    res_m = measure_mcal_shear_quants(res_m)
    if res_m is None:
        return None

    return res_p, res_m


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

    m, merr, c, cerr = estimate_m_and_c(res_p, res_m, 0.02, jackknife=len(res_p))

    print("m: %f +/- %f [1e-3, 3-sigma]" % (m/1e-3, 3*merr/1e-3), flush=True)
    print("c: %f +/- %f [1e-5, 3-sigma]" % (c/1e-5, 3*cerr/1e-5), flush=True)

    assert np.abs(m) < max(5e-4, 3*merr), (m, merr)
    assert np.abs(c) < 4.0*cerr, (c, cerr)


@pytest.mark.slow
def test_metacal_slow():
    nsims = 1_000_000
    chunk_size = multiprocessing.cpu_count() * 100
    nchunks = nsims // chunk_size

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = []
    res_m = []
    loc = 0
    for chunk in range(nchunks):
        _seeds = seeds[loc:loc + chunk_size]
        jobs = [
            joblib.delayed(_run_single_sim_pair)(seed, 20)
            for seed in _seeds
        ]
        outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)
        for res in outputs:
            if res is not None:
                res_p.append(res[0])
                res_m.append(res[1])

        if len(res_p) < 500:
            njack = len(res_p)
        else:
            njack = 100

        m, merr, c, cerr = estimate_m_and_c(
            res_p, res_m, 0.02, jackknife=njack,
        )

        print("# of sims:", len(res_p), flush=True)
        print("m: %f +/- %f [1e-3, 3-sigma]" % (m/1e-3, 3*merr/1e-3), flush=True)
        print("c: %f +/- %f [1e-5, 3-sigma]" % (c/1e-5, 3*cerr/1e-5), flush=True)

        loc += chunk_size

    assert np.abs(m) < max(5e-4, 3*merr), (m, merr)
    assert np.abs(c) < 4.0*cerr, (c, cerr)
