import numpy as np
import ngmix
import galsim
import joblib
import multiprocessing

import pytest

from ..metacal import metacal_wide_and_deep_psf_matched
from ..utils import fit_mcal_res_gauss_mom, timer


def _make_single_sim(*, rng, psf, obj, nse, dither):
    cen = (73-1)/2
    scale = 0.2

    im = obj.drawImage(nx=73, ny=73, offset=dither, scale=scale).array
    im += rng.normal(size=im.shape, scale=nse)

    psf_im = psf.drawImage(nx=73, ny=73, scale=scale).array

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


def _make_sim(*, seed, g1, g2, deep_noise_fac):
    rng = np.random.RandomState(seed=seed)

    gal = galsim.Exponential(half_light_radius=0.5).shear(g1=g1, g2=g2)
    psf = galsim.Gaussian(fwhm=0.8)
    deep_psf = galsim.Gaussian(fwhm=0.8)
    obj = galsim.Convolve([gal, psf])
    deep_obj = galsim.Convolve([gal, deep_psf])

    # estimate noise level
    dither = np.zeros(2)
    nse = 8e-3

    dither = rng.uniform(size=2, low=-0.5, high=0.5)
    obs_wide = _make_single_sim(
        rng=rng,
        psf=psf,
        obj=obj,
        nse=nse,
        dither=dither,
    )

    obs_deep = _make_single_sim(
        rng=rng,
        psf=deep_psf,
        obj=deep_obj,
        nse=nse * deep_noise_fac,
        dither=dither,
    )

    obs_deep_noise = _make_single_sim(
        rng=rng,
        psf=deep_psf,
        obj=deep_obj.withFlux(0),
        nse=nse * deep_noise_fac,
        dither=dither,
    )

    return obs_wide, obs_deep, obs_deep_noise


def _run_single_sim(
    seed, g1, g2, deep_noise_fac, skip_wide, skip_deep,
):
    obs_w, obs_d, obs_dn = _make_sim(
        seed=seed, g1=g1, g2=g2, deep_noise_fac=deep_noise_fac,
    )
    mcal_res = metacal_wide_and_deep_psf_matched(
        obs_w, obs_d, obs_dn,
        skip_obs_wide_corrections=skip_wide,
        skip_obs_deep_corrections=skip_deep,
    )
    return fit_mcal_res_gauss_mom(mcal_res)


def _run_sim_pair(seed, deep_noise_fac, skip_wide, skip_deep):
    res_p = _run_single_sim(
        seed, 0.02, 0.0, deep_noise_fac, skip_wide, skip_deep,
    )

    res_m = _run_single_sim(
        seed, -0.02, 0.0, deep_noise_fac, skip_wide, skip_deep,
    )

    return res_p, res_m


def _measure_one(data):
    def _msk(d, shear):
        return (
            (d["s2n"] > 10)
            & (d["T_ratio"] > 1.2)
            & np.all(np.isfinite(d["g"]), axis=1)
            & (d["flags"] == 0)
            & (d["shear"] == shear)
        )

    msk = _msk(data, "noshear")
    g1 = np.mean(data["g"][msk, 0])
    g2 = np.mean(data["g"][msk, 1])

    msk = _msk(data, "1p")
    g1p = np.mean(data["g"][msk, 0])
    msk = _msk(data, "1m")
    g1m = np.mean(data["g"][msk, 0])

    msk = _msk(data, "2p")
    g2p = np.mean(data["g"][msk, 1])
    msk = _msk(data, "2m")
    g2m = np.mean(data["g"][msk, 1])

    R11 = (g1p - g1m)/2/0.01
    R22 = (g2p - g2m)/2/0.01

    return R11, g1, R22, g2


def _measure_pair(pdata, mdata):
    R11_p, g1_p, R22_p, g2_p = _measure_one(pdata)
    R11_m, g1_m, R22_m, g2_m = _measure_one(mdata)

    g1 = (g1_p - g1_m)/2
    R11 = (R11_p + R11_m)/2

    g2 = (g2_p + g2_m)/2
    R22 = (R22_p + R22_m)/2

    return g1/R11, g2/R22


def estimate_m_and_c_lists(
    pdata,
    mdata,
    g_true,
    jackknife=100,
    silent=False,
):
    """
    Returns
    -------
    m : float
        Estimate of the multiplicative bias.
    merr : float
        Estimat of the 1-sigma standard error in `m`.
    c : float
        Estimate of the additive bias.
    cerr : float
        Estimate of the 1-sigma standard error in `c`.
    """

    with timer("running jackknife", silent=silent):
        n = pdata.shape[0] // 5

        n_per = n // jackknife
        if n_per < 1:
            n_per = 1
            jackknife = n
        n = n_per * jackknife

        n *= 5
        n_per *= 5
        assert n <= pdata.shape[0]

        pdata = pdata[:n]
        mdata = mdata[:n]
        g1s = np.zeros(jackknife)
        g2s = np.zeros(jackknife)
        loc = 0
        for i in range(jackknife):
            _p = np.delete(pdata, list(range(loc, loc+n_per)), axis=0)
            _m = np.delete(mdata, list(range(loc, loc+n_per)), axis=0)
            _g1, _g2 = _measure_pair(_p, _m)
            g1s[i] = _g1
            g2s[i] = _g2
            loc += n_per

        g1, g2 = _measure_pair(pdata, mdata)
        g1err = np.sqrt((n - n_per) / n * np.sum((g1s-g1)**2))
        g2err = np.sqrt((n - n_per) / n * np.sum((g2s-g2)**2))

    m, merr, c, cerr = (
        g1/g_true-1,
        g1err/g_true,
        g2,
        g2err,
    )

    print("# of sims:", n // 5, flush=True)
    print("m: %f +/- %f [1e-3, 3-sigma]" % (m/1e-3, 3*merr/1e-3), flush=True)
    print("c: %f +/- %f [1e-5, 3-sigma]" % (c/1e-5, 3*cerr/1e-5), flush=True)

    return m, merr, c, cerr


@pytest.mark.slowdeepmdet
@pytest.mark.parametrize("skip_wide,skip_deep", [
    (True, False),
    (False, True),
    (True, True),
])
def test_deep_metacal_slow_terms(skip_wide, skip_deep):
    nsims = 100_000
    chunk_size = multiprocessing.cpu_count() * 100
    nchunks = int(np.ceil(nsims // chunk_size))
    noise_fac = 1/np.sqrt(10)
    nsims = nchunks * chunk_size

    rng = np.random.RandomState(seed=4243562)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    res_p = None
    res_m = None
    loc = 0
    for chunk in range(nchunks):
        with timer("running chunk %d of %d" % (chunk+1, nchunks)):
            _seeds = seeds[loc:loc + chunk_size]
            jobs = [
                joblib.delayed(_run_sim_pair)(seed, noise_fac, skip_wide, skip_deep)
                for seed in _seeds
            ]
            outputs = joblib.Parallel(n_jobs=-1, verbose=0)(jobs)

        with timer("collecting results"):
            _res_p = []
            _res_m = []
            for _p, _m in outputs:
                _res_p.append(_p)
                _res_m.append(_m)

            _res_p = np.concatenate(_res_p, axis=0)
            _res_m = np.concatenate(_res_m, axis=0)

            if res_p is None:
                res_p = _res_p
                res_m = _res_m
            else:
                res_p = np.concatenate([res_p, _res_p], axis=0)
                res_m = np.concatenate([res_m, _res_m], axis=0)

        m, merr, c, cerr = estimate_m_and_c_lists(
            res_p,
            res_m,
            0.02,
            jackknife=100,
        )

        loc += chunk_size

    assert np.abs(m) >= max(5e-4, 3*merr), (m, merr)
    assert np.abs(c) < 4.0*cerr, (c, cerr)
