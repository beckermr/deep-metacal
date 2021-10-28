import ngmix
import numpy as np
import galsim

from ..pixel_cov import meas_pixel_cov
from esutil.pbar import prange

from ..metacal import (
    metacal_op,
    match_psf,
    get_gauss_reconv_psf,
    add_ngmix_obs,
)


def _get_reconv_psf(psf_w, psf_d):
    mc_psf_w = get_gauss_reconv_psf(psf_w)
    mc_psf_d = get_gauss_reconv_psf(psf_d)
    if mc_psf_w.fwhm > mc_psf_d.fwhm:
        return mc_psf_w
    else:
        return mc_psf_d


def _make_ngmix_obs(*, img, psf, dim, scale, nse_img, nse_level):
    cen = (dim-1)/2
    jac = ngmix.DiagonalJacobian(
        scale=scale,
        row=cen,
        col=cen,
    )
    psf_obs = ngmix.Observation(
        image=psf,
        jacobian=jac,
        weight=np.ones_like(img),
    )
    obs = ngmix.Observation(
        image=img,
        jacobian=jac,
        weight=np.ones_like(img) / nse_level**2,
        noise=nse_img,
        psf=psf_obs,
    )
    return obs


def _simple_noise_sim():
    scale = 0.263
    wide_noise = 1
    deep_noise = wide_noise / np.sqrt(10)
    dim = 53
    seed = None

    psf_w = galsim.Gaussian(fwhm=0.8)
    psf_d = galsim.Gaussian(fwhm=0.9)
    reconv_psf = _get_reconv_psf(psf_w, psf_d)
    psf_w_img = psf_w.drawImage(nx=dim, ny=dim, scale=scale).array
    psf_d_img = psf_d.drawImage(nx=dim, ny=dim, scale=scale).array

    rng = np.random.RandomState(seed=seed)

    nse_d_obs = _make_ngmix_obs(
        img=rng.normal(size=(dim, dim), scale=deep_noise),
        psf=psf_d_img,
        dim=dim,
        scale=scale,
        nse_img=rng.normal(size=(dim, dim), scale=deep_noise),
        nse_level=deep_noise,
    )

    gal_d_obs = _make_ngmix_obs(
        img=rng.normal(size=(dim, dim), scale=deep_noise),
        psf=psf_d_img,
        dim=dim,
        scale=scale,
        nse_img=rng.normal(size=(dim, dim), scale=deep_noise),
        nse_level=deep_noise,
    )

    nse_w_obs = _make_ngmix_obs(
        img=rng.normal(size=(dim, dim), scale=wide_noise),
        psf=psf_w_img,
        dim=dim,
        scale=scale,
        nse_img=rng.normal(size=(dim, dim), scale=wide_noise),
        nse_level=wide_noise,
    )
    gal_w_obs = _make_ngmix_obs(
        img=rng.normal(size=(dim, dim), scale=wide_noise),
        psf=psf_w_img,
        dim=dim,
        scale=scale,
        nse_img=rng.normal(size=(dim, dim), scale=wide_noise),
        nse_level=wide_noise,
    )

    # do wide image
    mwide = add_ngmix_obs(
        match_psf(gal_w_obs, reconv_psf),
        metacal_op(nse_d_obs, reconv_psf, 0, 0)
    )

    # do deep image
    mdeep = add_ngmix_obs(
        metacal_op(gal_d_obs, reconv_psf, 0, 0),
        match_psf(nse_w_obs, reconv_psf),
    )

    mwide_mcal = metacal_op(gal_w_obs, reconv_psf, 0, 0)

    return {
        "r_wide": mwide,
        "r_deep": mdeep,
        "mcal_wide": mwide_mcal,
    }


def test_noise_handling():
    print(flush=True)
    covs = {}
    for _ in prange(1000):
        res = _simple_noise_sim()
        for k, v in res.items():
            if k not in covs:
                covs[k] = []
            covs[k].append(
                meas_pixel_cov(v.image.copy(), np.ones_like(v.image).astype(bool))
            )

    for k in covs:
        covs[k] = np.mean(covs[k], axis=0)

    assert covs["mcal_wide"][1, 1] > covs["r_wide"][1, 1]*1.4
    assert np.allclose(covs["r_wide"], covs["r_deep"], rtol=0, atol=7e-4)
