import ngmix
import numpy as np
import galsim

from ..pixel_cov import meas_pixel_cov
from esutil.pbar import prange

from ..metacal import (
    metacal_op_g1g2,
    match_psf,
    add_ngmix_obs,
    get_max_gauss_reconv_psf_galsim,
    metacal_wide_and_deep_psf_matched,
)


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
    psf_d = galsim.Gaussian(fwhm=0.8)
    reconv_psf = get_max_gauss_reconv_psf_galsim(psf_w, psf_d)
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

    gal_w_obs = _make_ngmix_obs(
        img=rng.normal(size=(dim, dim), scale=wide_noise),
        psf=psf_w_img,
        dim=dim,
        scale=scale,
        nse_img=rng.normal(size=(dim, dim), scale=wide_noise),
        nse_level=wide_noise,
    )

    if False:
        # this is older code that is now inside metacal_wide_and_deep_psf_matched
        # I am keeping it here for debugging just in case
        nse_w_obs = _make_ngmix_obs(
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
            metacal_op_g1g2(nse_d_obs, reconv_psf, 0, 0)
        )

        # do deep image
        mdeep = add_ngmix_obs(
            metacal_op_g1g2(gal_d_obs, reconv_psf, 0, 0),
            match_psf(nse_w_obs, reconv_psf),
        )

        mwide_mcal = metacal_op_g1g2(gal_w_obs, reconv_psf, 0, 0)

        return {
            "r_wide": mwide,
            "r_deep": mdeep,
            "mcal_wide": mwide_mcal,
        }
    else:
        mcal_res = metacal_wide_and_deep_psf_matched(
            gal_w_obs, gal_d_obs, nse_d_obs, shears=["noshear"]
        )

        mwide_mcal = metacal_op_g1g2(
            gal_w_obs, mcal_res["noshear"].psf.galsim_obj, 0, 0
        )

    return {
        "noshear": mcal_res["noshear"],
        "noshear_deep": mcal_res["noshear_deep"],
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

    for k, v in covs.items():
        print("%s:\n" % k, v, flush=True)

    assert covs["mcal_wide"][1, 1] > covs["noshear"][1, 1]*1.5
    assert np.allclose(covs["noshear"], covs["noshear_deep"], rtol=0, atol=7e-4)
