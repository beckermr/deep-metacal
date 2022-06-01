import numpy as np
import descwl
import ngmix
import galsim
import fitsio
import joblib

from deep_metacal.metacal import metacal_wide_and_deep_psf_matched, metacal_op_shears
import deep_metacal.utils


def get_survey(bands):
    # for iband, band in enumerate(bands):
    # make the survey and code to build galaxies from it
    pars = descwl.survey.Survey.get_defaults(
        survey_name='LSST',
        filter_band='r',
    )

    pars['survey_name'] = 'LSST'
    pars['filter_band'] = 'r'
    pars['pixel_scale'] = 0.2

    # note in the way we call the descwl package, the image width
    # and height is not actually used
    pars['image_width'] = 73
    pars['image_height'] = 73

    # some versions take in the PSF and will complain if it is not
    # given
    try:
        survey = descwl.survey.Survey(**pars)
    except Exception:
        pars['psf_model'] = None
        survey = descwl.survey.Survey(**pars)

    builder = descwl.model.GalaxyBuilder(
        survey=survey,
        no_disk=False,
        no_bulge=False,
        no_agn=False,
        verbose_model=False)

    noise = np.sqrt(survey.mean_sky_level)
    return survey, builder, noise


def make_ngmix_obs(*, img, psf, dim, scale, nse_img, nse_level):
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


def mdet(
    gal_w0, gal_d0,
    wide_noise=1e-2,
    deep_noise=1e-2/np.sqrt(10),
    fwhm_w=0.9,
    fwhm_d=0.9,
    seed=None,
    use_mcal=False,
):
    scale = 0.2
    dim = 73
    rng = np.random.RandomState(seed=seed)

    psf_w = galsim.Gaussian(fwhm=fwhm_w)
    psf_d = galsim.Gaussian(fwhm=fwhm_d)

    psf_w_img = psf_w.drawImage(nx=dim, ny=dim, scale=scale).array
    psf_d_img = psf_d.drawImage(nx=dim, ny=dim, scale=scale).array

    gal_w = galsim.Convolve(gal_w0, psf_w)
    # use same object here to reduce noise
    gal_d = galsim.Convolve(gal_w0, psf_d)

    gal_d_img = gal_d.drawImage(nx=dim, ny=dim, scale=scale).array
    gal_w_img = gal_w.drawImage(nx=dim, ny=dim, scale=scale).array

    nse_d_obs = make_ngmix_obs(
        img=rng.normal(size=(dim, dim), scale=deep_noise),
        psf=psf_d_img,
        dim=dim,
        scale=scale,
        nse_img=rng.normal(size=(dim, dim), scale=deep_noise),
        nse_level=deep_noise,
    )

    gal_d_obs = make_ngmix_obs(
        img=gal_d_img + rng.normal(size=(dim, dim), scale=deep_noise),
        psf=psf_d_img,
        dim=dim,
        scale=scale,
        nse_img=rng.normal(size=(dim, dim), scale=deep_noise),
        nse_level=deep_noise,
    )

    # nse_w_obs = make_ngmix_obs(
    #     img=rng.normal(size=(dim, dim), scale=wide_noise),
    #     psf=psf_w_img,
    #     dim=dim,
    #     scale=scale,
    #     nse_img=rng.normal(size=(dim, dim), scale=wide_noise),
    #     nse_level=wide_noise,
    # )
    gal_w_obs = make_ngmix_obs(
        img=gal_w_img + rng.normal(size=(dim, dim), scale=wide_noise),
        psf=psf_w_img,
        dim=dim,
        scale=scale,
        nse_img=rng.normal(size=(dim, dim), scale=wide_noise),
        nse_level=wide_noise,
    )

    mcal_res = metacal_op_shears(
        gal_w_obs,
        shears=['noshear', '1p', '1m', '2p', '2m'],
    )
    dmcal_res = metacal_wide_and_deep_psf_matched(
        gal_w_obs, gal_d_obs, nse_d_obs,
        shears=['noshear', '1p', '1m', '2p', '2m'],
    )

    return (
        deep_metacal.utils.fit_mcal_res_gauss_mom(mcal_res),
        deep_metacal.utils.fit_mcal_res_gauss_mom(dmcal_res),
    )


def worker(seed, use_mcal):
    rng = np.random.RandomState(seed=seed)

    bands = ['r']
    survey, builder, noise_wide = get_survey(bands)

    noise_deep = noise_wide/np.sqrt(10)

    cat = deep_metacal.utils.cached_descwl_catalog_read()

    # build wide galaxy
    cat['pa_disk'] = rng.uniform(
                low=0.0, high=360.0, size=cat.size)
    cat['pa_bulge'] = cat['pa_disk']
    rind = rng.choice(cat.size)
    angle = rng.uniform() * 360
    gal_w = builder.from_catalog(
        cat[rind], 0, 0,
        survey.filter_band
    ).model.rotate(
        angle * galsim.degrees
    )

    # build deep galaxy
    cat['pa_disk'] = rng.uniform(
            low=0.0, high=360.0, size=cat.size)
    cat['pa_bulge'] = cat['pa_disk']
    rind = rng.choice(cat.size)
    angle = rng.uniform() * 360
    gal_d = builder.from_catalog(
        cat[rind], 0, 0,
        survey.filter_band
    ).model.rotate(
        angle * galsim.degrees
    )

    gal_wp = gal_w.shear(g1=0.02, g2=0.0)
    gal_dp = gal_d.shear(g1=0.02, g2=0.0)

    fwhm_w = 0.8
    fwhm_d = 0.8
    sigma_w = noise_wide
    sigma_d = noise_deep

    seed = rng.randint(low=0, high=2**32)

    d_p = mdet(
        gal_wp, gal_dp, wide_noise=sigma_w, deep_noise=sigma_d,
        fwhm_w=fwhm_w, fwhm_d=fwhm_d,
        seed=seed, use_mcal=use_mcal,
    )
    return d_p


if __name__ == "__main__":

    nsims = 100_000
    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)
    jobs = [
        joblib.delayed(worker)(seed, False)
        for seed in seeds
    ]
    outputs = joblib.Parallel(n_jobs=-1, verbose=10)(jobs)

    fitsio.write(
        "mcal.fits",
        np.concatenate([
            out[0] for out in outputs if out[0] is not None
        ], axis=0),
        clobber=True,
    )

    fitsio.write(
        "dmcal.fits",
        np.concatenate([
            out[1] for out in outputs if out[1] is not None
        ], axis=0),
        clobber=True,
    )
