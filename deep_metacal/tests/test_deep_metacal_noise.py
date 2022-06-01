import numpy as np
import ngmix
import galsim


from ..metacal import metacal_wide_and_deep_psf_matched, metacal_op_shears
from ..utils import fit_mcal_res_gauss_mom


def _make_single_sim(*, rng, psf, obj, nse, dither):
    cen = (53-1)/2
    scale = 0.263

    im = obj.drawImage(nx=53, ny=53, offset=dither, scale=scale).array
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


def _make_sim(*, seed, g1, g2, s2n, deep_noise_fac, deep_psf_fac, zero_flux):
    rng = np.random.RandomState(seed=seed)

    gal = galsim.Exponential(half_light_radius=0.7).shear(g1=g1, g2=g2)
    psf = galsim.Gaussian(fwhm=0.8)
    deep_psf = galsim.Gaussian(fwhm=0.8*deep_psf_fac)
    obj = galsim.Convolve([gal, psf])
    deep_obj = galsim.Convolve([gal, deep_psf])

    # estimate noise level
    dither = np.zeros(2)
    scale = 0.263
    im = obj.drawImage(nx=53, ny=53, offset=dither, scale=scale).array
    nse = np.sqrt(np.sum(im**2)) / s2n

    if zero_flux:
        obj = obj.withFlux(0)

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
    seed, s2n, g1, g2, deep_noise_fac, deep_psf_fac, use_mcal, zero_flux,
):
    obs_w, obs_d, obs_dn = _make_sim(
        seed=seed, g1=g1, g2=g2, s2n=s2n, deep_noise_fac=deep_noise_fac,
        deep_psf_fac=deep_psf_fac, zero_flux=zero_flux,
    )
    if use_mcal:
        mcal_res = metacal_op_shears(
            obs_w,
        )
    else:
        mcal_res = metacal_wide_and_deep_psf_matched(
            obs_w, obs_d, obs_dn,
        )
    return fit_mcal_res_gauss_mom(mcal_res), mcal_res


def test_deep_metacal_noise():
    nsims = 10
    noise_fac = 1/np.sqrt(10)
    s2n = 10

    rng = np.random.RandomState(seed=34132)
    seeds = rng.randint(size=nsims, low=1, high=2**29)

    dmcal_res = []
    mcal_res = []
    for seed in seeds:
        dmcal_res.append(_run_single_sim(
            seed, s2n, 0.02, 0.0, noise_fac, 1.0, False, False,
        ))
        mcal_res.append(_run_single_sim(
            seed, s2n, 0.02, 0.0, noise_fac, 1.0, True, False,
        ))

    dmcal_res = np.concatenate([d[0] for d in dmcal_res if d is not None], axis=0)
    mcal_res = np.concatenate([d[0] for d in mcal_res if d is not None], axis=0)
    dmcal_res = dmcal_res[dmcal_res["shear"] == "noshear"]
    mcal_res = mcal_res[mcal_res["shear"] == "noshear"]

    ratio = (np.median(dmcal_res["s2n"])/np.median(mcal_res["s2n"]))**2
    print("s2n ratio squared:", ratio)
    assert np.allclose(ratio, 2, atol=0, rtol=0.2), ratio

    dmcal_res = []
    mcal_res = []
    for seed in seeds:
        dmcal_res.append(_run_single_sim(
            seed, s2n, 0.02, 0.0, noise_fac, 1.0, False, True,
        ))
        mcal_res.append(_run_single_sim(
            seed, s2n, 0.02, 0.0, noise_fac, 1.0, True, True,
        ))

    dmcal_res = np.array([
        np.std(d[1]["noshear"].image)
        for d in dmcal_res if d is not None
    ])
    mcal_res = np.array([
        np.std(d[1]["noshear"].image)
        for d in mcal_res if d is not None
    ])

    ratio = (np.median(dmcal_res)/np.median(mcal_res))**2
    print("noise ratio squared:", ratio)
    assert np.allclose(ratio, 0.5, atol=0, rtol=0.2), ratio
