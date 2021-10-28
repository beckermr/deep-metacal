import galsim
import numpy as np


def get_gauss_reconv_psf(psf, flux=1):
    """Gets the target reconvolution PSF for an input PSF object.

    This is taken from galsim/tests/test_metacal.py and assumes the psf is
    centered.

    Parameters
    ----------
    psf : galsim object
        The PSF.
    flux : float
        The output flux of the PSF. Defaults to 1.

    Returns
    -------
    reconv_psf : galsim object
        The reconvolution PSF.
    """
    dk = psf.stepk/4.0

    small_kval = 1.e-2    # Find the k where the given psf hits this kvalue
    smaller_kval = 3.e-3  # Target PSF will have this kvalue at the same k

    kim = psf.drawKImage(scale=dk)
    karr_r = kim.real.array
    # Find the smallest r where the kval < small_kval
    nk = karr_r.shape[0]
    kx, ky = np.meshgrid(np.arange(-nk/2, nk/2), np.arange(-nk/2, nk/2))
    ksq = (kx**2 + ky**2) * dk**2
    ksq_max = np.min(ksq[karr_r < small_kval * psf.flux])

    # We take our target PSF to be the (round) Gaussian that is even smaller at
    # this ksq
    # exp(-0.5 * ksq_max * sigma_sq) = smaller_kval
    sigma_sq = -2. * np.log(smaller_kval) / ksq_max

    return galsim.Gaussian(sigma=np.sqrt(sigma_sq), flux=flux)


def _render_psf_and_build_obs(image, obs, reconv_psf, weight_fac=1):
    pim = reconv_psf.drawImage(
        nx=obs.psf.image.shape[1],
        ny=obs.psf.image.shape[0],
        wcs=obs.psf.jacobian.get_galsim_wcs(),
        center=galsim.PositionD(
            x=obs.psf.jacobian.get_col0(),
            y=obs.psf.jacobian.get_row0(),
        ),
    ).array
    psf_obs = obs.psf.copy()
    psf_obs.image = pim
    obs = obs.copy()
    obs.image = image
    obs.psf = psf_obs
    obs.weight = obs.weight * weight_fac
    return obs


def metacal_op(obs, reconv_psf, g1, g2):
    """Run metacal on an ngmix observation."""
    wcs = obs.jacobian.get_galsim_wcs()
    image = get_galsim_object_from_ngmix_obs(obs, kind="image")
    noise = get_galsim_object_from_ngmix_obs(obs, kind="noise", rot90=1)
    psf = get_galsim_object_from_ngmix_obs(obs.psf, kind="image")
    psf_inv = galsim.Deconvolve(psf)

    ims = galsim.Convolve([
        galsim.Convolve([image, psf_inv]).shear(g1=g1, g2=g2),
        reconv_psf,
    ])

    ns = galsim.Convolve([
        galsim.Convolve([noise, psf_inv]).shear(g1=g1, g2=g2),
        reconv_psf,
    ])

    ims = ims.drawImage(nx=obs.image.shape[1], ny=obs.image.shape[0], wcs=wcs).array
    ns = np.rot90(
        ns.drawImage(nx=obs.image.shape[1], ny=obs.image.shape[0], wcs=wcs).array,
        k=-1,
    )
    return _render_psf_and_build_obs(ims+ns, obs, reconv_psf, weight_fac=0.5)


def match_psf(obs, reconv_psf):
    """Match the PSF on an ngmix observation to a new PSF."""
    wcs = obs.jacobian.get_galsim_wcs()
    image = get_galsim_object_from_ngmix_obs(obs, kind="image")
    psf = get_galsim_object_from_ngmix_obs(obs.psf, kind="image")

    ims = galsim.Convolve([image, galsim.Deconvolve(psf), reconv_psf])
    ims = ims.drawImage(nx=obs.image.shape[1], ny=obs.image.shape[0], wcs=wcs).array

    return _render_psf_and_build_obs(ims, obs, reconv_psf, weight_fac=1)


def add_ngmix_obs(obs1, obs2):
    """Add two ngmix observations"""
    obs = obs1.copy()
    obs.image = obs1.image + obs2.image
    msk = (obs1.weight > 0) & (obs2.weight > 0)
    new_wgt = np.zeros_like(obs1.weight)
    new_wgt[msk] = 1/(1/obs1.weight[msk] + 1/obs2.weight[msk])
    obs.weight = new_wgt
    return obs


def get_galsim_object_from_ngmix_obs(obs, kind="image", rot90=0):
    """Make an interpolated image from an ngmix obs."""
    return galsim.InterpolatedImage(
        galsim.ImageD(
            np.rot90(getattr(obs, kind).copy(), k=rot90),
            wcs=obs.jacobian.get_galsim_wcs(),
        ),
        x_interpolant="lanczos15",
    )


def get_galsim_object_from_ngmix_obs_nopix(obs, kind="image"):
    """Make an interpolated image from an ngmix obs w/o a pixel."""
    wcs = obs.jacobian.get_galsim_wcs()
    return galsim.Convolve([
        get_galsim_object_from_ngmix_obs(obs, kind=kind),
        galsim.Deconvolve(wcs.toWorld(galsim.Pixel(scale=1))),
    ])
