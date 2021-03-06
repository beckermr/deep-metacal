import functools
import fitsio
import os
import time
from contextlib import contextmanager
import sys

import numpy as np
from esutil.pbar import prange
from ngmix.gaussmom import GaussMom

from .metacal import DEFAULT_SHEARS

GLOBAL_START_TIME = time.time()


@functools.lru_cache()
def cached_descwl_catalog_read():
    fname = os.path.join(os.environ["CATSIM_DIR"], "OneDegSq.fits")
    cat = fitsio.read(fname)
    cut = cat['r_ab'] < 26.0
    cat = cat[cut]
    return cat


def measure_mcal_shear_quants(data, s2n_cut=10, t_ratio_cut=1.2):
    """Measure metacal shear results.

    Parameters
    ----------
    data : array
        An array of the metacal data.
    s2n_cut : float, optional
        The cut in S/N. Default is 10.
    t_ratio_cut : float, optional
        The cut in T_ratio. Default is 1.2.

    Results
    -------
    shear_res : 6-tuple
        A tuple of g1p, g1m, g1, g2p, g2m, g2.
    """
    def _msk_it(*, d, shear):
        return (
            (d["shear"] == shear) &
            (d["flags"] == 0) &
            (d["s2n"] > s2n_cut) &
            (d["T_ratio"] > t_ratio_cut)
        )

    msks = {}
    for shear in DEFAULT_SHEARS:
        msks[shear] = _msk_it(d=data, shear=shear)
        if not np.any(msks[shear]):
            return None

    g1p = np.mean(data['g'][msks['1p'], 0])
    g1m = np.mean(data['g'][msks['1m'], 0])
    g2p = np.mean(data['g'][msks['2p'], 1])
    g2m = np.mean(data['g'][msks['2m'], 1])
    g1 = np.mean(data['g'][msks['noshear'], 0])
    g2 = np.mean(data['g'][msks['noshear'], 1])

    return g1p, g1m, g1, g2p, g2m, g2


def fit_mcal_res_gauss_mom(mcal_res):
    """Fit a metacal result using Gaussian moments.

    Parameters
    ----------
    mcal_res : dict
        The metacal result.

    Returns
    -------
    data : array
        The fit results
    """
    fitter = GaussMom(1.2)
    psf_res = fitter.go(mcal_res["noshear"].psf)
    if psf_res["flags"] != 0:
        return None

    dt = [
        ("flags", "i4"),
        ("g", "f8", (2,)),
        ("T_ratio", "f8"),
        ("s2n", "f8"),
        ("shear", "U7"),
    ]
    vals = []
    for shear, obs in mcal_res.items():
        if shear not in DEFAULT_SHEARS:
            continue
        res = fitter.go(obs)
        vals.append(
            (res["flags"], res["e"], res["T"]/psf_res["T"], res["s2n"], shear)
        )
    return np.array(vals, dtype=dt)


@contextmanager
def timer(name, silent=False):
    t0 = time.time()
    if not silent:
        print(
            "[% 8ds] %s" % (t0 - GLOBAL_START_TIME, name),
            flush=True,
            file=sys.stderr,
        )
    yield
    t1 = time.time()
    if not silent:
        print(
            "[% 8ds] %s done (%f seconds)" % (
                t1 - GLOBAL_START_TIME,
                name,
                t1 - t0
            ),
            flush=True,
            file=sys.stderr,
        )


def cut_nones(presults, mresults):
    """Cut entries that are None in a pair of lists. Any entry that is None
    in either list will exclude the item in the other.

    Parameters
    ----------
    presults : list
        One the list of things.
    mresults : list
        The other list of things.

    Returns
    -------
    pcut : list
        The cut list.
    mcut : list
        The cut list.
    """
    prr_keep = []
    mrr_keep = []
    for pr, mr in zip(presults, mresults):
        if pr is None or mr is None:
            continue
        prr_keep.append(pr)
        mrr_keep.append(mr)

    return prr_keep, mrr_keep


def _run_boostrap(x1, y1, x2, y2, wgts, silent):
    rng = np.random.RandomState(seed=100)
    mvals = []
    cvals = []
    if silent:
        itrl = range(500)
    else:
        itrl = prange(500, desc='running bootstrap')
    for _ in itrl:
        ind = rng.choice(len(y1), replace=True, size=len(y1))
        _wgts = wgts[ind].copy()
        _wgts /= np.sum(_wgts)
        mvals.append(np.mean(y1[ind] * _wgts) / np.mean(x1[ind] * _wgts) - 1)
        cvals.append(np.mean(y2[ind] * _wgts) / np.mean(x2[ind] * _wgts))

    return (
        np.mean(y1 * wgts) / np.mean(x1 * wgts) - 1, np.std(mvals),
        np.mean(y2 * wgts) / np.mean(x2 * wgts), np.std(cvals))


def _run_jackknife(x1, y1, x2, y2, wgts, jackknife):
    n_per = x1.shape[0] // jackknife
    n = n_per * jackknife
    x1j = np.zeros(jackknife)
    y1j = np.zeros(jackknife)
    x2j = np.zeros(jackknife)
    y2j = np.zeros(jackknife)
    wgtsj = np.zeros(jackknife)

    loc = 0
    for i in range(jackknife):
        wgtsj[i] = np.sum(wgts[loc:loc+n_per])
        x1j[i] = np.sum(x1[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        y1j[i] = np.sum(y1[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        x2j[i] = np.sum(x2[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        y2j[i] = np.sum(y2[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]

        loc += n_per

    mbar = np.mean(y1[:n] * wgts[:n]) / np.mean(x1[:n] * wgts[:n]) - 1
    cbar = np.mean(y2[:n] * wgts[:n]) / np.mean(x2[:n] * wgts[:n])
    mvals = np.zeros(jackknife)
    cvals = np.zeros(jackknife)
    for i in range(jackknife):
        _wgts = np.delete(wgtsj, i)
        mvals[i] = (
            np.sum(np.delete(y1j, i) * _wgts) / np.sum(np.delete(x1j, i) * _wgts)
            - 1
        )
        cvals[i] = (
            np.sum(np.delete(y2j, i) * _wgts) / np.sum(np.delete(x2j, i) * _wgts)
        )

    return (
        mbar,
        np.sqrt((n - n_per) / n * np.sum((mvals-mbar)**2)),
        cbar,
        np.sqrt((n - n_per) / n * np.sum((cvals-cbar)**2)),
    )


def estimate_m_and_c(
    presults,
    mresults,
    g_true,
    swap12=False,
    step=0.01,
    weights=None,
    jackknife=None,
    silent=False,
):
    """Estimate m and c from paired lensing simulations.

    Parameters
    ----------
    presults : list of iterables or np.ndarray
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a `g1` shear in the 1-component and
        0 true shear in the 2-component. If an array, it should have the named
        columns.
    mresults : list of iterables or np.ndarray
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a -`g1` shear in the 1-component and
        0 true shear in the 2-component. If an array, it should have the named
        columns.
    g_true : float
        The true value of the shear on the 1-axis in the simulation. The other
        axis is assumd to havea true value of zero.
    swap12 : bool, optional
        If True, swap the roles of the 1- and 2-axes in the computation.
    step : float, optional
        The step used in metadetect for estimating the response. Default is
        0.01.
    weights : list of weights, optional
        Weights to apply to each sample. Will be normalized if not already.
    jackknife : int, optional
        The number of jackknife sections to use for error estimation. Default of
        None will do no jackknife and default to bootstrap error bars.
    silent : bool, optional
        If True, do not print to stderr/stdout.

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

    with timer("prepping data for m,c measurement", silent=silent):
        if isinstance(presults, list) or isinstance(mresults, list):
            prr_keep, mrr_keep = cut_nones(presults, mresults)

            def _get_stuff(rr):
                _a = np.vstack(rr)
                g1p = _a[:, 0]
                g1m = _a[:, 1]
                g1 = _a[:, 2]
                g2p = _a[:, 3]
                g2m = _a[:, 4]
                g2 = _a[:, 5]

                if swap12:
                    g1p, g1m, g1, g2p, g2m, g2 = g2p, g2m, g2, g1p, g1m, g1

                return (
                    g1, (g1p - g1m) / 2 / step * g_true,
                    g2, (g2p - g2m) / 2 / step)

            g1p, R11p, g2p, R22p = _get_stuff(prr_keep)
            g1m, R11m, g2m, R22m = _get_stuff(mrr_keep)
        else:
            if swap12:
                g1p = presults["g2"]
                R11p = (presults["g2p"] - presults["g2m"]) / 2 / step * g_true
                g2p = presults["g1"]
                R22p = (presults["g1p"] - presults["g1m"]) / 2 / step

                g1m = mresults["g2"]
                R11m = (mresults["g2p"] - mresults["g2m"]) / 2 / step * g_true
                g2m = mresults["g1"]
                R22m = (mresults["g1p"] - mresults["g1m"]) / 2 / step
            else:
                g1p = presults["g1"]
                R11p = (presults["g1p"] - presults["g1m"]) / 2 / step * g_true
                g2p = presults["g2"]
                R22p = (presults["g2p"] - presults["g2m"]) / 2 / step

                g1m = mresults["g1"]
                R11m = (mresults["g1p"] - mresults["g1m"]) / 2 / step * g_true
                g2m = mresults["g2"]
                R22m = (mresults["g2p"] - mresults["g2m"]) / 2 / step

        if weights is not None:
            wgts = np.array(weights).astype(np.float64)
        else:
            wgts = np.ones(len(g1p)).astype(np.float64)
        wgts /= np.sum(wgts)

        msk = (
            np.isfinite(g1p) &
            np.isfinite(R11p) &
            np.isfinite(g1m) &
            np.isfinite(R11m) &
            np.isfinite(g2p) &
            np.isfinite(R22p) &
            np.isfinite(g2m) &
            np.isfinite(R22m))
        g1p = g1p[msk]
        R11p = R11p[msk]
        g1m = g1m[msk]
        R11m = R11m[msk]
        g2p = g2p[msk]
        R22p = R22p[msk]
        g2m = g2m[msk]
        R22m = R22m[msk]
        wgts = wgts[msk]

        x1 = (R11p + R11m)/2
        y1 = (g1p - g1m) / 2

        x2 = (R22p + R22m) / 2
        y2 = (g2p + g2m) / 2

    if jackknife:
        with timer("running jackknife", silent=silent):
            return _run_jackknife(x1, y1, x2, y2, wgts, jackknife)
    else:
        with timer("running bootstrap", silent=silent):
            return _run_boostrap(x1, y1, x2, y2, wgts, silent)
