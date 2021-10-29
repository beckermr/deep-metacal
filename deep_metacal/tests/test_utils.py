import numpy as np
import pytest

from ..utils import estimate_m_and_c, cut_nones


@pytest.mark.parametrize('swap12', [True, False])
@pytest.mark.parametrize('step', [0.005, 0.01])
@pytest.mark.parametrize('g_true', [0.05, 0.01, 0.02])
@pytest.mark.parametrize('jackknife', [None, 100])
def test_estimate_m_and_c(g_true, step, swap12, jackknife):
    rng = np.random.RandomState(seed=10)

    def _shear_meas(g_true, _step, e1, e2):
        if _step == 0:
            _gt = g_true * (1.0 + 0.01)
            cadd = 0.05 * 10
        else:
            _gt = g_true
            cadd = 0.0
        if swap12:
            return np.mean(e1) + cadd + _step*10, np.mean(10*(_gt+_step)+e2)
        else:
            return np.mean(10*(_gt+_step)+e1), np.mean(e2) + cadd + _step*10

    sn = 0.01
    n_gals = 10000
    n_sim = 1000
    pres = []
    mres = []
    for i in range(n_sim):
        e1 = rng.normal(size=n_gals) * sn
        e2 = rng.normal(size=n_gals) * sn

        g1, g2 = _shear_meas(g_true, 0, e1, e2)
        g1p, g2p = _shear_meas(g_true, step, e1, e2)
        g1m, g2m = _shear_meas(g_true, -step, e1, e2)
        pres.append((g1p, g1m, g1, g2p, g2m, g2))

        g1, g2 = _shear_meas(-g_true, 0, e1, e2)
        g1p, g2p = _shear_meas(-g_true, step, e1, e2)
        g1m, g2m = _shear_meas(-g_true, -step, e1, e2)
        mres.append((g1p, g1m, g1, g2p, g2m, g2))
        if i == 0:
            pres[-1] = None

        if i == 250:
            mres[-1] = None

        if i == 750:
            pres[-1] = None
            mres[-1] = None

    m, merr, c, cerr = estimate_m_and_c(
        pres, mres, g_true, swap12=swap12, step=step, jackknife=jackknife,
        silent=True
    )

    assert np.allclose(m, 0.01)
    assert np.allclose(c, 0.05)


@pytest.mark.parametrize('seed', [1, 3, 454, 3454, 23443, 42])
@pytest.mark.parametrize('jackknife', [None, 100])
def test_estimate_m_and_c_err(jackknife, seed):
    g_true = 0.02
    step = 0.01
    swap12 = False

    rng = np.random.RandomState(seed=seed)

    def _shear_meas(g_true, _step, e1, e2):
        if _step == 0:
            _gt = g_true * (1.0 + 0.01)
            cadd = 0.05 * 10
        else:
            _gt = g_true
            cadd = 0.0
        if swap12:
            return np.mean(e1) + cadd + _step*10, np.mean(10*(_gt+_step)+e2)
        else:
            return np.mean(10*(_gt+_step)+e1), np.mean(e2) + cadd + _step*10

    sn = 0.5
    n_gals = 10
    n_sim = 1000
    pres = []
    mres = []
    for i in range(n_sim):
        e1 = rng.normal(size=n_gals) * sn
        e2 = rng.normal(size=n_gals) * sn

        e1p = rng.normal(size=n_gals) * sn
        e2p = rng.normal(size=n_gals) * sn

        e1m = rng.normal(size=n_gals) * sn
        e2m = rng.normal(size=n_gals) * sn

        g1, g2 = _shear_meas(g_true, 0, e1+e1p, e2+e2p)
        g1p, g2p = _shear_meas(g_true, step, e1+e1p, e2+e2p)
        g1m, g2m = _shear_meas(g_true, -step, e1+e1p, e2+e2p)
        pres.append((g1p, g1m, g1, g2p, g2m, g2))

        g1, g2 = _shear_meas(-g_true, 0, e1+e1m, e2+e2m)
        g1p, g2p = _shear_meas(-g_true, step, e1+e1m, e2+e2m)
        g1m, g2m = _shear_meas(-g_true, -step, e1+e2m, e2+e2m)
        mres.append((g1p, g1m, g1, g2p, g2m, g2))
        if i == 0:
            pres[-1] = None

        if i == 250:
            mres[-1] = None

        if i == 750:
            pres[-1] = None
            mres[-1] = None

    m, merr, c, cerr = estimate_m_and_c(
        pres, mres, g_true, swap12=swap12, step=step,
        jackknife=jackknife, silent=True
    )

    assert np.abs(m - 0.01) <= 3*merr, (m, merr)
    assert np.abs(c - 0.05) <= 3*cerr, (c, cerr)


def test_cut_nones():
    pres = [1, None, 2, None, 4]
    mres = [None, 11, 12, None, 14]

    pres, mres = cut_nones(pres, mres)

    assert pres == [2, 4]
    assert mres == [12, 14]
