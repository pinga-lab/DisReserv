import numpy as np
from numpy.testing import assert_almost_equal as aae
import pytest
import geertsma as ge


# Integrals of Bessel functions
def test_Int1():
    'compare with reference results'
    r1 = 0.2
    q1 = 0.4
    R1 = 1.2
    r2 = 0.4
    q2 = 0.4
    R2 = 1.0
    reference = np.array([0.0595689, 0.162188])
    computed = np.array([ge.Int1(q1, r1, R1), ge.Int1(q2, r2, R2)])
    aae(reference, computed, decimal=6)


def test_Int2():
    'compare with reference results'
    r1 = 0.2
    q1 = 0.4
    R1 = 1.2
    r2 = 0.4
    q2 = 0.4
    R2 = 1.0
    reference = np.array([0.0461035, 0.197103])
    computed = np.array([ge.Int2(q1, r1, R1), ge.Int2(q2, r2, R2)])
    aae(reference, computed, decimal=6)


def test_Int3():
    'compare with reference results'
    r1 = 0.2
    q1 = 0.4
    R1 = 1.2
    r2 = 0.4
    q2 = 0.4
    R2 = 1.0
    reference = np.array([0.565282, 0.592418])
    computed = np.array([ge.Int3(q1, r1, R1), ge.Int3(q2, r2, R2)])
    aae(reference, computed, decimal=6)


def test_Int4():
    'compare with reference results'
    r1 = 0.2
    q1 = 0.4
    R1 = 1.2
    r2 = 0.4
    q2 = 0.4
    R2 = 1.0
    reference = np.array([0.598431, 0.818683])
    computed = np.array([ge.Int4(q1, r1, R1), ge.Int4(q2, r2, R2)])
    aae(reference, computed, decimal=6)


def test_Int6():
    'compare with reference results'
    r1 = 0.2
    q1 = 0.4
    R1 = 1.2
    r2 = 0.4
    q2 = 0.4
    R2 = 1.0
    reference = np.array([0.477736, 1.15435])
    computed = np.array([ge.Int6(q1, r1, R1), ge.Int6(q2, r2, R2)])
    aae(reference, computed, decimal=6)
