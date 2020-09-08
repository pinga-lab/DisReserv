import numpy as np
import scipy as sp
from numpy.testing import assert_almost_equal as aae
import pytest
import compaction as cp
import geertsma as ge


# volume integral versus Geertsma's model
def test_vol_integral_versus_Geertsma_displacement():
    'compare volume integral and Geertsma model'
    # models
    y1 = -5
    y2 = 5
    x1 = -5
    x2 = 5
    z1 = 200
    z2 = 210
    prism = np.array([y1, y2, x1, x2, z2, z1])
    disk = [
        0.5*(y1+y2), 0.5*(x1+x2), 0.5*(z1+z2), (y2-y1)*np.sqrt(1/np.pi), z2-z1
    ]
    # computation points
    np.random.seed(6)
    yp = -10 + 20*np.random.rand(10)
    xp = -10 + 20*np.random.rand(10)
    zp = np.zeros(10)
    coordinates = np.vstack([yp, xp, zp])

    poisson = 0.025
    young = 1000
    pressure = -10

    # volume integral
    dx1 = cp.displacement_x_component(
        coordinates, prism, pressure, poisson, young
    )
    dy1 = cp.displacement_y_component(
        coordinates, prism, pressure, poisson, young
    )
    dz1 = cp.displacement_z_component(
        coordinates, prism, pressure, poisson, young
    )
    dr1 = np.sqrt(dx1**2 + dy1**2)

    # Geertsma
    dr2, dz2 = ge.Geertsma_displacement(
        coordinates, disk, pressure, poisson, young
    )

    aae(dz1, dz2, decimal=8)
    aae(dr1, dr2, decimal=8)
