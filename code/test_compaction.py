import numpy as np
import scipy as sp
from numpy.testing import assert_almost_equal as aae
import pytest
import compaction as cp
import geertsma_disk as ge


# volume integral versus Geertsma's disk-shaped model
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

    # volume integral (proposed methodology)
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

    # Geertsma's disk-shaped model
    dr2, dz2 = ge.Geertsma_disk_displacement(
        coordinates, disk, pressure, poisson, young
    )

    aae(dz1, dz2, decimal=4)
    aae(dr1, dr2, decimal=4)


def test_null_stress_free_surface():
    'stress must be null at the free surface'
    # points at the free surface
    y = np.linspace(-1000, 1000, 20)
    x = np.linspace(-2500, -1500, 20)
    y, x = np.meshgrid(y, x)
    y = np.ravel(y)
    x = np.ravel(x)
    z = np.zeros_like(x)
    coordinates = np.vstack([y, x, z])

    # reference values
    reference = np.zeros_like(x)

    # Poisson ratio and Youg modulus
    poisson = 0.25
    young = 3300

    # model
    model = np.array([[-100, 0, 100, 250, 350, 300]])
    pressure = np.zeros_like(model.shape[0]) - 10

    # stress components
    sx = cp.stress_x_component(
        coordinates, model, pressure, poisson, young
    )
    sy = cp.stress_y_component(
        coordinates, model, pressure, poisson, young
    )
    sz = cp.stress_z_component(
        coordinates, model, pressure, poisson, young
    )
    aae(sx, reference, decimal=10)
    aae(sy, reference, decimal=10)
    aae(sz, reference, decimal=10)


def test_bad_prisms():
    'must stop if prisms have bad arguments'
    # dummy computation points
    y = np.zeros(10)
    x = np.zeros(10)
    z = np.zeros(10)
    coordinates = np.vstack([y, x, z])

    # Poisson ratio and Youg modulus
    poisson = 0.25
    young = 3300

    model1 = np.array([[-100, 0, 100, 250, 300, 350]])
    model2 = np.array([[-100, 0, 250, 100, 350, 300]])
    model3 = np.array([[0, -100, 100, 250, 350, 300]])
    models = [model1, model2, model3]
    pressure = np.zeros(1)
    for modeli in models:
        with pytest.raises(ValueError):
            cp.stress_x_component(
                coordinates, modeli, pressure, poisson, young
            )
            cp.stress_y_component(
                coordinates, modeli, pressure, poisson, young
            )
            cp.stress_z_component(
                coordinates, modeli, pressure, poisson, young
            )
            cp.displacement_x_component(
                coordinates, modeli, pressure, poisson, young
            )
            cp.displacement_y_component(
                coordinates, modeli, pressure, poisson, young
            )
            cp.displacement_z_component(
                coordinates, modeli, pressure, poisson, young
            )


def test_bad_pressure():
    'must stop if pressure size not equal to number of prisms'
    # dummy computation points
    y = np.zeros(10)
    x = np.zeros(10)
    z = np.zeros(10)
    coordinates = np.vstack([y, x, z])

    # Poisson ratio and Youg modulus
    poisson = 0.25
    young = 3300

    model = np.array([[-100, 0, 100, 250, 300, 350]])
    pressures = [np.zeros(2), np.zeros(3), np.zeros(4)]
    for pressure in pressures:
        with pytest.raises(ValueError):
            cp.stress_x_component(
                coordinates, model, pressure, poisson, young
            )
            cp.stress_y_component(
                coordinates, model, pressure, poisson, young
            )
            cp.stress_z_component(
                coordinates, model, pressure, poisson, young
            )
            cp.displacement_x_component(
                coordinates, model, pressure, poisson, young
            )
            cp.displacement_y_component(
                coordinates, model, pressure, poisson, young
            )
            cp.displacement_z_component(
                coordinates, model, pressure, poisson, young
            )


def test_bad_kernel():
    'must stop if kernel name is invalid'
    # dummy computation points
    y = np.zeros(10)
    x = np.zeros(10)
    z = np.zeros(10)
    coordinates = np.vstack([y, x, z])

    # Poisson ratio and Youg modulus
    poisson = 0.25
    young = 3300

    model = np.array([[-100, 0, 100, 250, 300, 350]])
    pressure = np.zeros(1)
    kernels = ['chdjd', 2, 'meuovo']
    for kernel in kernels:
        with pytest.raises(ValueError):
            cp.field_component(
                coordinates, model, pressure, poisson, young, kernel
            )


def test_single_versus_multiple_prisms():
    'result for single prism must be equal to that for multiple prisms'
    # computation points
    y = np.linspace(-900, 1000, 23)
    x = np.linspace(-2500, -1400, 20)
    y, x = np.meshgrid(y, x)
    y = np.ravel(y)
    x = np.ravel(x)
    np.random.seed(89)
    z = 2*np.random.rand(x.size) - 1
    coordinates = np.vstack([y, x, z])

    # Poisson ratio and Youg modulus
    poisson = 0.25
    young = 3300

    single = np.array([[-100, 0, 100, 250, 350, 300]])
    multiple = np.array([[-100, -50, 100, 150, 350, 330],
                         [-100, -50, 150, 250, 350, 330],
                         [-50, 0, 100, 150, 350, 330],
                         [-50, 0, 150, 250, 350, 330],
                         [-100, -50, 100, 150, 330, 300],
                         [-100, -50, 150, 250, 330, 300],
                         [-50, 0, 100, 150, 330, 300],
                         [-50, 0, 150, 250, 330, 300]])
    pressure = np.zeros(8) - 100
    result_single = cp.stress_x_component(
        coordinates, single, pressure[0], poisson, young
    )
    result_multiple = cp.stress_x_component(
        coordinates, multiple, pressure, poisson, young
    )
    aae(result_single, result_multiple)
    result_single = cp.stress_y_component(
        coordinates, single, pressure[0], poisson, young
    )
    result_multiple = cp.stress_y_component(
        coordinates, multiple, pressure, poisson, young
    )
    aae(result_single, result_multiple)
    result_single = cp.stress_z_component(
        coordinates, single, pressure[0], poisson, young
    )
    result_multiple = cp.stress_z_component(
        coordinates, multiple, pressure, poisson, young
    )
    aae(result_single, result_multiple)
    result_single = cp.displacement_x_component(
        coordinates, single, pressure[0], poisson, young
    )
    result_multiple = cp.displacement_x_component(
        coordinates, multiple, pressure, poisson, young
    )
    aae(result_single, result_multiple)
    result_single = cp.displacement_y_component(
        coordinates, single, pressure[0], poisson, young
    )
    result_multiple = cp.displacement_y_component(
        coordinates, multiple, pressure, poisson, young
    )
    aae(result_single, result_multiple)
    result_single = cp.displacement_z_component(
        coordinates, single, pressure[0], poisson, young
    )
    result_multiple = cp.displacement_z_component(
        coordinates, multiple, pressure, poisson, young
    )
    aae(result_single, result_multiple)
