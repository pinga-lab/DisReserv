"""
Forward modelling of elastic reservoir deformation produced by pore-pressure
variations in prismatic cells
"""
import numpy as np
from numba import njit

def d_field_x_component(coordinates, model, DP, poisson, young):
    """
    x-component of the displacement
    """
    d_x1  = field_component(coordinates, model, DP, poisson, young, kernel='d_x1')
    d_x2  = field_component(coordinates, model, DP, poisson, young, kernel='d_x2')
    d_xz2 = field_component(coordinates, model, DP, poisson, young, kernel='d_xz2')
    
    return d_x1, d_x2, d_xz2

def d_field_y_component(coordinates, model, DP, poisson, young):
    """
    y-component of the displacement
    """
    d_y1 = field_component(coordinates, model, DP, poisson, young, kernel='d_y1')

    d_y2 = field_component(coordinates, model, DP, poisson, young, kernel='d_y2')

    d_yz2 = field_component(coordinates, model, DP, poisson, young, kernel='d_yz2')
    
    return d_y1, d_y2, d_yz2

def d_field_z_component(coordinates, model, DP, poisson, young):
    """
    z-component of the displacement
    """
    d_z1 = field_component(coordinates, model, DP, poisson, young, kernel='d_z1')

    d_z2 = field_component(coordinates, model, DP, poisson, young, kernel='d_z2')

    d_zz2 = field_component(coordinates, model, DP, poisson, young, kernel='d_zz2')
    
    return d_z1, d_z2, d_zz2

### STRESS FIELD

def s_field_x_component(coordinates, model, DP, poisson, young):
    """
    x-component of the stress
    """

    s_xz1  = field_component(coordinates, model, DP, poisson, young, kernel='s_xz1')

    s_xz2  = field_component(coordinates, model, DP, poisson, young, kernel='s_xz2')

    s_xzz2 = field_component(coordinates, model, DP, poisson, young, kernel='s_xzz2')
    
    return s_xz1, s_xz2, s_xzz2

def s_field_y_component(coordinates, model, DP, poisson, young):
    """
    y-component of the stress
    """

    s_yz1 = field_component(coordinates, model, DP, poisson, young, kernel='s_yz1')

    s_yz2 = field_component(coordinates, model, DP, poisson, young, kernel='s_yz2')

    s_yzz2 = field_component(coordinates, model, DP, poisson, young, kernel='s_yzz2')
    
    return s_yz1, s_yz2, s_yzz2

def s_field_z_component(coordinates, model, DP, poisson, young):
    """
    z-component of the stress
    """

    s_zz1 = field_component(coordinates, model, DP, poisson, young, kernel='s_zz1')

    s_zz2 = field_component(coordinates, model, DP, poisson, young, kernel='s_zz2')

    s_zzz2 = field_component(coordinates, model, DP, poisson, young, kernel='s_zzz2')

    return s_zz1, s_zz2, s_zzz2



def field_component(
    coordinates, prisms, pressure, poisson, young, kernel, dtype="float64",
    disable_checks=False
):
    """
    Displacement and stress components produced by pore-pressure variations in
    right-rectangular prisms

    The displacement and stress components are computed by integrating the solution
    proposed by Tempone et al. (2010) on the volume of rectangular cells. This
    strategy is similar to that proposed by Muñoz and Roehl (2017). The
    difference here is that we solve the volume integrals according to
    Nagy et al. (2000) and Nagy et al. (2002).
    The present code is heavily based on Harmonica (Uieda et al., 2020) and
    uses expressions which are valid at any point, either outside or inside the
    prisms.

    By following Uieda et al. (2020), the implemented expressions make use of
    the modified arctangent function proposed by Fukushima (2019) so that the
    solution to satisfies Poisson's equation in the entire domain. Moreover,
    the logarithm function was also modified in order to solve the
    singularities that the analytical solution has on some points.

    Parameters
    ----------
    coordinates : 2d-array
        2d numpy array containing ``y``, ``x`` and ``z`` Cartesian cordinates
        of the computation points. All coordinates should be in meters.
    prisms : 2d-array
        2d array containing the Cartesian coordinates of the prism(s). Each
        line contains the coordinates of a prism in following order: y1, y2,
        x1, x2, z2 and z1. All coordinates should be in meters.
    pressure : 1d array
        1d array containing the pressure of each prism in MPa.
    poisson : float
        Poisson’s ratio.
    young : float
        Young’s modulus in MPa.
    kernel : func
        Kernel function to be used for computing the desired field component.
        The available kernels for displacement components are:
        -  x-component of the 1st system: ``d_x1``
        -  y-component of the 1st system: ``d_y1``
        -  z-component of the 1st system: ``d_z1``
        -  x-component of the 2nd system: ``d_x2``
        -  y-component of the 2nd system: ``d_y2``
        -  z-component of the 2nd system: ``d_z2``
        - xz-component of the 2nd system: ``d_xz2``
        - yz-component of the 2nd system: ``d_yz2``
        - zz-component of the 2nd system: ``d_zz2``
        The available kernels for stress components are:
        -  xz-component of the 1nd system: ``s_xz1``
        -  yz-component of the 1nd system: ``s_yz1``
        -  zz-component of the 1nd system: ``s_zz1``
        - xzz-component of the 2nd system: ``s_xzz2``
        - yzz-component of the 2nd system: ``s_yzz2``
        - zzz-component of the 2nd system: ``s_zzz2``
        -  xz-component of the 2nd system: ``s_xz2``
        -  yz-component of the 2nd system: ``s_yz2``
        -  zz-component of the 2nd system: ``s_zz2``
    dtype : data-type (optional)
        Data type assigned to the resulting field component. Default to
        ``np.float64``.
    disable_checks : bool (optional)
        Flag that controls whether to perform a sanity check on the model.
        Should be set to ``True`` only when it is certain that the input model
        is valid and it does not need to be checked.
        Default to ``False``.

    Returns
    -------
    result : array
        Field component generated by the prisms at the computation points.

    References
    ----------

    Nagy, D., Papp, G. and Benedek, J. (2000). The gravitational potential and
    its derivatives for the prism. Journal of Geodesy 74: 552.
    doi:10.1007/s001900000116

    Nagy, D., Papp, G. and Benedek, J. (2002). Corrections to “The gravitational
    potential and its derivatives for the prism”. Journal of Geodesy 76: 475.
    doi:10.1007/s00190-002-0264-7

    Tempone, P., Fjær, E. and Landrø, M. (2010). Improved solution of
    displacements due to a compacting reservoir over a rigid basement.
    Applied Mathematical Modelling 34: 3352. doi:10.1016/j.apm.2010.02.025

    Muñoz, L. F. P. and Roehl, D. (2017). An analytical solution for
    displacements due to reservoir compaction under arbitrary pressure changes.
    Applied Mathematical Modelling 52: 145. doi:10.1016/j.apm.2017.06.023

    Fukushima, T. (2020) Speed and accuracy improvements in standard algorithm for prismatic gravitational field, 
    Geophys. J. Int.,  222, 1898–1908. doi: doi: 10.1093/gji/ggaa240


    Uieda, Leonardo, Soler, Santiago R., Pesce, Agustina, Oliveira Jr,
    Vanderlei C, and Shea, Nicholas. (2020, February 27). Harmonica: Forward
    modeling, inversion, and processing gravity and magnetic data
    (Version v0.1.0). Zenodo. doi:10.5281/zenodo.3628742

    """
    kernels = {
        "d_x1": kernel_d_x1,
        "d_y1": kernel_d_y1,
        "d_z1": kernel_d_z1,
        "d_x2": kernel_d_x2,
        "d_y2": kernel_d_y2,
        "d_z2": kernel_d_z2,
        "d_xz2": kernel_d_xz2,
        "d_yz2": kernel_d_yz2,
        "d_zz2": kernel_d_zz2,
        "s_xz1": kernel_s_xz1,
        "s_yz1": kernel_s_yz1,
        "s_zz1": kernel_s_zz1,
        "s_xz2": kernel_s_xz2,
        "s_yz2": kernel_s_yz2,
        "s_zz2": kernel_s_zz2,
        "s_xzz2": kernel_s_xzz2,
        "s_yzz2": kernel_s_yzz2,
        "s_zzz2": kernel_s_zzz2
    }
    if kernel not in kernels:
        raise ValueError("Kernel {} not recognized".format(kernel))
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=dtype)
    # Convert coordinates, prisms and pressure to arrays with proper shape
    coordinates = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
    prisms = np.atleast_2d(prisms)
    pressure = np.atleast_1d(pressure).ravel()
    # Sanity checks
    if not disable_checks:
        if pressure.size != prisms.shape[0]:
            raise ValueError(
                "Number of elements in pressure ({}) ".format(pressure.size)
                + "mismatch the number of prisms ({})".format(prisms.shape[0])
            )
        _check_prisms(prisms)
    # Compute the component
    jit_field_component(
        coordinates, prisms, pressure, kernels[kernel], result
    )
    result *= -Cm(poisson, young)/(4*np.pi)
    return result.reshape(cast.shape)



@njit
def jit_field_component(
    coordinates, prisms, pressure, kernel, out
):
    """
    Compute the displacement or stress component at the computations points

    Parameters
    ----------
    coordinates : 1d array
        1d array containing ``y``, ``x`` and ``z`` Cartesian coordinates of the
        computation points (in meters).
    prisms : 2d-array
        2d array containing the Cartesian coordinates of the prism(s). Each
        line contains the coordinates of a prism in following order: y1, y2,
        x1, x2, z2 and z1. All coordinates should be in meters.
    pressure : 1d array
        1d array containing the pressure of each prism in MPa.
    kernel : func
        Kernel function to be used for computing the desired field component.
    out : 1d-array
        Array where the resulting field component values will be stored.
        Must have the same size as the arrays contained on ``coordinates``.
    """
    # Iterate over computation points and prisms
    for l in range(coordinates[0].size):
        for m in range(prisms.shape[0]):
            # Iterate over the prism boundaries to compute the result of the
            # integration (see Nagy et al., 2000)
            c_z = 0.5 * (prisms[m, 4] + prisms[m, 5])
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        y_prism = prisms[m, 1 - i]
                        x_prism = prisms[m, 3 - j]
                        z_prism = prisms[m, 5 - k]
                        # If i, j or k is 1, the shift_* will refer to the
                        # lower boundary, meaning the corresponding term should
                        # have a minus sign
                        out[l] += (
                            pressure[m]
                            * (-1) ** (i + j + k)
                            * kernel(
                                y_prism,
                                x_prism,
                                z_prism,
                                c_z,
                                coordinates[0][l],
                                coordinates[1][l],
                                coordinates[2][l]
                            )
                        )

@njit
def kernel_d_x1(y, x, z, zc, yp, xp, zp):
    """
    Kernel for x-component of displacement in the infinite space domain
    (1st system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        Y * safe_log(Z + rho)
        + Z * safe_log(Y + rho)
        - X * safe_atan2(Y * Z, X * rho)
    )
    return kernel


@njit
def kernel_d_y1(y, x, z, zc, yp, xp, zp):
    """
    Kernel for y-component of displacement in the infinite space domain
    (1st system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        X * safe_log(Z + rho)
        + Z * safe_log(X + rho)
        - Y * safe_atan2(X * Z, Y * rho)
    )
    return kernel


@njit
def kernel_d_z1(y, x, z, zc, yp, xp, zp):
    """
    Kernel for z-component of displacement in the infinite space domain
    (1st system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        X * safe_log(Y + rho)
        + Y * safe_log(X + rho)
        - Z * safe_atan2(X * Y, Z * rho)
    )
    return kernel


@njit
def kernel_d_x2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for x-component of displacement in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        Y * safe_log(Z + rho)
        + Z * safe_log(Y + rho)
        - X * safe_atan2(Y * Z, X * rho)
    )
    return kernel


@njit
def kernel_d_y2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for y-component of displacement in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        X * safe_log(Z + rho)
        + Z * safe_log(X + rho)
        - Y * safe_atan2(X * Z, Y * rho)
    )
    return kernel


@njit
def kernel_d_z2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for z-component of displacement in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        X * safe_log(Y + rho)
        + Y * safe_log(X + rho)
        - Z * safe_atan2(X * Y, Z * rho)
    )
    return kernel


@njit
def kernel_d_xz2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for xz-component of displacement in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = 2 * zp * (
        safe_log(Y + rho)
    )
    return kernel


@njit
def kernel_d_yz2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for yz-component of displacement in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = 2 * zp * (
        safe_log(X + rho)
    )
    return kernel


@njit
def kernel_d_zz2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for zz-component of displacement in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = 2 * zp * (
        - safe_atan2(X * Y, Z * rho)
    )
    return kernel


@njit
def kernel_s_xz1(y, x, z, zc, yp, xp, zp):
    """
    Kernel for xz-component of stress in the infinite space domain (1st system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        safe_log(Y + rho)
    )
    return kernel


@njit
def kernel_s_yz1(y, x, z, zc, yp, xp, zp):
    """
    Kernel for yz-component of stress in the infinite space domain (1st system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        safe_log(X + rho)
    )
    return kernel


@njit
def kernel_s_zz1(y, x, z, zc, yp, xp, zp):
    """
    Kernel for zz-component of stress in the infinite space domain (1st system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        - safe_atan2(X * Y, Z * rho)
    )
    return kernel


@njit
def kernel_s_xz2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for xz-component of stress in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        safe_log(Y + rho)
    )
    return kernel


@njit
def kernel_s_yz2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for yz-component of stress in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        safe_log(X + rho)
    )
    return kernel


@njit
def kernel_s_zz2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for zz-component of stress in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
    kernel = (
        - safe_atan2(X * Y, Z * rho)
    )
    return kernel


@njit
def kernel_s_xzz2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for xzz-component of stress in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    aux = X ** 2 + Z ** 2
    rho = np.sqrt(Y ** 2 + aux)
    kernel = 2 * zp * (
        - (Y * Z)/(rho * aux)
    )
    return kernel


@njit
def kernel_s_yzz2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for yzz-component of stress in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    aux = Y ** 2 + Z ** 2
    rho = np.sqrt(X ** 2 + aux)
    kernel = 2 * zp * (
        - (X * Z)/(rho * aux)
    )
    return kernel


@njit
def kernel_s_zzz2(y, x, z, zc, yp, xp, zp):
    """
    Kernel for zzz-component of stress in the semi-infinite space domain
    (2nd system)
    """
    Y = y - yp
    X = x - xp
    Z = z - zp - 2 * zc
    X2 = X ** 2
    Y2 = Y ** 2
    Z2 = Z ** 2
    rho = np.sqrt(X2 + Y2 + Z2)
    kernel = 2 * zp * (
        ((X * Y)/rho)*((1./(X2 + Z2)) + (1./(Y2 + Z2)))
    )
    return kernel


def _check_prisms(prisms):
    """
    Check if prisms boundaries are well defined

    Parameters
    ----------
    prisms : 2d-array
        Array containing the boundaries of the prisms in the following order:
        ``y1``, ``y2``, ``x1``, ``x2``, ``z2``, ``z1``.
        The array must have the following shape: (``n_prisms``, 6), where
        ``n_prisms`` is the total number of prisms.
        This array of prisms must have valid boundaries.
        Run ``_check_prisms`` before.
    """
    y1, y2, x1, x2, z2, z1 = tuple(prisms[:, i] for i in range(6))
    err_msg = "Invalid prism or prisms. "
    bad_y = y1 > y2
    bad_x = x1 > x2
    bad_z = z1 > z2
    if bad_y.any():
        err_msg += "The y1 boundary can't be greater than the y2 one.\n"
        for prism in prisms[bad_y]:
            err_msg += "\tInvalid prism: {}\n".format(prism)
        raise ValueError(err_msg)
    if bad_x.any():
        err_msg += "The x1 boundary can't be greater than the x2 one.\n"
        for prism in prisms[bad_x]:
            err_msg += "\tInvalid prism: {}\n".format(prism)
        raise ValueError(err_msg)
    if bad_z.any():
        err_msg += "The z2 radius boundary can't be greater than the z1 one.\n"
        for prism in prisms[bad_z]:
            err_msg += "\tInvalid prism: {}\n".format(prism)
        raise ValueError(err_msg)


@njit
def safe_atan2(numerator, denominator):
    """
    Principal value of the arctangent expressed as a two variable function

    This modification has to be made to the arctangent function so the
    harmonic field of the prism satisfies the Poisson's equation.
    Therefore, it guarantees that the fields satisfies the symmetry properties
    of the prism. This modified function has been defined according to
    Fukushima (2019).
    """
    if denominator != 0:
        result = np.arctan(numerator / denominator)
    else:
        if numerator > 0:
            result = np.pi / 2
        elif numerator < 0:
            result = -np.pi / 2
        else:
            result = 0
    return result


@njit
def safe_log(argument):
    """
    Modified log to return 0 for log(0).
    The limits in the formula terms tend to 0 (see Nagy et al., 2000).
    """
    if np.abs(argument) < 1e-10:
        result = 0
    else:
        result = np.log(argument)
    return result


@njit
def Cm(poisson, young):
    """
    Uniaxial compaction coefficient Cm (Tempone et al, 2010).
    """
    result = ((1+poisson)*(1-2*poisson))/(young*(1-poisson))
    return result


def prism_layer_rectangular(region, shape, bottom, top):
    '''
    Create a rectangular planar layer of prisms.
    '''
    y1, y2, x1, x2 = region
    assert y2 > y1, 'y2 must be greater than y1'
    assert x2 > x1, 'x2 must be greater than x1'
    assert bottom > top, 'bottom must be greater than top (z points downward)'
    dy = (y2 - y1)/shape[0]
    dx = (x2 - x1)/shape[1]
    layer = []
    y = y1
    for i in range(shape[0]):
        x = x1
        for j in range(shape[1]):
            layer.append([y, y+dy, x, x+dx, bottom, top])
            x += dx
        y += dy
    layer = np.array(layer)
    return layer


def prism_layer_circular(center, radius, shape, bottom, top):
    '''
    Create a circular planar layer of prisms.
    '''
    y0, x0 = center
    assert radius > 0, 'radius must be positive'
    assert bottom > top, 'bottom must be greater than top (z points downward)'
    y_min = y0 - radius
    y_max = y0 + radius
    x_min = x0 - radius
    x_max = x0 + radius
    dy = (y_max - y_min)/shape[0]
    dx = (x_max - x_min)/shape[1]
    half_dy = 0.5*dy
    half_dx = 0.5*dx
    layer = []
    y = y_min
    for i in range(shape[0]):
        yc = y + half_dy - y0
        x = x_min
        for j in range(shape[1]):
            xc = x + half_dx - x0
            r_prism = np.sqrt(xc**2 + yc**2)
            if r_prism <= radius:
                layer.append([y, y+dy, x, x+dx, bottom, top])
            x += dx
        y += dy
    layer = np.array(layer)
    return layer


# @njit
# def kernel_u_x1_MR(y, x, z, zc , yp, xp, zp):
#     """
#     Kernel for x-component of the infinite space condition (first system)
#     computed with expressions given by Muñoz and Roehl (2017)
#     """
#     Y = yp - y
#     X = xp - x
#     Z = zp - z
#     rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
#     epsilon = np.sign(zp - zc)
#     kernel = epsilon * (
#         Y * safe_log(Z + rho)
#         + Z * safe_log(Y + rho)
#         - 0.5 * X * 1j * safe_log(X ** 2 + Y ** 2 + X * Z * 1j + Y * rho)
#         + 0.5 * X * 1j * safe_log(X ** 2 + Y ** 2 - X * Z * 1j + Y * rho)
#     ).real
#     return kernel
#
#
# @njit
# def kernel_u_x2_MR(y, x, z, zc, yp, xp, zp):
#     """
#     Kernel for x-component of the semi-space condition (second system)
#     computed with expressions given by Muñoz and Roehl (2017)
#     """
#     Y = yp - y
#     X = xp - x
#     Z = zp - z + 2 * zc
#     rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
#     kernel = (
#         Y * safe_log(Z + rho)
#         + Z * safe_log(Y + rho)
#         - 0.5 * X * 1j * safe_log(X ** 2 + Y ** 2 + X * Z * 1j + Y * rho)
#         + 0.5 * X * 1j * safe_log(X ** 2 + Y ** 2 - X * Z * 1j + Y * rho)
#     ).real
#     return kernel
#
#
# @njit
# def kernel_u_xz2_MR(y, x, z, zc, yp, xp, zp):
#     """
#     Kernel for xz-component of the semi-space condition (second system)
#     computed with expressions given by Muñoz and Roehl (2017)
#     """
#     Y = yp - y
#     X = xp - x
#     Z = zp - z + 2 * zc
#     rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
#     kernel = - (1/3) * zp * (
#         safe_log(X ** 2 + Y ** 2 - X * Z * 1j + Y * rho)
#         + safe_log(X ** 2 + Y ** 2 + X * Z * 1j + Y * rho)
#     ).real
#     return kernel
#
#
# @njit
# def kernel_u_y1_MR(y, x, z, zc, yp, xp, zp):
#     """
#     Kernel for y-component of the infinite space condition (first system)
#     computed with expressions given by Muñoz and Roehl (2017)
#     """
#     Y = yp - y
#     X = xp - x
#     Z = zp - z
#     rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
#     epsilon = np.sign(zp - zc)
#     kernel = epsilon * (
#         X * safe_log(Z + rho)
#         + Z * safe_log(X + rho)
#         - 0.5 * Y * 1j * safe_log(X ** 2 + Y ** 2 + Y * Z * 1j + X * rho)
#         + 0.5 * Y * 1j * safe_log(X ** 2 + Y ** 2 - Y * Z * 1j + X * rho)
#     ).real
#     return kernel
#
#
# @njit
# def kernel_u_y2_MR(y, x, z, zc, yp, xp, zp):
#     """
#     Kernel for y-component of the semi-space condition (second system)
#     computed with expressions given by Muñoz and Roehl (2017)
#     """
#     Y = yp - y
#     X = xp - x
#     Z = zp - z + 2 * zc
#     rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
#     kernel = (
#         X * safe_log(Z + rho)
#         + Z * safe_log(X + rho)
#         - 0.5 * Y * 1j * safe_log(X ** 2 + Y ** 2 + Y * Z * 1j + X * rho)
#         + 0.5 * Y * 1j * safe_log(X ** 2 + Y ** 2 - Y * Z * 1j + X * rho)
#     ).real
#     return kernel
#
#
# @njit
# def kernel_u_yz2_MR(y, x, z, zc, yp, xp, zp):
#     """
#     Kernel for yz-component of the semi-space condition (second system)
#     computed with expressions given by Muñoz and Roehl (2017)
#     """
#     Y = yp - y
#     X = xp - x
#     Z = zp - z + 2 * zc
#     rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
#     kernel = - (1/3) * zp * (
#         safe_log(X ** 2 + Y ** 2 - Y * Z * 1j + X * rho)
#         + safe_log(X ** 2 + Y ** 2 + Y * Z * 1j + X * rho)
#     ).real
#     return kernel
#
#
# @njit
# def kernel_u_z1_MR(y, x, z, zc, yp, xp, zp):
#     """
#     Kernel for z-component of the infinite space condition (first system)
#     computed with expressions given by Muñoz and Roehl (2017)
#     """
#     Y = yp - y
#     X = xp - x
#     Z = zp - z
#     rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
#     epsilon = np.sign(zp - zc)
#     kernel = epsilon * (
#         X * safe_log(Y + rho)
#         + Y * safe_log(X + rho)
#         + 0.5 * Z * 1j * safe_log(Y ** 2 + Z ** 2 + X * Z * 1j + Y * rho)
#         - 0.5 * Z * 1j * safe_log(- Y ** 2 - Z ** 2 + X * Z * 1j - Y * rho)
#     ).real
#     return kernel
#
#
# @njit
# def kernel_u_z2_MR(y, x, z, zc, yp, xp, zp):
#     """
#     Kernel for z-component of the semi-space condition (second system)
#     computed with expressions given by Muñoz and Roehl (2017)
#     """
#     Y = yp - y
#     X = xp - x
#     Z = zp - z + 2 * zc
#     rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
#     kernel = (
#         X * safe_log(Y + rho)
#         + Y * safe_log(X + rho)
#         + 0.5 * Z * 1j * safe_log(Y ** 2 + Z ** 2 + X * Z * 1j + Y * rho)
#         - 0.5 * Z * 1j * safe_log(- Y ** 2 - Z ** 2 + X * Z * 1j - Y * rho)
#     ).real
#     return kernel
#
#
# @njit
# def kernel_u_zz2_MR(y, x, z, zc, yp, xp, zp):
#     """
#     Kernel for zz-component of the semi-space condition (second system)
#     computed with expressions given by Muñoz and Roehl (2017)
#     """
#     Y = yp - y
#     X = xp - x
#     Z = zp - z + 2 * zc
#     rho = np.sqrt(Y ** 2 + X ** 2 + Z ** 2)
#     kernel = (1/3) * zp * 1j * (
#         safe_log(- X ** 2 - Z ** 2 + Y * Z * 1j + X * rho)
#         - safe_log(X ** 2 + Z ** 2 + Y * Z * 1j + X * rho)
#     )
#     return kernel.real