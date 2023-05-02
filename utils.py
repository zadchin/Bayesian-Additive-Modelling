import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy import sparse
from scipy.optimize import minimize
import blinpy as bp


def custom_interp_matrix_2d(data, grid_points):
    x1 = data[:, 0]
    x2 = data[:, 1]

    # Create a meshgrid based on the grid_points
    grid_x1, grid_x2 = np.meshgrid(grid_points[0], grid_points[1], indexing='ij')

    # Create an array representing the 'z' values at the grid points
    z = np.arange(len(grid_points[0]) * len(grid_points[1])).reshape(len(grid_points[0]), len(grid_points[1]))

    # Create a RegularGridInterpolator object to perform the interpolation
    interpolator = RegularGridInterpolator((grid_points[0], grid_points[1]), z, method='linear', bounds_error=False, fill_value=None)

    # Calculate the interpolated values
    interp_values = interpolator(np.vstack((x1, x2)).T)

    # Create the interpolation matrix
    A = np.zeros((len(x1), len(grid_points[0]) * len(grid_points[1])))

    for i, value in enumerate(interp_values):
        # Find the closest index in the flattened z array
        closest_index = np.argmin(np.abs(z.ravel() - value))
        A[i, closest_index] = 1

    return A

def custom_interp_matrix_1d(data, grid_points):
    x = data

    # Create an array representing the 'z' values at the grid points
    z = np.arange(len(grid_points))

    # Create a RegularGridInterpolator object to perform the interpolation
    interpolator = RegularGridInterpolator((grid_points,), z, method='linear', bounds_error=False, fill_value=None)

    # Calculate the interpolated values
    interp_values = interpolator(x)

    # Create the interpolation matrix
    A = np.zeros((len(x), len(grid_points)))

    for i, value in enumerate(interp_values):
        # Find the closest index in the z array
        closest_index = np.argmin(np.abs(z - value))
        A[i, closest_index] = 1

    return A

def smooth_diff1(input_col, xfit, mu=0, std=1e9, diff_mu=0, diff_std=1,
                 diff_order=2, sparse=True, name=None):
    nfit = len(xfit)
    D = bp.utils.diffmat(nfit, order=diff_order, sparse=sparse)
    diff_mu = diff_mu*np.ones(nfit-diff_order)
    diff_cov = diff_std**2*np.ones(nfit-diff_order)

    fmu = mu*np.ones(nfit)
    fcov = std**2*np.ones(nfit)

    return {
        'fun': lambda df: custom_interp_matrix_1d(df[input_col].values, xfit),
        'name': 'f({0})'.format(input_col) if name is None else name,
        'prior': {
            'B': np.vstack((D, np.eye(nfit))) if not sparse else
                bp.utils.vstack((D, bp.utils.eye(nfit))),
            'mu': np.concatenate((diff_mu, fmu)),
            'cov': np.concatenate((diff_cov, fcov))
        }
    }