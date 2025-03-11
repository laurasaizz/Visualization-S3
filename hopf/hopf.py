import numpy as np
from mds.rotation3sphere import cartesian_to_hyperspherical

def hopf_fibration_inverse(points, num_samples, hyperspherical=False):
    """
    Computes the inverse Hopf fibration for multiple points on S^2.
    
    Parameters:
        points (ndarray): Array of shape (N, 2) containing spherical coordinates (theta, phi) of points on S^2.
                          theta: Polar angle (0 <= theta <= pi)
                          phi: Azimuthal angle (0 <= phi < 2pi)
        num_samples (int): Number of points to sample from each fiber.
    
    Returns:
        list of tuples: Each tuple contains:
            - fiber_index (int): The index of the point on S^2 the fiber belongs to.
            - sampled_fiber (ndarray): Array of shape (num_samples, 4) containing the sampled fiber points in S^3.
    """
    fibers = []
    
    for index, (theta, phi) in enumerate(points):
        
        psi_values = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        
        half_theta = theta / 2
        z1 = np.cos(half_theta) * np.exp(1j * psi_values)  
        z2 = np.sin(half_theta) * np.exp(1j * (phi + psi_values))  
        
        x1, y1 = np.real(z1), np.imag(z1)
        x2, y2 = np.real(z2), np.imag(z2)
        
        # Stack into a (num_samples, 4) array
        sampled_fiber = np.column_stack([x1, y1, x2, y2])

        fibers.append((index, sampled_fiber))

    if hyperspherical:
        return [ (index, np.array([cartesian_to_hyperspherical(p) for p in sampled_fiber])) for index, sampled_fiber in fibers]
    else:
        return fibers

def localcoordsampling_2sphere(n):
    """
    Generate approximately equidistant points on the 2-sphere and return them in
    spherical coordinates (local parametrization).

    Parameters:
    n (int): Number of points to sample.

    Returns:
    np.ndarray: Array of shape (n, 2), where each row is (theta, phi).
    """
    phi = (1 + np.sqrt(5)) / 2

    points = np.zeros((n, 2))

    for k in range(n):
        theta = np.arccos(1 - 2 * (k + 0.5) / n)  # Latitude (theta), range [0, π]
        phi_k = 2 * np.pi * (k / phi**2)          # Longitude (phi), range [0, 2π]

        points[k, 0] = theta
        points[k, 1] = phi_k

    return points