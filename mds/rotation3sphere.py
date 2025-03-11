import numpy as np

def cartesian_to_hyperspherical(p):
    """
    Convert Cartesian coordinates to hyperspherical coordinates.
    """
    w, x, y, z = p
    r = np.sqrt(w**2 + x**2 + y**2 + z**2)
    phi1 = np.arccos(w / r)
    phi2 = np.arctan2(np.sqrt(x**2 + y**2), x)
    phi3 = np.arctan2(y, z)
    return phi1, phi2, phi3

def hyperspherical_to_cartesian(p):
    """
    Convert hyperspherical coordinates to Cartesian coordinates.
    """ 
    phi1, phi2, phi3 = p
    w = np.cos(phi1)
    x = np.sin(phi1) * np.cos(phi2)
    y = np.sin(phi1) * np.sin(phi2) * np.cos(phi3)
    z = np.sin(phi1) * np.sin(phi2) * np.sin(phi3)
    return np.array([w, x, y, z])

def spherical_to_cartesian(p):
    """
    Convert spherical coordinates (S^2) to cartesian coordinates (R^3).
    """
    theta, phi = p
    return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)

def align_to_north_pole(p, q):
    """
    Align p to the north pole and adjust q.
    """
    phi1_1, phi2_1, phi3_1 = cartesian_to_hyperspherical(p)
    phi1_2, phi2_2, phi3_2 = cartesian_to_hyperspherical(q)
    
    # Align p to the north pole
    # Subtract P1's angles from P2's angles
    phi1_2_adjusted = phi1_2 - phi1_1
    phi2_2_adjusted = phi2_2 - phi2_1
    phi3_2_adjusted = phi3_2 - phi3_1
    
    q_adjusted = hyperspherical_to_cartesian((phi1_2_adjusted, phi2_2_adjusted, phi3_2_adjusted))
    return np.array([1, 0, 0, 0]), q_adjusted 
