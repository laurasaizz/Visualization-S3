import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def compress_to_sphere(target_dim, *coords):
    """
    Compress points into a unit sphere S1 or S2, given as separated coordinates,
    with a specified target dimension.
    
    Parameters:
        target_dim (int): Target dimension (2 for S1, 3 for S2).
        *coords: Separate arrays of coordinates (x, y, [z] for 3D points).
                 Each array should have the same length.
    
    Returns:
        tuple: Compressed coordinates as separate arrays (x', y', [z']).
    """
    # Combine the input coordinates into an (n_points, n_dims) array
    points = np.vstack(coords).T
    
    input_dim = len(coords)
    if input_dim != target_dim:
        raise ValueError(f"Input dimension ({input_dim}) does not match target dimension ({target_dim}).")
    
    # Find the farthest distance between points and scale so the farthest distance becomes 2
    max_distance = np.max(np.linalg.norm(points, axis=1))
    scaled_points = points / max_distance * 2
    
    # Normalize all points to lie on the sphere
    compressed_points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return tuple(compressed_points[:, i] for i in range(target_dim))

def add_axis_cube(axis, side_len, shift=[0,0,0], colour='#00000020'):
    """
    Draw cube with side length(s) side_len centered on [0,0,0] into axis.

    Parameters
        axis: axis to draw cube into
        side_len: Scalar or np.ndarray of shape (3,)
            Specifies the side length in all or each of the axes
        shift: Shift the center of the cube by this vector
        colour: Colour of the cube drawn
    """
    vertices = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1], [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]])/2*side_len + shift
    # indices of vertices to connect for edges
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    for edge in edges:
        axis.plot(vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], colour)