import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_interpolation_2d(x, y, point_idxs, cmap='viridis'):
    """
    Plots a scatter plot with points color-coded by geodesic they belong to and 
    adds interpolating lines for points within the same geodesic.

    Parameters:
    - x (array-like): x-coordinates of points.
    - y (array-like): y-coordinates of points.
    - point_idxs (array-like): Group index for each point (used for color coding).
    - cmap (str): Colormap to use for the groups (default: 'Spectral').
    """
    colormap = plt.get_cmap(cmap)
    
    unique_idxs = np.unique(point_idxs)
    norm = plt.Normalize(vmin=min(unique_idxs), vmax=max(unique_idxs))
    
    sorted_groups = sorted(unique_idxs, key=lambda group: np.mean(x[point_idxs == group]))

    for group in unique_idxs:
        mask = point_idxs == group
        x_group = np.array(x)[mask]
        y_group = np.array(y)[mask]
        
        sorted_indices = np.argsort(x_group)
        x_group = x_group[sorted_indices]
        y_group = y_group[sorted_indices]
        
        if len(x_group) > 1: 
            x_interp = np.linspace(x_group[0], x_group[-1], 200)
            y_interp = np.interp(x_interp.real, x_group.real, y_group.real)
            
            # Plot the interpolated line with the same color as the group
            plt.plot(x_interp, y_interp, color=colormap(norm(group)), label=f"Geodesic {group}", linewidth=1 + len(unique_idxs) - sorted_groups.index(group))
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Geodesics via MDS and grouped interpolations")
    plt.legend(fontsize=8)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.png"
    plt.savefig(filename)
    plt.show()

def plot_interpolation_3d(x, y, z, point_idxs, cmap='viridis'):
    """
    Plots a 3D plot with interpolating lines for points within the same geodesic, 
    color-coded by geodesic group. Scatter points are not displayed.

    Parameters:
    - x (array-like): x-coordinates of points.
    - y (array-like): y-coordinates of points.
    - z (array-like): z-coordinates of points.
    - point_idxs (array-like): Group index for each point (used for color coding).
    - cmap (str): Colormap to use for the groups (default: 'viridis').
    """
    colormap = plt.get_cmap(cmap)
    
    unique_idxs = np.unique(point_idxs)
    norm = plt.Normalize(vmin=min(unique_idxs), vmax=max(unique_idxs))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    sorted_groups = sorted(unique_idxs, key=lambda group: np.mean(z[point_idxs == group]))

    # Interpolate points and plot lines for each group
    for group in unique_idxs:
        mask = point_idxs == group
        x_group = np.array(x)[mask]
        y_group = np.array(y)[mask]
        z_group = np.array(z)[mask]
        
        sorted_indices = np.argsort(x_group)
        x_group = x_group[sorted_indices]
        y_group = y_group[sorted_indices]
        z_group = z_group[sorted_indices]
        
        if len(x_group) > 1:
            x_interp = np.linspace(x_group[0], x_group[-1], 200)
            y_interp = np.interp(x_interp.real, x_group.real, y_group.real)
            z_interp = np.interp(x_interp.real, x_group.real, z_group.real)
            
            # Plot the interpolated line with the same color as the group
            ax.plot(x_interp, y_interp, z_interp, color=colormap(norm(group)), label=f"Geodesic {group}", linewidth=1 + len(unique_idxs) - sorted_groups.index(group))
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Geodesics via MDS and grouped interpolations")
    ax.legend(fontsize=8)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.png"
    plt.savefig(filename)
    plt.show()

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
    compressed_points = scaled_points / np.linalg.norm(scaled_points, axis=1, keepdims=True)
    
    return tuple(compressed_points[:, i] for i in range(target_dim))