import numpy as np
from .rotation3sphere import align_to_north_pole
import itertools as it

# Building the distance matrix via the family of geodesics connecting the antipodal 
# points (1,0) and (-1, 0)

def get_z1(rho, u):
    return lambda s: np.exp(-1j*u*rho*s)*(np.cos(rho*s)+1j*u*np.sin(rho*s))

def get_z2(rho, u, alpha=0):
    r = 1 - u**2
    return lambda s: r*np.exp(-1j*(u*rho*s+alpha))*np.sin(rho*s)

def geodesic(m, p, number):
    """Sample geodesic between antipodal points (1,0), (-1,0)."""
    if not ( m == round(m) and m > 0 ):
        raise ValueError(f"m={m} is an invalid value. Should be a nonzero natural number.")
    rho = np.pi*m
    if m % 2:
        # m odd
        if not ( -(m-1)/2 <= p and p <= (m-1)/2 ):
            raise ValueError(f"p={p} is out of bounds for odd m={m}: {-(m-1)/2} <= p <= {(m-1)/2}")
        u = 2*p/m
    else:
        # m even
        if not ( -m/2 <= p and p <= m/2-1 ):
            raise ValueError(f"p={p} is out of bound for even m={m}: {-(m-1)/2} <= p <= {(m-1)/2}")
        u = (2*p+1)/m
    z1, z2 = get_z1(rho, u), get_z2(rho, u)
    return [ np.array([z1(s).real, z1(s).imag, z2(s).real, z2(s).imag]) for s in np.linspace(0, 1, number) ]

def geodesic_get_allowed_p(m):
    """Return a list of all allowed values of p for a given m."""
    if not ( m == round(m) and m > 0 ):
        raise ValueError(f"m={m} is an invalid value. Should be a nonzero natural number.")
    if m % 2:
        # m odd
        return list(range(-int((m-1)/2), int((m+1)/2)))
    else:
        # m even
        return list(range(-int(m/2), int(m/2)))
    
def geodesic_sample_points(m, n):
    """Sample points for multiple geodesics in family specified by m

    Samples n points from each geodesic parameterised by m, with all p allowed for a given m.

    Parameters:
    m   Parameter m for geodesics
    n   Number of points to sample for each geodesic

    Returns:
    List of (idx, point) tuples. idx is the index of the geodesic point was sampled from.
    """
    return list(it.chain(*[zip(it.repeat(idx), geodesic(m, p, n)) for idx, p in enumerate(geodesic_get_allowed_p(m))]))

def distance(p, q):
    p, q = align_to_north_pole(p, q)
    return np.abs(np.arccos(q[0]+1j*q[1]))

def build_squared_distance_matrix(points, distance):
    """Build the matrix of squared distances from the arrray of points."""
    mat = np.zeros((len(points),)*2)
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            mat[i,j] = distance(points[i], points[j])
    return (mat + mat.T)**2

def b_matrix(A):
    n_points = A.shape[0]
    H = np.identity(n_points) - np.ones((n_points, n_points)) / n_points
    return H@A@H

#run mds algorithm
def mds(points, dimensions_target, distance=distance):
    """Calculate distances and run MDS scheme on points. Our resulting distance
    matrix B is symmetric.
    Dimension reduction to dimensions target
    New coordinates are returned such that unpacking into separate coordinates is possible, i.e.
      x, y, z, … = mds(…)
    """
    distances = build_squared_distance_matrix(points, distance)
    A = -0.5 * distances
    B = b_matrix(A)
    
    eigvals, eigvecs = np.linalg.eigh(B)  
    #Sorted in descending order
    eigvals = eigvals[::-1]  
    eigvecs = eigvecs[:, ::-1]

    small_thresh = 1e-8
    print(f"There are {eigvals.size} eigenvalues.")
    eigvals_semipos = np.max(np.vstack((eigvals, [0]*eigvals.size)).real, axis=0)
    n_eigvals_semipos = sum(eigvals_semipos.flatten() > small_thresh)
    print(f"Number of large, non-zero eigenvalues for approximate embedding: {n_eigvals_semipos} ({n_eigvals_semipos/eigvals.size*100:.2f}%)")
    
    vlambda = eigvecs @ np.sqrt(np.diag(np.maximum(eigvals, 0)))
    #approximate embedding into target dimension (sketch mds), shape (n_points,target_dimension)
    selected_coords = vlambda[:, :dimensions_target]
    
    return tuple(selected_coords[:, i] for i in range(dimensions_target))
