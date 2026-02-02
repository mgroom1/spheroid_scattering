import numpy as np

def cart_to_obl(a, cart):
    """
    Convert Cartesian coordinates (x, y, z) to oblate spheroidal 
    coordinates (eta, xi, phi).
    """
    x = cart[0, :]
    y = cart[1, :]
    z = cart[2, :]
    
    # Pre-calculate constants and intermediate terms
    a2 = a**2
    b = -a2
    c_term = x**2 + y**2 + z**2 - a2
    d_term = z**2
    
    # Calculate xi (radial-like coordinate)
    xi = np.sqrt((-c_term - np.sqrt(c_term**2 - 4.0 * b * d_term)) / (2.0 * b))
    
    # Calculate eta (angular-like coordinate)
    # np.maximum(0, ...) ensures no negative values due to precision errors before sqrt
    eta_sq = 1.0 - (x**2 + y**2) / (a2 * (xi**2 + 1.0))
    eta = np.sqrt(np.maximum(0, eta_sq))
    
    # Apply sign of z to eta to cover the full range [-1, 1]
    eta[z < 0.0] = -eta[z < 0.0]
    
    # Calculate phi (azimuthal angle)
    phi = np.arctan2(y, x)
    
    # Stack into a (3, N) array
    return np.vstack([eta, xi, phi])

def obl_to_cart(a, obl):
    """
    Convert oblate spheroidal coordinates (eta, xi, phi) to 
    Cartesian coordinates (x, y, z).
    """
    eta = obl[0, :]
    xi = obl[1, :]
    phi = obl[2, :]
    
    # Pre-calculate shared terms
    sq_eta = np.sqrt(1.0 - eta**2)
    sq_xi = np.sqrt(xi**2 + 1.0)
    
    x = a * sq_eta * sq_xi * np.cos(phi)
    y = a * sq_eta * sq_xi * np.sin(phi)
    z = a * eta * xi
    
    return np.vstack([x, y, z])

def obl_calculate_D(a, obl):
    """
    Calculate the transformation matrix D for a single point 
    in oblate spheroidal coordinates.
    """
    eta = obl[0]
    xi = obl[1]
    phi = obl[2]
    
    # Pre-calculate common terms for efficiency
    sq_eta = np.sqrt(1.0 - eta**2)
    sq_xi = np.sqrt(xi**2 + 1.0)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    D = np.zeros((3, 3))
    
    # Row 1
    D[0, 0] = -a * (1.0 / sq_eta) * eta * sq_xi * cos_p
    D[0, 1] = -a * (1.0 / sq_eta) * eta * sq_xi * sin_p
    D[0, 2] = a * xi
    
    # Row 2
    D[1, 0] = a * sq_eta * (1.0 / sq_xi) * xi * cos_p
    D[1, 1] = a * sq_eta * (1.0 / sq_xi) * xi * sin_p
    D[1, 2] = a * eta
    
    # Row 3
    D[2, 0] = -a * sq_eta * sq_xi * sin_p
    D[2, 1] = a * sq_eta * sq_xi * cos_p
    D[2, 2] = 0.0
    
    return D

def grad_cart_to_obl(a, cart, grad_cart):
    """
    Convert a gradient from Cartesian to Oblate Spheroidal coordinates.
    """
    
    obl = cart_to_obl(a, cart)
    num_points = cart.shape[1]
    grad_obl = np.zeros((3, num_points), dtype=complex)
    
    for i in range(num_points):
        # Calculate the transformation matrix D for the i-th point
        D = obl_calculate_D(a, obl[:, i])
        
        # Check the eta coordinate for singularities/boundaries
        if abs(obl[0, i]) < 1.0:
            # Matrix-vector multiplication: D @ grad_cart[:, i]
            grad_obl[:, i] = D @ grad_cart[:, i]
        else:
            # Handle the case where eta is exactly 1 or -1
            grad_obl[:, i] = [
                np.nan, 
                D[1, 2] * grad_cart[2, i], 
                np.nan
            ]
            
    return grad_obl

def grad_obl_to_cart(a, obl, grad_obl):
    """
    Convert a gradient from Oblate Spheroidal to Cartesian coordinates.
    """
    
    num_points = obl.shape[1]
    grad_cart = np.zeros((3, num_points), dtype=complex)
    
    for i in range(num_points):
        # Current point and transformation matrix
        current_obl = obl[:, i]
        D = obl_calculate_D(a, current_obl)
        
        # Check eta for singularities
        if abs(current_obl[0]) < 1.0:
            grad_cart[:, i] = np.linalg.solve(D, grad_obl[:, i])
        else:
            # Boundary case (poles)
            grad_cart[:, i] = [
                np.nan, 
                np.nan, 
                grad_obl[1, i] / D[1, 2]
            ]
            
    return grad_cart

def create_slice(corner, a, b, a_nels, b_nels):
    """
    Create a 2D slice (grid) in 3D space defined by a corner point 
    and two spanning vectors.
    """
    # Ensure inputs are numpy arrays for vector math
    corner = np.array(corner)
    a_vec = np.array(a)
    b_vec = np.array(b)
    
    # Generate linear weights from 0 to 1
    s = np.linspace(0, 1, a_nels)
    t = np.linspace(0, 1, b_nels)
    
    # Use broadcasting to calculate all points p = corner + s*a + t*b
    # np.outer(s, a_vec) creates a matrix of shape (a_nels, 3)
    # np.outer(t, b_vec) creates a matrix of shape (b_nels, 3)
    # We expand dimensions to sum them into an (a_nels, b_nels, 3) grid
    grid = (corner[np.newaxis, np.newaxis, :] + 
            s[:, np.newaxis, np.newaxis] * a_vec + 
            t[np.newaxis, :, np.newaxis] * b_vec)
    
    # Extract x, y, z components
    x = grid[:, :, 0]
    y = grid[:, :, 1]
    z = grid[:, :, 2]
    
    return x, y, z
