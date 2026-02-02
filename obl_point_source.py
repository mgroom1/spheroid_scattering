import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.io as sio

from obl_geo import cart_to_obl, grad_obl_to_cart, create_slice

def generate_obl_point_source_scat_figure(k, a, x0, z0, xi1, alpha, w, nels, path='saved'):
    """
    Generates and plots the total field (incident + scattered).
    """
    # Create the evaluation slice
    x_2d, y_2d, z_2d = create_slice([-w, 0.0, -w], [0.0, 0.0, 2.0*w], [2.0*w, 0.0, 0.0], nels, nels)
    
    x = x_2d.ravel()
    y = y_2d.ravel()
    z = z_2d.ravel()
    
    # Calculate incident field
    v_in, _ = point_source_in(k, x0, 0.0, z0, x, y, z)
    
    # Get source coordinates in oblate spheroidal system
    source_cart = np.array([[x0], [0.0], [z0]])
    source_obl = cart_to_obl(a, source_cart)
    eta0, xi0 = source_obl[0, 0], source_obl[1, 0]
    
    # Calculate scattered field based on boundary condition
    if isinstance(alpha, str):
        if alpha == 'soft':
            v_scat, _, _ = obl_point_source_scat_soft(k, a, eta0, xi0, path, xi1, x, y, z)
        else: # hard
            v_scat, _, _ = obl_point_source_scat_hard(k, a, eta0, xi0, path, xi1, x, y, z)
    else:
        v_scat, _, _ = obl_point_source_scat_robin(k, a, eta0, xi0, path, xi1, alpha, x, y, z)
    
    # Total field and normalization
    v_total = v_in + v_scat
    # Normalize by source amplitude at origin
    s_norm, _ = point_source_in(k, x0, 0.0, z0, 0.0, 0.0, 0.0)
    v_norm = v_total / np.abs(s_norm)
    
    # Reshape back to grid for plotting
    v_plot = np.real(v_norm).reshape(nels, nels)
    
    # Calculate spheroid boundary for plotting
    semi_major = a*np.sqrt(xi1**2 + 1.0)
    semi_minor = a*xi1

    # Plotting
    plt.figure(figsize=(8, 6))
    
    im = plt.imshow(v_plot, extent=[-w, w, -w, w], origin='lower', 
                    cmap='magma', vmin=-2.0, vmax=2.0)
    
    # Add the spheroid boundary as an ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(semi_major*np.cos(theta), semi_minor*np.sin(theta), 'w--', linewidth=1.5)
    
    plt.colorbar(im, label='Normalized Magnitude')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.title(f'Point Source Scattering (xi1={xi1})')
    
    plt.savefig("obl_point_source.png")
    plt.show()

def obl_point_source_scat_hard(k, a, eta0, xi0, path, xi1, x, y, z):
    """
    Wrapper for calculating scattered field from a sound-hard oblate spheroid.
    """

    # Define the term calculator as a lambda that fixes the source 
    # and boundary parameters (eta0, xi0, xi1, and 'hard')
    term_func = lambda k, a, c, m, n, everything, eta, xi, phi: \
        calculate_point_source_scat_term(k, a, c, m, n, everything, eta, xi, phi, eta0, xi0, xi1, 'hard')

    # Execute the sum
    v_scat, grad_scat_cart, max_abs_change_scat = obl_calculate_sum(
        k, a, path, x, y, z, term_func
    )

    return v_scat, grad_scat_cart, max_abs_change_scat

def obl_point_source_scat_soft(k, a, eta0, xi0, path, xi1, x, y, z):
    """
    Wrapper for calculating scattered field from a sound-soft oblate spheroid.
    """

    # Define the term calculator as a lambda that fixes the source 
    # and boundary parameters (eta0, xi0, xi1, and 'soft')
    term_func = lambda k, a, c, m, n, everything, eta, xi, phi: \
        calculate_point_source_scat_term(k, a, c, m, n, everything, eta, xi, phi, eta0, xi0, xi1, 'soft')

    # Execute the sum
    v_scat, grad_scat_cart, max_abs_change_scat = obl_calculate_sum(
        k, a, path, x, y, z, term_func
    )

    return v_scat, grad_scat_cart, max_abs_change_scat

def obl_point_source_scat_robin(k, a, eta0, xi0, path, xi1, alpha, x, y, z):
    """
    Wrapper for calculating scattered field from an oblate spheroid with Robin boundary conditions.
    """

    # Define the term calculator as a lambda that fixes the source 
    # and boundary parameters (eta0, xi0, xi1, and alpha)
    term_func = lambda k, a, c, m, n, everything, eta, xi, phi: \
        calculate_point_source_scat_term(k, a, c, m, n, everything, eta, xi, phi, eta0, xi0, xi1, alpha)

    # Execute the sum
    v_scat, grad_scat_cart, max_abs_change_scat = obl_calculate_sum(
        k, a, path, x, y, z, term_func
    )

    return v_scat, grad_scat_cart, max_abs_change_scat

def calculate_point_source_scat_term(k, a, c, m, n, everything, eta, xi, phi, eta0, xi0, xi1, alpha):
    """
    Calculate a term in the spheroidal wave function expansion for the scattered field.
    """
    epsilon = 2.0 if m > 0 else 1.0
    
    # Internal helper for spline interpolation
    def spline_interp(x_points, y_points, query_points):
        f = interp1d(x_points, y_points, kind='cubic', fill_value="extrapolate")
        return f(query_points)
    
    s1 = everything.S1
    r = everything.R

    # Interpolate at source position
    S1_0 = spline_interp(s1.eta[s1.S1_idxs], s1.S1[s1.S1_idxs], eta0)
    R3_0 = spline_interp(r.xi[r.idxs], r.R3[r.idxs], xi0)
    
    # Interpolate at spheroid boundary xi1
    R1_1 = spline_interp(r.xi[r.idxs], r.R1[r.idxs], xi1)
    R1p_1 = spline_interp(r.xi[r.idxs], r.R1p[r.idxs], xi1)
    R3_1 = spline_interp(r.xi[r.idxs], r.R3[r.idxs], xi1)
    R3p_1 = spline_interp(r.xi[r.idxs], r.R3p[r.idxs], xi1)
    
    A = ((1j*k)/(2.0*np.pi))*epsilon*S1_0*R3_0
    
    # Boundary condition logic
    if isinstance(alpha, str):
        if alpha == 'soft':
            B = -A*(R1_1/R3_1)
        else: # 'hard'
            B = -A*(R1p_1/R3p_1)
    else:
        # Impedance/Robin boundary condition
        B = -A*((R1_1 + alpha*R1p_1)/(R3_1 + alpha*R3p_1))
        
    # Interpolate at evaluation points
    S1 = spline_interp(s1.eta[s1.S1_idxs], s1.S1[s1.S1_idxs], eta)
    S1p = spline_interp(s1.eta[s1.S1p_idxs], s1.S1p[s1.S1p_idxs], eta)
    R3 = spline_interp(r.xi[r.idxs], r.R3[r.idxs], xi)
    R3p = spline_interp(r.xi[r.idxs], r.R3p[r.idxs], xi)
    
    Phi = np.cos(m*phi)
    Phip = -m*np.sin(m*phi)
    
    # Field term
    dv = B*S1*R3*Phi
    
    # Gradient in spheroidal coordinates
    dgrad = B*np.vstack([
        S1p*R3*Phi,   # d/d_eta
        S1*R3p*Phi,   # d/d_xi
        S1*R3*Phip    # d/d_phi
    ])
    
    # Max absolute change, only considering points outside or on the spheroid
    mask = xi >= xi1
    if np.any(mask):
        # Filter dv and each row of dgrad by the mask
        dv_masked = dv[mask]
        dgrad_masked = dgrad[:, mask]
        combined = np.concatenate([dv_masked.flatten(), dgrad_masked.flatten()])
        max_abs_change = np.max(np.abs(combined))
    else:
        max_abs_change = 0.0
        
    return dv, dgrad, max_abs_change

def obl_calculate_sum(k, a, path, x, y, z, obl_calculate_term):
    """
    Calculate an oblate spheroidal wave function expansion.
    """
    c = k * a
    cart = np.vstack([x, y, z])
    
    obl = cart_to_obl(a, cart)
    eta = obl[0, :]
    xi = obl[1, :]
    phi = obl[2, :]
    
    # Initialize containers
    # Using length of x (assuming x is a 1D array)
    num_points = x.size if isinstance(x, np.ndarray) else len(x)
    v = np.zeros(num_points, dtype=complex)
    grad_obl = np.zeros((3, num_points), dtype=complex)
    
    # Using a dictionary or a list of lists for max_abs_change 
    max_abs_change_dict = {}

    for m in range(500):
        break_again = 0
        for n in range(m, m + 500):
            try:
                # Load precomputed data
                everything = obl_open_everything(path, c, m, n)
                
                # Reconstruct complex radial functions
                everything.R.R3 = everything.R.R1 + 1j*everything.R.R2
                everything.R.R3p = everything.R.R1p + 1j*everything.R.R2p
                
                everything.S1.S1_idxs = np.isfinite(everything.S1.S1)
                everything.S1.S1p_idxs = np.isfinite(everything.S1.S1p)
                
                everything.R.idxs = (np.isfinite(everything.R.R1) & 
                                     np.isfinite(everything.R.R1p) & 
                                     np.isfinite(everything.R.R2) & 
                                     np.isfinite(everything.R.R2p))
                
                # Calculate the term
                dv, dgrad_obl, change = obl_calculate_term(k, a, c, m, n, everything, eta, xi, phi)
                
                v += dv
                grad_obl += dgrad_obl
                max_abs_change_dict[(m, n)] = change
                
                print(f"m = {m:3d}, n = {n:3d}, max_abs_change = {change:.5e}")
                
                # Check convergence against machine epsilon
                if change < np.finfo(float).eps:
                    break_again += 1
                else:
                    break_again = 0
                    
            except Exception:
                # implementation follows original MATLAB but this is really 
                # sloppy.
                break_again += 1
            
            if break_again == 3:
                break
                
    grad_cart = grad_obl_to_cart(a, obl, grad_obl)
    
    return v, grad_cart, max_abs_change_dict

def obl_open_everything(path, c, m, n):
    """
    Load precomputed oblate spheroidal wave functions from a .mat file.
    """
    
    file_name = f"{path}/obl_{generate_name(c, m, n)}.mat"
    
    # Load the mat file
    # struct_as_record=False makes the data accessible more like MATLAB objects
    # squeeze_me=True removes singleton dimensions
    data = sio.loadmat(file_name, struct_as_record=False, squeeze_me=True)
    everything = data['everything']
    
    return everything

def generate_name(c, m, n):
    """
    Generate a filename string based on k*a, m, and n.
    """
    # round(1000.0 * c) is formatted as an 8-digit integer with leading zeros
    # m and n are formatted as 3-digit integers with leading zeros
    return f"{int(round(1000.0 * c)):08d}_{m:03d}_{n:03d}"

def point_source_in(k, px, py, pz, x, y, z):
    """
    Calculate the incident field due to a point source.
    """
    # Calculate distance
    r = np.sqrt((x - px)**2 + (y - py)**2 + (z - pz)**2)
    
    # Incident field
    v_in = np.exp(1j*k*r)/(4.0*np.pi*r)
    
    # Common factor for the gradient calculation
    grad_factor = v_in*(1j*k - 1.0/r)
    
    # Cartesian gradient components
    grad_in_cart = np.array([
        grad_factor*((x - px)/r),
        grad_factor*((y - py)/r),
        grad_factor*((z - pz)/r)
    ])
    
    return v_in, grad_in_cart

if __name__ == "__main__":
    generate_obl_point_source_scat_figure(10.0, 1.0, 0.0, 3.0, 0.0, 'hard', 5.0, 200, path='spheroidal/saved')
    #generate_obl_point_source_scat_figure(10.0, 1.0, 0.0, 3.0, 0.0, 'hard', 5.0, 200, path='saved')
