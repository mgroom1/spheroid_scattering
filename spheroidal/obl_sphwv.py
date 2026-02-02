import os
import zipfile
import glob
import numpy as np
from types import SimpleNamespace
import subprocess
import math
import scipy.io as sio

def obl_calculate_everything(path, max_memory, precision, c, m, n, 
                             eta_min, eta_max, n_eta, xi_min, xi_max, n_xi, p,
                             n_dr, log10_dr_min, n_dr_neg, log10_dr_neg_min, 
                             n_c2k, log10_c2k_min, n_B2r, log10_B2r_min):
    """
    Calls the external obl_sphwv solver, captures ASCII output, and zips it.
    """
    
    base_name = generate_name(c, m, n)
    
    print('calculating lambda_approx...')
    obl_calculate_lambdamn_approx(c, m, n)
    
    print('calculating lambda, dr, dr_neg, N, F, k1, k2, c2k, Q, and B2r...')
    
    # Construct the command
    # Using nice_number to ensure high-precision float-to-string conversion
    executable = os.path.join(path, "obl_sphwv")
    
    command_coeff = [
        executable,
        "-max_memory", str(max_memory),
        "-precision", str(precision),
        "-verbose", "n",
        "-c", nice_number(c, 20),
        "-m", str(m),
        "-n", str(n),
        "-w", "everything",
        "-n_dr", str(n_dr),
        "-dr_min", f"1.0e{log10_dr_min}",
        "-n_dr_neg", str(n_dr_neg),
        "-dr_neg_min", f"1.0e{log10_dr_neg_min}",
        "-n_c2k", str(n_c2k),
        "-c2k_min", f"1.0e{log10_c2k_min}",
        "-n_B2r", str(n_B2r),
        "-B2r_min", f"1.0e{log10_B2r_min}"
    ]

    print(f"Executing: {' '.join(command_coeff)}")

    # Run the system commands
    try:
        # check=True will raise an exception if the return code is non-zero
        # text=True and capture_output=False mimics the '-echo' behavior
        result = subprocess.run(command_coeff, check=True)
    except subprocess.CalledProcessError:
        print("External solver failed. Terminating calculation.")
        return
    
    print('calculating S1...')
    
    deta = (eta_max - eta_min)/n_eta
    command_S1 = [
        executable,
        "-max_memory", str(max_memory),
        "-precision", str(precision),
        "-verbose", "n",
        "-c", nice_number(c, 20),
        "-m", str(m),
        "-n", str(n),
        "-w", "S1",
        "-a", nice_number(eta_min, 20),
        "-b", nice_number(eta_max, 20),
        "-d", nice_number(deta, 20),
        "-arg_type", "eta",
        "-p", str(p)
    ]
    
    print(f"Executing: {' '.join(command_S1)}")

    # Run the system commands
    try:
        # check=True will raise an exception if the return code is non-zero
        # text=True and capture_output=False mimics the '-echo' behavior
        with open(f"data/obl_{base_name}_S1.txt", "w") as f:
            result = subprocess.run(command_S1, stdout=f, check=True)
    except subprocess.CalledProcessError:
        print("External solver failed. Terminating calculation.")
        return
    
    print('calculating R...')
    
    dxi = (xi_max - xi_min)/n_xi
    command_R = [
        executable,
        "-max_memory", str(max_memory),
        "-precision", str(precision),
        "-verbose", "n",
        "-c", nice_number(c, 20),
        "-m", str(m),
        "-n", str(n),
        "-w", "R",
        "-a", nice_number(xi_min, 20),
        "-b", nice_number(xi_max, 20),
        "-d", nice_number(dxi, 20),
        "-arg_type", "xi",
        "-which", "R1_1,R1_2,R2_1,R2_2,R2_31,R2_32",
        "-p", str(p)
    ]
    
    print(f"Executing: {' '.join(command_R)}")

    # Run the system commands
    try:
        # check=True will raise an exception if the return code is non-zero
        # text=True and capture_output=False mimics the '-echo' behavior
        with open(f"data/obl_{base_name}_R.txt", "w") as f:
            result = subprocess.run(command_R, stdout=f, check=True)
    except subprocess.CalledProcessError:
        print("External solver failed. Terminating calculation.")
        return
    
    # List of files to zip
    suffixes = [
        'lambda_approx', 'lambda', 'log_abs_lambda', 'dr', 'log_abs_dr',
        'dr_neg', 'log_abs_dr_neg', 'N', 'log_abs_N', 'F', 'log_abs_F',
        'k1', 'log_abs_k1', 'k2', 'log_abs_k2', 'c2k', 'log_abs_c2k',
        'Q', 'log_abs_Q', 'B2r', 'log_abs_B2r', 'S1', 'R'
    ]
    
    zip_filename = os.path.join('data', f'obl_{base_name}.zip')
    
    # Create the Zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for suffix in suffixes:
            fname = f'obl_{base_name}_{suffix}.txt'
            # Assuming files are generated in the 'data' folder
            full_path = os.path.join('data', fname)
            
            if os.path.exists(full_path):
                # arcname=fname stores the file in the zip without the 'data/' prefix
                zipf.write(full_path, arcname=fname)
            else:
                print(f"Warning: Expected file {fname} not found.")

    # Cleanup: Delete the .txt files after zipping
    #for suffix in suffixes:
    #    fname = f'data/obl_{base_name}_{suffix}.txt'
    #    os.remove(fname)
    
    print(f"Successfully zipped coefficients to {zip_filename}")

def obl_calculate_lambdamn_approx(c, m, n):
    """
    Calculate and save an approximate characteristic value using the 
    eigenvalues of a tridiagonal matrix.
    """
    
    N_size = m + n + 200
    
    # Determine the starting r based on parity of (n-m)
    r_start = 0 if (n - m)%2 == 0 else 1
    
    # Generate the sequence of r values: [r_start, r_start+2, ..., r_start+2*(N-1)]
    r_vals = r_start + 2*np.arange(N_size)
    
    # Vectorized calculation of diagonal and off-diagonal elements
    betas  = [calculate_betar(c, m, r) for r in r_vals]
    alphas = [calculate_alphar(c, m, r) for r in r_vals[:-1]] # upper diagonal
    gammas = [calculate_gammar(c, m, r) for r in r_vals[1:]]  # lower diagonal
    
    # Construct tridiagonal matrix A
    A = np.diag(betas) + np.diag(alphas, k=1) + np.diag(gammas, k=-1)
    
    # Calculate eigenvalues and sort them
    d = np.sort(np.linalg.eigvals(A))
    
    # Select the specific eigenvalue corresponding to mode n
    if (n - m)%2 == 0:
        lambda_approx = d[(n - m)//2]
    else:
        lambda_approx = d[(n - m - 1)//2]
        
    # Write to file
    base_name = generate_name(c, m, n)
    file_path = os.path.join('data', f'obl_{base_name}_lambda_approx.txt')
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write(f"{nice_number(lambda_approx, 20)}\n")

def calculate_alphar(c, m, r):
    return (((2*m + r + 2)*(2*m + r + 1)) / 
            ((2*m + 2*r + 5)*(2*m + 2*r + 3)))*(-(c**2))

def calculate_betar(c, m, r):
    return ((m + r)*(m + r + 1) + 
            ((2*(m + r)*(m + r + 1) - 2*(m**2) - 1) / 
             ((2*m + 2*r - 1)*(2*m + 2*r + 3)))*(-(c**2)))

def calculate_gammar(c, m, r):
    return ((r*(r - 1))/((2*m + 2*r - 3)*(2*m + 2*r - 1)))*(-(c**2))

def obl_load_everything(path, c, m, n):
    """
    Unzip spheroidal coefficients, load text data, and organize into an object.
    """
    
    # We use SimpleNamespace to allow dot notation like everything.c
    everything = SimpleNamespace()
    everything.c = c
    everything.m = m
    everything.n = n
    
    base_name = generate_name(c, m, n)

    # Load Coefficients
    try:
        everything.coefficients = SimpleNamespace()
        zip_path = os.path.join(path, f'obl_{base_name}.zip')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
            
        def load_txt(suffix, transpose=False):
            data = np.loadtxt(os.path.join(path, f'obl_{base_name}_{suffix}.txt'))
            return data.T if transpose else data

        coeffs = everything.coefficients
        coeffs.lambda_val = load_txt('lambda')
        coeffs.log_abs_lambda = load_txt('log_abs_lambda')
        coeffs.dr = load_txt('dr', True)
        coeffs.log_abs_dr = load_txt('log_abs_dr', True)
        coeffs.dr_neg = load_txt('dr_neg', True)
        coeffs.log_abs_dr_neg = load_txt('log_abs_dr_neg', True)
        coeffs.N = load_txt('N')
        coeffs.log_abs_N = load_txt('log_abs_N')
        coeffs.F = load_txt('F')
        coeffs.log_abs_F = load_txt('log_abs_F')
        coeffs.k1 = load_txt('k1')
        coeffs.log_abs_k1 = load_txt('log_abs_k1')
        coeffs.k2 = load_txt('k2')
        coeffs.log_abs_k2 = load_txt('log_abs_k2')
        coeffs.c2k = load_txt('c2k', True)
        coeffs.log_abs_c2k = load_txt('log_abs_c2k', True)
        coeffs.Q = load_txt('Q')
        coeffs.log_abs_Q = load_txt('log_abs_Q')
        coeffs.B2r = load_txt('B2r', True)
        coeffs.log_abs_B2r = load_txt('log_abs_B2r', True)
    except Exception as e:
        print(f"Can't open one or more of the coefficients... {e}")

    # Load S1 (Angular functions)
    try:
        everything.S1 = SimpleNamespace()
        zip_path_s1 = os.path.join(path, f'obl_{base_name}.zip')
        with zipfile.ZipFile(zip_path_s1, 'r') as zip_ref:
            zip_ref.extractall(path)
            
        temp = np.loadtxt(os.path.join(path, f'obl_{base_name}_S1.txt'), delimiter=',').T
        eta = temp[1, :]
        ix = np.argsort(eta)
        
        s1 = everything.S1
        s1.eta = eta[ix]
        s1.S1_1 = temp[2, ix]
        s1.S1p_1 = temp[3, ix]
        s1.S1_2 = temp[4, ix]
        s1.S1p_2 = temp[5, ix]
        s1.S1_log_abs_difference = temp[6, ix]
        s1.S1p_log_abs_difference = temp[7, ix]
        s1.S1 = s1.S1_1
        s1.S1p = s1.S1p_1
    except Exception as e:
        print(f"Can't open S1... {e}")

    # Load R (Radial functions)
    try:
        r_file = os.path.join(path, f'obl_{base_name}_R.txt')
        everything.R = obl_load_everything_R(r_file)
    except Exception as e:
        print(f"Can't open R... {e}")

    # Cleanup extracted text files
    pattern = os.path.join(path, f'obl_{base_name}_*.txt')
    for f in glob.glob(pattern):
        os.remove(f)
        
    return everything

def obl_load_everything_R(filename):
    """
    Loads radial functions and selects the best version based on Wronskian error.
    """
    R = SimpleNamespace()
    temp = np.loadtxt(filename, delimiter=',').T
    
    # Map data
    R.xi = temp[1, :]
    R.R1_1, R.R1p_1 = temp[2, :], temp[3, :]
    R.R1_2, R.R1p_2 = temp[4, :], temp[5, :]
    
    # Error columns for Wronskian
    errors = temp[30:36, :]
    
    # Vectorized minimization: find index of best method for each point i
    min_errors = np.min(errors, axis=0)
    best_idx = np.argmin(errors, axis=0) 

    # Pre-allocate result arrays
    n = len(R.xi)
    R.R1 = np.zeros(n)
    R.R1p = np.zeros(n)
    R.R2 = np.zeros(n)
    R.R2p = np.zeros(n)
    R.W_log_abs_error = min_errors

    # Vectorized Selection using boolean masks (much faster than a loop)
    # idx == 0 (Method 1), idx == 1 (Method 2)...
    for i in range(6):
        mask = (best_idx == i)
        if not np.any(mask): continue
        
        if i < 3: # Methods 1, 2, 3 use R1_1
            R.R1[mask] = R.R1_1[mask]
            R.R1p[mask] = R.R1p_1[mask]
        else:     # Methods 4, 5, 6 use R1_2
            R.R1[mask] = R.R1_2[mask]
            R.R1p[mask] = R.R1p_2[mask]
            
        # Map R2 and R2p based on the index (matching MATLAB temp logic)
        r2_indices = [8, 10, 12, 8, 10, 14] # indices in temp for R2
        R.R2[mask] = temp[r2_indices[i], mask]
        R.R2p[mask] = temp[r2_indices[i]+1, mask]

    return R

def obl_save_everything(path, everything):
    """
    Save the 'everything' object back to a .mat file, 
    matching the MATLAB structure.
    """
    
    # Helper to recursively convert SimpleNamespace/objects to dicts
    def to_dict(obj):
        if hasattr(obj, "__dict__"):
            return {k: to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [to_dict(i) for i in obj]
        return obj

    file_name = f"obl_{generate_name(everything.c, everything.m, everything.n)}.mat"
    full_path = os.path.join(path, file_name)
    
    # Convert our Python object back to a dictionary structure
    data_to_save = {'everything': to_dict(everything)}
    
    # Save as a standard -mat file
    sio.savemat(full_path, data_to_save)
    print(f"Saved: {full_path}")

def generate_name(c, m, n):
    """
    Generate a filename string based on k*a, m, and n.
    """
    # round(1000.0 * c) is formatted as an 8-digit integer with leading zeros
    # m and n are formatted as 3-digit integers with leading zeros
    return f"{int(round(1000.0 * c)):08d}_{m:03d}_{n:03d}"

def nice_number(number, p):
    """
    Format a number into a clean scientific notation string with 'p' 
    significant figures, stripping unnecessary padding in the exponent.
    """
    if math.isnan(number):
        return 'nan'
    
    if math.isinf(number):
        return '-inf' if number < 0 else 'inf'
    
    # Format to scientific notation with p-1 decimals (total p sig figs)
    # The 'e' format specifier produces something like '1.234567e+02'
    formatted = f"{number:.{p-1}e}"
    
    # Split the coefficient and the exponent
    coefficient, exponent = formatted.split('e')
    
    # Convert exponent to int and back to string to remove leading zeros/plus signs
    # e.g., "+02" -> 2 -> "2" or "-05" -> -5 -> "-5"
    clean_exponent = str(int(exponent))
    
    return f"{coefficient}e{clean_exponent}"

if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('saved', exist_ok=True)

    # Path to the compiled solver executables
    solver_path = os.path.join(os.getcwd(), 'sphwv')

    c = 10.0
    for m in range(1):
        for n in range(m, m+50):
            print(f"Processing c = {c}, m = {m}, n = {n}")
            
            obl_calculate_everything(
                solver_path, 2000, 500, c, m, n, 
                -1.0, 1.0, 32768, 0.0, 8.0, 2048, 20,
                10, -300, 10, -300, 10, -300, 10, -300
            )
            obl_ev = obl_load_everything('data', c, m, n)
            obl_save_everything('saved', obl_ev)
    
