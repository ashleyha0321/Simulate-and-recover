import numpy as np
import scipy.stats as stats
import os

# classes were constructed with the assistance of chatGPT

# parameter ranges definitions
boundary_range = (0.5, 2)
drift_range = (0.5, 2)
nondecision_range = (0.1, 0.5)

# sample sizes
n_values = [10, 40, 4000]
iterations = 1000

def generate_parameters():
    """Generate random parameters within the specified ranges."""
    a = np.random.uniform(*boundary_range)
    v = np.random.uniform(*drift_range)
    t = np.random.uniform(*nondecision_range)
    return a, v, t

def forward_equations(a, v, t):
    """Compute the forward equations to obtain predicted summary statistics."""
    y = np.exp(-a * v)
    R_pred = 1 / (y + 1)
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / ((y + 1)**2))
    return R_pred, M_pred, V_pred

def simulate_observed_data(R_pred, M_pred, V_pred, N):
    """Simulate observed data using sampling distributions."""
    T_obs = np.random.binomial(N, R_pred)
    R_obs = T_obs / N
    R_obs = np.clip(R_obs, 1e-5, 1 - 1e-5)  # Avoids exact 0 or 1
    M_obs = np.random.normal(M_pred, np.sqrt(max(V_pred / N, 1e-5)))
    V_obs = np.random.gamma(max((N - 1) / 2, 1e-5), max((2 * V_pred) / (N - 1), 1e-5))
    return R_obs, M_obs, V_obs

def inverse_equations(R_obs, M_obs, V_obs):
    """Compute the inverse equations to estimate model parameters."""
    L = np.log(R_obs / (1 - R_obs))
    L = np.clip(L, -10, 10)  # Prevent extreme values
    
    if V_obs <= 0:
        return np.nan, np.nan, np.nan
    
    v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(max(L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs, 1e-5))
    if np.isnan(v_est) or v_est == 0:
        return np.nan, np.nan, np.nan
    
    a_est = L / v_est
    t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))
    
    return a_est, v_est, t_est

def run_simulation():
    """Run the simulate-and-recover process for multiple iterations and sample sizes."""
    results = {}
    for N in n_values:
        biases = []
        squared_errors = []
        for _ in range(iterations):
            true_a, true_v, true_t = generate_parameters()
            R_pred, M_pred, V_pred = forward_equations(true_a, true_v, true_t)
            R_obs, M_obs, V_obs = simulate_observed_data(R_pred, M_pred, V_pred, N)
            est_a, est_v, est_t = inverse_equations(R_obs, M_obs, V_obs)
            
            if np.isnan(est_a) or np.isnan(est_v) or np.isnan(est_t):
                continue  
            
            bias = np.array([true_a - est_a, true_v - est_v, true_t - est_t])
            squared_error = bias**2
            
            biases.append(bias)
            squared_errors.append(squared_error)
        
        if len(biases) == 0:
            results[N] = {
                "mean_bias": np.array([np.nan, np.nan, np.nan]),
                "mean_squared_error": np.array([np.nan, np.nan, np.nan])
            }
        else:
            biases = np.array(biases)
            squared_errors = np.array(squared_errors)
            results[N] = {
                "mean_bias": biases.mean(axis=0),
                "mean_squared_error": squared_errors.mean(axis=0)
            }
    
    # printing results to version.md
    with open("version.md", "w") as file:
        for N, metrics in results.items():
            file.write(f"### Sample Size {N}:\n")  
            file.write(f"- **Mean Bias:** {metrics['mean_bias']}\n")  
            file.write(f"- **Mean Squared Error:** {metrics['mean_squared_error']}\n\n")  
    
    return results

if __name__ == "__main__":
    results = run_simulation()
    for N, res in results.items():
        print(f"N={N} - Bias Mean: {res['mean_bias']}, Squared Error Mean: {res['mean_squared_error']}")
