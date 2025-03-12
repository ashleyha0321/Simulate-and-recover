import numpy as np
from scipy.stats import binom, norm, gamma

def forward_equations(v, a, t):
    """Compute predicted summary statistics from EZ diffusion model parameters."""
    y = np.exp(-a * v)
    R_pred = 1 / (y + 1)
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / (y + 1)**2)
    return R_pred, M_pred, V_pred

def inverse_equations(R_obs, M_obs, V_obs):
    """Recover EZ diffusion model parameters from observed summary statistics."""
    R_obs = np.clip(R_obs, 1e-6, 1 - 1e-6)  # Prevent division errors
    L = np.log(R_obs / (1 - R_obs))

    # Prevent invalid sqrt inputs
    sqrt_term = L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs
    sqrt_term = max(sqrt_term, 1e-6)  # Ensure non-negative

    v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(sqrt_term)
    if np.isnan(v_est) or v_est == 0:
        v_est = 1e-6  # Small fallback value

    a_est = L / v_est
    t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))

    return v_est, a_est, t_est

def sample_observed_statistics(R_pred, M_pred, V_pred, N):
    """Simulate observed summary statistics from the predicted values."""
    T_obs = binom.rvs(N, R_pred) / N
    M_obs = norm.rvs(M_pred, np.sqrt(V_pred / N))
    V_obs = gamma.rvs((N - 1) / 2, scale=(2 * V_pred / (N - 1)))
    return T_obs, M_obs, V_obs

def simulate_and_recover(N, v_true, a_true, t_true):
    """Perform one simulate-and-recover iteration."""
    R_pred, M_pred, V_pred = forward_equations(v_true, a_true, t_true)
    R_obs, M_obs, V_obs = sample_observed_statistics(R_pred, M_pred, V_pred, N)
    v_est, a_est, t_est = inverse_equations(R_obs, M_obs, V_obs)
    bias = np.array([v_true - v_est, a_true - a_est, t_true - t_est])
    squared_error = bias ** 2
    return bias, squared_error

def run_simulation(iterations=1000, sample_sizes=[10, 40, 4000]):
    """Run the full simulate-and-recover experiment."""
    results = {}
    for N in sample_sizes:
        biases = []
        squared_errors = []
        for _ in range(iterations):
            v_true = np.random.uniform(0.5, 2)
            a_true = np.random.uniform(0.5, 2)
            t_true = np.random.uniform(0.1, 0.5)
            bias, se = simulate_and_recover(N, v_true, a_true, t_true)
            biases.append(bias)
            squared_errors.append(se)
        results[N] = {
            "mean_bias": np.mean(biases, axis=0),
            "mean_squared_error": np.mean(squared_errors, axis=0)
        }
    return results

#if __name__ == "__main__":
    #results = run_simulation()
    #for N, metrics in results.items():
        #print(f"Sample Size {N}:")
        #print(f"  Mean Bias: {metrics['mean_bias']}")
        #print(f"  Mean Squared Error: {metrics['mean_squared_error']}")

if __name__ == "__main__":
    results = run_simulation()

    # Open version.md in write mode
    with open("version.md", "w") as file:
        for N, metrics in results.items():
            file.write(f"### Sample Size {N}:\n")  
            file.write(f"- **Mean Bias:** {metrics['mean_bias']}\n")  
            file.write(f"- **Mean Squared Error:** {metrics['mean_squared_error']}\n\n")  

