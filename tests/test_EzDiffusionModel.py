import unittest
import numpy as np
from src.EzDiffusionModel import generate_parameters, forward_equations, simulate_observed_data, inverse_equations, run_simulation

# unittest suite was constructed with the assistance of chatGPT

class TestEzDiffusionModel(unittest.TestCase):
    
    def test_generate_parameters(self):
        """Test that parameters fall in required range."""
        a, v, t = generate_parameters()
        self.assertTrue(0.5 <= a <= 2)
        self.assertTrue(0.5 <= v <= 2)
        self.assertTrue(0.1 <= t <= 0.5)
    
    def test_forward_equations(self):
        """Test that forward equations return reasonable values."""
        a, v, t = 1.0, 1.0, 0.2  
        R_pred, M_pred, V_pred = forward_equations(a, v, t)
        self.assertTrue(0 <= R_pred <= 1)
        self.assertTrue(M_pred > 0)
        self.assertTrue(V_pred > 0)
    
    def test_inverse_equations(self):
        """Test that inverse equations recover logical estimates."""
        R_obs, M_obs, V_obs = 0.7, 0.5, 0.1  
        a_est, v_est, t_est = inverse_equations(R_obs, M_obs, V_obs)
        self.assertFalse(np.isnan(a_est))
        self.assertFalse(np.isnan(v_est))
        self.assertFalse(np.isnan(t_est))
    
    def test_simulate_observed_data(self):
        """Test that simulated data stays within expected ranges."""
        R_pred, M_pred, V_pred = 0.7, 0.5, 0.1
        N = 100
        R_obs, M_obs, V_obs = simulate_observed_data(R_pred, M_pred, V_pred, N)
        self.assertTrue(0 <= R_obs <= 1)
        self.assertTrue(M_obs > 0)
        self.assertTrue(V_obs > 0)
    
    def test_run_simulation(self):
        """Test that simulation runs and produces logical output."""
        results = run_simulation()
        for N in [10, 40, 4000]:
            self.assertIn(N, results)
            self.assertIn("mean_bias", results[N])
            self.assertIn("mean_squared_error", results[N])
            self.assertEqual(len(results[N]["mean_bias"]), 3)
            self.assertEqual(len(results[N]["mean_squared_error"]), 3)

if __name__ == "__main__":
    unittest.main()
