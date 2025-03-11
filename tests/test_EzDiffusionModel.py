import unittest
import numpy as np
from src.EzDiffusionModel import forward_equations, inverse_equations, sample_observed_statistics, simulate_and_recover

class TestEzDiffusionModel(unittest.TestCase):
    def test_forward_equations(self):
        """Test if forward equations produce reasonable summary statistics."""
        v, a, t = 1.0, 1.0, 0.2
        R_pred, M_pred, V_pred = forward_equations(v, a, t)
        self.assertTrue(0 < R_pred < 1, "Accuracy rate should be between 0 and 1")
        self.assertTrue(M_pred > 0, "Mean RT should be positive")
        self.assertTrue(V_pred > 0, "Variance RT should be positive")
    
    def test_inverse_equations(self):
        """Test if inverse equations correctly recover parameters when given noise-free input."""
        v_true, a_true, t_true = 1.0, 1.0, 0.2
        R_pred, M_pred, V_pred = forward_equations(v_true, a_true, t_true)
        v_est, a_est, t_est = inverse_equations(R_pred, M_pred, V_pred)
        self.assertAlmostEqual(v_true, v_est, places=4)
        self.assertAlmostEqual(a_true, a_est, places=4)
        self.assertAlmostEqual(t_true, t_est, places=4)
    
    def test_sample_observed_statistics(self):
        """Test if sampled observed statistics have reasonable values."""
        v, a, t = 1.0, 1.0, 0.2
        R_pred, M_pred, V_pred = forward_equations(v, a, t)
        R_obs, M_obs, V_obs = sample_observed_statistics(R_pred, M_pred, V_pred, N=100)
        self.assertTrue(0 < R_obs < 1, "Observed accuracy rate should be between 0 and 1")
        self.assertTrue(M_obs > 0, "Observed mean RT should be positive")
        self.assertTrue(V_obs > 0, "Observed variance RT should be positive")
    
    def test_simulate_and_recover(self):
        """Test the full simulate-and-recover process with known parameters."""
        v_true, a_true, t_true = 1.0, 1.0, 0.2
        bias, squared_error = simulate_and_recover(N=100, v_true=v_true, a_true=a_true, t_true=t_true)
        self.assertEqual(len(bias), 3, "Bias should have three elements")
        self.assertEqual(len(squared_error), 3, "Squared error should have three elements")
    
if __name__ == "__main__":
    unittest.main()
