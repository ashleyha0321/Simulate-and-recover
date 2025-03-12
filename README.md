# Cogs106 Final Project: Simulate & Recover 


## Introduction
The purpose of the Simulate-and-Recover exercise is to evaluate the accuaracy at which the EZ Diffusion Model can estimate parameters from simulated response time (RT) data. This process allows us to test a model's ability to recover the parameters 
from the data it generates itself. By using different sample sizes while repeatedly simulating the data, we are can assess the model's overall performance and sensitivity to noise.

## Experimental Setup
The EZ Diffusion Model is used to simplify the standard Drift Diffusion Model (DDF). It consists of three central parameters known as drift rate (v), boundary separation (a), and nondecision time (t). These are functions of the summary statistics that are obtained from the various RT distributions. The simulation occurs as follows:

1. True parameters are randomly generated, falling within the required ranges of:

   Boundary separation (a): [0.5,2]
   Drift Rate (v): [0.5, 2]
   Nondecision time (t): [0.1, 0.5]

2. Use forward equations to compute the predicted summary statistics, which include accuracy, mean RT, and variance.
3. Simulte the observed data through sampling from distributions.
4. Apply inverse equations in order to estimate the parameters from observed distributions.
5. Repeat the process for 1000 iterations at each sample size of N = 10, 40, and 4000 — this results in a total of 3000 iterations.
6. Use the results to analyze the bias and squared error to get a measure of the model's accuracy/performance.

## Results & Discussion
The results from running the iterations is stored in a output/version.md. The results obtained are as follows:

### Sample Size 10:

Mean Bias: [0.956, -12.91, -0.319]

Mean Squared Error: [1.040, 473.41, 0.147]

### Sample Size 40:

Mean Bias: [0.939, -6.65, -0.305]

Mean Squared Error: [1.046, 82.67, 0.141]

### Sample Size 4000:

Mean Bias: [0.959, -5.21, -0.322]

Mean Squared Error: [1.063, 41.90, 0.134]

For each sample size, the drift rate (v) seems to be fairly consistent, being overestimated at about 0.95 units above the true v used in the simulation. With the boundary separation (a), there is a notable bias with a being -12.91 at N = 10 compared to a being -5.21 at N = 4000. This indicates that the estimation of a improves as the sample becomes larger. The nondecision time bias is much more miniscule ranging from -0.32 to 0.30, which means there is a slight underestimation for t. 


