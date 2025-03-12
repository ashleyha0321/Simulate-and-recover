#!/bin/bash

# Run the full simulation
python3 -c '
from src.EzDiffusionModel import run_simulation

results = run_simulation(iterations=1000, sample_sizes=[10, 40, 4000])

with open("version.md", "w") as file:
    for N, metrics in results.items():
        file.write(f"### Sample Size {N}:\n")
        file.write(f"- **Mean Bias:** {metrics["mean_bias"]}\n")
        file.write(f"- **Mean Squared Error:** {metrics["mean_squared_error"]}\n\n")'


