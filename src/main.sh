#!/bin/bash

set -e

# Navigate to the script directory
cd "$(dirname "$0")"

# Run the EZ Diffusion Model simulation
echo "Running EZ Diffusion Model simulation..."
/usr/bin/python3 ../src/EzDiffusionModel.py


