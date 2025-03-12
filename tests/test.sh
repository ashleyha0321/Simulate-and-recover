#!/bin/bash
cd "$(dirname "$0")/.."
python3 -m unittest discover -s tests -p "Test_EzDiffusionModel.py"
