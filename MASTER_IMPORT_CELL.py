"""
Master Import Cell for WORKSHOP_Quantum_Starlink_Optimization.ipynb

Copy this entire cell and paste it as the FIRST executable cell in your notebook.
Run this cell BEFORE running any other cells!
"""

# ============================================================================
# MASTER IMPORTS - RUN THIS CELL FIRST!
# ============================================================================

print("🔄 Loading all required libraries...")
print("=" * 70)

# Standard library imports
import os
import sys
import time
import math
from datetime import datetime

print("✅ Standard library modules loaded")

# Data science libraries
try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} loaded")
except ImportError:
    print("❌ NumPy not found. Install: pip install numpy")
    raise

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__} loaded")
except ImportError:
    print("❌ Pandas not found. Install: pip install pandas")
    raise

# Visualization libraries
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly
    print(f"✅ Plotly {plotly.__version__} loaded")
except ImportError:
    print("❌ Plotly not found. Install: pip install plotly")
    raise

try:
    import matplotlib.pyplot as plt
    import matplotlib
    print(f"✅ Matplotlib {matplotlib.__version__} loaded")
except ImportError:
    print("⚠️  Matplotlib not found (optional)")

# Quantum computing libraries (optional)
qiskit_available = False
try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_optimization.applications import Maxcut
    from qiskit_optimization.converters import QuadraticProgramToQubo
    qiskit_available = True
    print("✅ Qiskit optimization libraries loaded")
except ImportError:
    print("⚠️  Qiskit not available (needed for quantum optimization cells)")
    print("   Install: pip install qiskit qiskit-optimization qiskit-algorithms")

# Custom project modules
telemetry_available = False
try:
    from starlink_telemetry import StarlinkTelemetryGenerator, print_telemetry_summary
    telemetry_available = True
    print("✅ Starlink telemetry module loaded")
except ImportError:
    print("⚠️  starlink_telemetry.py not found in current directory")

# API libraries (optional)
try:
    import requests
    print("✅ Requests library loaded")
except ImportError:
    print("⚠️  Requests not found (needed for N2YO API)")

print("=" * 70)
print("✅ LIBRARY LOADING COMPLETE!")
print("=" * 70)

# Display system information
print("\n📊 System Information:")
print(f"  Python version: {sys.version.split()[0]}")
print(f"  Platform: {sys.platform}")
print(f"  Working directory: {os.getcwd()}")

# Check for required data files
print("\n📁 Data Files:")
csv_file = "starlink_telemetry_dataset.csv"
if os.path.exists(csv_file):
    size_mb = os.path.getsize(csv_file) / (1024 * 1024)
    print(f"  ✅ {csv_file} found ({size_mb:.2f} MB)")
else:
    print(f"  ⚠️  {csv_file} not found")

print("\n🎯 Ready to run notebook cells!")
print("=" * 70)

# Create convenience flags for conditional execution
QISKIT_AVAILABLE = qiskit_available
TELEMETRY_AVAILABLE = telemetry_available

# Set random seeds for reproducibility
np.random.seed(42)
print("\n🎲 Random seed set to 42 for reproducibility")
