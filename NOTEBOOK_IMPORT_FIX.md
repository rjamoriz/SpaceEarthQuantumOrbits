# 🔧 Notebook Import Issues - Complete Fix

## Problem

Your notebook cells are missing required imports, causing `NameError` exceptions:
- **Cell 14**: `NameError: name 'np' is not defined`
- **Cell 15**: `NameError: name 'go' is not defined`

## Root Cause

Each cell in Jupyter notebooks needs to have access to the libraries it uses. If imports are done in one cell, they need to be **run before** cells that use those libraries.

## ✅ Complete Solution

### Option 1: Create a Master Import Cell (RECOMMENDED)

Create a **new cell at the very beginning** (after the markdown cells) with all imports:

```python
# Cell: Master Imports - Run this first!

# Standard library
import os
import sys
import time
import math
from datetime import datetime

# Data science
import numpy as np
import pandas as pd

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Quantum computing (if available)
try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import NumPyMinimumEigensolver
    print("✅ Qiskit libraries loaded")
except ImportError:
    print("⚠️  Qiskit not available (optional for some cells)")

# Custom modules
try:
    from starlink_telemetry import StarlinkTelemetryGenerator
    print("✅ Starlink telemetry module loaded")
except ImportError:
    print("⚠️  starlink_telemetry.py not found")

print("\n" + "="*70)
print("✅ ALL LIBRARIES IMPORTED SUCCESSFULLY!")
print("="*70)
print(f"📦 NumPy version: {np.__version__}")
print(f"📦 Pandas version: {pd.__version__}")
print(f"📦 Plotly version: {go.__version__ if hasattr(go, '__version__') else 'installed'}")
print("="*70)
```

**Then run this cell FIRST before running any other cells!**

### Option 2: Add Imports to Each Cell

If you prefer each cell to be self-contained, add imports at the top of each cell:

#### **Cell 14 Fix:**

```python
# Cell 14: Scale-up Point

# Add these imports at the top
import numpy as np
import time
from starlink_telemetry import StarlinkTelemetryGenerator

# Configuration for large-scale simulation
LARGE_CONSTELLATION_SIZE = 100  # Increase to 200, 500, or more for stress testing

print("🚀 Large-Scale Constellation Simulation")
print("=" * 70)
print(f"Constellation Size: {LARGE_CONSTELLATION_SIZE} satellites")

# ... rest of your code
```

#### **Cell 15 Fix:**

```python
# Cell 15: Starlink Telemetry Dataset Analysis

# Add these imports at the top
import plotly.graph_objects as go
import pandas as pd
import os

# Dataset URL
DATASET_URL = "https://www.opendatabay.com/data/science-research/ce2cc978-11e7-462c-ab7e-3c06e6841ea3"
LOCAL_CSV_PATH = "starlink_telemetry_dataset.csv"

# ... rest of your code
```

## 📋 Required Libraries by Cell

Here's what each cell type typically needs:

### Data Analysis Cells:
```python
import numpy as np
import pandas as pd
```

### Visualization Cells:
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
```

### Quantum Optimization Cells:
```python
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
```

### Telemetry Cells:
```python
from starlink_telemetry import StarlinkTelemetryGenerator
import numpy as np
```

### Timing/Performance Cells:
```python
import time
```

## 🎯 Best Practice Workflow

1. **Create master import cell** (Option 1 above)
2. **Run it first** every time you open the notebook
3. **Restart kernel** → **Run All Cells** to ensure everything works in order

### In Jupyter:
- Click: **Kernel** → **Restart & Run All**
- This ensures all imports are loaded before cells that need them

## 🔍 Debugging Import Issues

If you still get `NameError`, check:

### 1. **Verify library is installed:**
```python
# Run in a cell
import sys
!{sys.executable} -m pip list | grep -E "numpy|pandas|plotly"
```

### 2. **Check import worked:**
```python
# Run in a cell
try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} imported")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")
```

### 3. **Restart kernel:**
Sometimes imports get cached incorrectly:
- **Kernel** → **Restart Kernel**
- Re-run import cell
- Re-run problem cell

## 📦 Install Missing Libraries

If any library is missing, install it:

```bash
# In terminal
pip install numpy pandas plotly matplotlib

# For quantum computing
pip install qiskit qiskit-optimization qiskit-algorithms
```

Or in a notebook cell:
```python
import sys
!{sys.executable} -m pip install numpy pandas plotly matplotlib
```

## ✅ Quick Fix for Your Current Issue

**Right now, to fix cell 14:**

1. **Go to cell 14**
2. **Add at the very top:**
   ```python
   import numpy as np
   import time
   ```
3. **Re-run the cell**

**To fix cell 15:**

1. **Go to cell 15**
2. **Add at the very top:**
   ```python
   import plotly.graph_objects as go
   import pandas as pd
   import os
   ```
3. **Re-run the cell**

## 🎓 Understanding Jupyter Notebook Execution

Key points:
- **Cells execute independently** but share the same Python kernel
- **Imports persist** once run in the kernel
- **Order matters**: Import cells must run before cells using those imports
- **Kernel restart clears everything**: Must re-run imports after restart

## 📝 Recommended Notebook Structure

```
Cell 1: Title & Description (Markdown)
Cell 2: Master Imports ← RUN THIS FIRST!
Cell 3: Configuration & Constants
Cell 4: Data Loading
Cell 5-N: Analysis & Visualization
```

This ensures all imports are available for all subsequent cells!

---

**After adding the imports, your cells will work perfectly!** 🎉
