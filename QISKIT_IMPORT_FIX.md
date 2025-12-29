# 🔧 Qiskit Import Fix - Sampler Update

## The Problem

In Qiskit 2.x (you have version 2.2.3), the `Sampler` class has been moved and renamed. The old import path no longer works.

## ✅ Quick Fix

Replace the old import:
```python
# OLD (doesn't work in Qiskit 2.x)
from qiskit.primitives import Sampler
```

With the new import:
```python
# NEW (works in Qiskit 2.x)
from qiskit.primitives import StatevectorSampler as Sampler
```

Or use the reference implementation:
```python
from qiskit.primitives import Sampler  # This should work
# If not, use:
from qiskit_algorithms.utils import algorithm_globals
```

## 📋 Complete Updated Imports for Quantum Cell

Replace your quantum optimization cell imports with this:

```python
# Cell: Quantum Optimization with Qiskit

import numpy as np
import time

# Qiskit optimization imports
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

# Qiskit algorithm imports (updated for Qiskit 2.x)
try:
    # Try new Qiskit 2.x import
    from qiskit.primitives import StatevectorSampler as Sampler
    print("✅ Using Qiskit 2.x StatevectorSampler")
except ImportError:
    try:
        # Fallback to older import
        from qiskit.primitives import Sampler
        print("✅ Using Qiskit Sampler")
    except ImportError:
        print("⚠️  Sampler not available, will use NumPy solver")
        Sampler = None

from qiskit_algorithms import QAOA, VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA, ADAM, SPSA

print("✅ Qiskit libraries imported successfully")
print(f"   Qiskit version: {qiskit.__version__ if 'qiskit' in dir() else 'unknown'}")
```

## 🎯 Alternative: Use NumPy Solver (Simpler)

If you just want to test the optimization without quantum hardware:

```python
# Simpler approach - use classical NumPy solver
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver

print("✅ Using classical NumPy solver (no quantum hardware needed)")

# Later in your code, use:
solver = NumPyMinimumEigensolver()
optimizer = MinimumEigenOptimizer(solver)
```

## 🔍 Qiskit Version Compatibility

### Qiskit 1.x (older):
```python
from qiskit.primitives import Sampler
from qiskit.algorithms import QAOA, VQE
```

### Qiskit 2.x (current - you have 2.2.3):
```python
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_algorithms import QAOA, VQE
```

## 📦 Check Your Qiskit Installation

Run this to verify:

```python
import qiskit
import qiskit_algorithms
import qiskit_optimization

print(f"Qiskit: {qiskit.__version__}")
print(f"Qiskit Algorithms: {qiskit_algorithms.__version__}")
print(f"Qiskit Optimization: {qiskit_optimization.__version__}")

# Check available primitives
from qiskit import primitives
print(f"Available primitives: {dir(primitives)}")
```

## 🚀 Complete Working Example

```python
# Complete quantum optimization cell for Qiskit 2.x

import numpy as np
import qiskit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA
from qiskit_algorithms.optimizers import COBYLA

print(f"✅ Qiskit {qiskit.__version__} loaded")

# Create a simple QUBO problem
qp = QuadraticProgram()
qp.binary_var('x')
qp.binary_var('y')
qp.minimize(linear={'x': -1, 'y': -2}, quadratic={('x', 'y'): 1})

print("✅ Quadratic program created")

# Solve with NumPy (classical)
numpy_solver = NumPyMinimumEigensolver()
numpy_optimizer = MinimumEigenOptimizer(numpy_solver)
result = numpy_optimizer.solve(qp)

print(f"✅ Solution found: {result.x}")
print(f"   Objective value: {result.fval}")

# Optional: Try QAOA (quantum)
try:
    qaoa = QAOA(optimizer=COBYLA(), reps=2)
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    qaoa_result = qaoa_optimizer.solve(qp)
    print(f"✅ QAOA solution: {qaoa_result.x}")
except Exception as e:
    print(f"⚠️  QAOA not available: {e}")
```

## 💡 Recommendation

For your notebook, I recommend using the **NumPy solver** for now since it:
- ✅ Works without quantum hardware
- ✅ No import issues
- ✅ Fast for small problems
- ✅ Same API as quantum solvers

You can always switch to QAOA later when you need quantum features!

---

**Update your imports to use `StatevectorSampler` or switch to `NumPyMinimumEigensolver`!** 🚀
