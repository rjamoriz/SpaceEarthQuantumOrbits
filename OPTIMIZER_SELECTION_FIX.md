# 🔧 Optimizer Selection - Variable Not Defined Fix

## The Problem

The code is trying to use `satellite_ids` before it's been defined. You need to either:
1. Define `satellite_ids` first, OR
2. Move the optimizer selection to after you create your satellite data

## ✅ Solution 1: Set Default in Setup Cell

In your optimizer setup cell, just set a default:

```python
# Cell: Quantum Optimization Setup with QAOA

import numpy as np
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

print("="*80)
print("QUANTUM OPTIMIZATION SETUP")
print("="*80)

# NumPy solver (classical)
numpy_solver = NumPyMinimumEigensolver()
numpy_optimizer = MinimumEigenOptimizer(numpy_solver)

# QAOA (quantum)
qaoa_depth = 3
sampler = StatevectorSampler()
qaoa_classical_opt = COBYLA(maxiter=100)
qaoa = QAOA(sampler=sampler, optimizer=qaoa_classical_opt, reps=qaoa_depth)
qaoa_optimizer = MinimumEigenOptimizer(qaoa)

# Set default optimizer (will be changed later based on problem size)
optimizer = numpy_optimizer  # Default to NumPy

print("✅ Both optimizers configured")
print("   Default: NumPy solver (will auto-select based on problem size)")
print("="*80)
```

## ✅ Solution 2: Smart Selection AFTER Data Creation

Put this code in a cell AFTER you've created your satellite data:

```python
# Cell: Smart Optimizer Selection (run AFTER creating satellite data)

# Check if we have satellite data
if 'large_sat_ids' in dir():
    satellite_ids = large_sat_ids
elif 'sat_ids' in dir():
    satellite_ids = sat_ids
else:
    print("⚠️  No satellite data found yet")
    print("   Using NumPy optimizer by default")
    optimizer = numpy_optimizer

# If we have satellite data, choose optimizer intelligently
if 'satellite_ids' in dir():
    n_vars = len(satellite_ids)
    qaoa_limit = 12  # QAOA can handle ~12 variables locally
    
    print("="*80)
    print("SMART OPTIMIZER SELECTION")
    print("="*80)
    print(f"\n📊 Problem size: {n_vars} satellites/variables")
    print(f"   QAOA local limit: ~{qaoa_limit} variables")
    
    if n_vars > qaoa_limit:
        print(f"\n⚠️  Problem too large for QAOA simulation")
        print(f"   Switching to NumPy solver (classical)")
        optimizer = numpy_optimizer
        optimizer_name = "NumPy (Classical)"
    else:
        print(f"\n✅ Problem size suitable for QAOA")
        print(f"   Using quantum circuit optimization")
        optimizer = qaoa_optimizer
        optimizer_name = "QAOA (Quantum)"
    
    print(f"\n🔷 Selected optimizer: {optimizer_name}")
    print("="*80)
```

## ✅ Solution 3: Inline Selection (Best)

Put this code RIGHT BEFORE you run the optimization:

```python
# Cell: Run Optimization with Smart Selection

print("="*80)
print("RUNNING QUANTUM OPTIMIZATION")
print("="*80)

# Determine problem size from your QUBO or satellite data
# Adjust these variable names based on what you have:
if 'large_sat_ids' in dir():
    n_vars = len(large_sat_ids)
elif 'qubo_problem' in dir():
    n_vars = qubo_problem.get_num_vars()
elif 'conflict_matrix' in dir():
    n_vars = conflict_matrix.shape[0]
else:
    n_vars = 100  # Default assumption

print(f"\n📊 Problem size: {n_vars} variables")

# Smart optimizer selection
qaoa_limit = 12
if n_vars > qaoa_limit:
    print(f"   Too large for QAOA → Using NumPy solver")
    optimizer = numpy_optimizer
else:
    print(f"   Suitable for QAOA → Using quantum optimizer")
    optimizer = qaoa_optimizer

# Now run the optimization
print(f"\n⏱️  Starting optimization...")
result = optimizer.solve(qubo_problem)

print(f"✅ Optimization complete!")
print(f"   Objective value: {result.fval:.4f}")
print("="*80)
```

## 📋 Complete Working Example

Here's a complete cell that checks for various variable names:

```python
# Cell: Flexible Optimizer Selection

print("="*80)
print("OPTIMIZER SELECTION")
print("="*80)

# Try to determine problem size from available variables
n_vars = None

# Check different possible variable names
if 'large_sat_ids' in dir():
    n_vars = len(large_sat_ids)
    print(f"✓ Found large_sat_ids: {n_vars} satellites")
elif 'sat_ids' in dir():
    n_vars = len(sat_ids)
    print(f"✓ Found sat_ids: {n_vars} satellites")
elif 'satellite_ids' in dir():
    n_vars = len(satellite_ids)
    print(f"✓ Found satellite_ids: {n_vars} satellites")
elif 'qubo_problem' in dir():
    n_vars = qubo_problem.get_num_vars()
    print(f"✓ Found qubo_problem: {n_vars} variables")
else:
    print("⚠️  No satellite data found yet")
    print("   Defaulting to NumPy optimizer")
    optimizer = numpy_optimizer
    n_vars = None

# If we found the problem size, choose optimizer
if n_vars is not None:
    qaoa_limit = 12
    
    print(f"\n📊 Problem analysis:")
    print(f"   Variables: {n_vars}")
    print(f"   QAOA limit: {qaoa_limit}")
    print(f"   Search space: 2^{n_vars} = {2**n_vars:,} configurations")
    
    if n_vars > qaoa_limit:
        print(f"\n✅ Using NumPy solver (problem too large for QAOA)")
        optimizer = numpy_optimizer
    else:
        print(f"\n✅ Using QAOA (problem size suitable)")
        optimizer = qaoa_optimizer

print("="*80)
```

## 💡 Recommendation

**Use Solution 1**: Just set `optimizer = numpy_optimizer` as the default in your setup cell, and don't worry about smart selection. NumPy solver works great for all problem sizes!

---

**The simplest fix: Just use NumPy optimizer everywhere!** 🚀
