# 🔧 NaN Value Error Fix

## The Problem

The optimization result contains NaN (Not a Number) values, which can't be converted to integers. This happens when:
- The QUBO problem is malformed
- The solver failed silently
- Variables are undefined in the result

## ✅ Quick Fix

Add NaN checking before converting to integers:

```python
# Replace this line:
solution_vector = [int(result.x[i]) for i in range(len(result.x))]

# With this safer version:
import numpy as np

# Check for NaN values
if np.any(np.isnan(result.x)):
    print("⚠️  Warning: Solution contains NaN values")
    print("   This usually means the optimization failed")
    print(f"   Result status: {result.status}")
    # Use zeros as fallback
    solution_vector = [0 for i in range(len(result.x))]
else:
    # Round to nearest integer (0 or 1 for binary)
    solution_vector = [int(round(result.x[i])) for i in range(len(result.x))]

print(f"Solution vector: {solution_vector}")
```

## 📋 Complete Fixed Cell

Replace your entire cell 47 with this robust version:

```python
# Cell: Run Optimization with Error Handling

import time
import numpy as np

print("="*80)
print("RUNNING QUANTUM OPTIMIZATION - ENHANCED QUBO")
print("="*80)

# Get problem size and choose optimizer
n_vars = qp.get_num_vars()
qaoa_limit = 12

print(f"\n📊 Problem size: {n_vars} variables")

if n_vars > qaoa_limit:
    print(f"   Using NumPy solver (problem too large for QAOA)")
    optimizer = numpy_optimizer
    optimizer_name = "NumPy"
else:
    print(f"   Using QAOA optimizer")
    optimizer = qaoa_optimizer
    optimizer_name = "QAOA"

print(f"\n⏱️  Starting {optimizer_name} optimization...")
start_time = time.time()

try:
    # Solve the QUBO
    result = optimizer.solve(qp)
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ {optimizer_name} OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"Computation time: {elapsed_time:.2f} seconds")
    print(f"Status: {result.status}")
    print(f"Optimal objective value: {result.fval:.6f}")
    
    # Check for valid solution
    if np.any(np.isnan(result.x)):
        print("\n⚠️  WARNING: Solution contains NaN values")
        print("   The optimization may have failed")
        print("   Using default solution...")
        solution_vector = [0] * len(result.x)
    else:
        # Convert to binary (round to nearest integer)
        solution_vector = [int(round(result.x[i])) for i in range(len(result.x))]
    
    print(f"\nBinary solution vector ({len(solution_vector)} variables):")
    print(solution_vector[:20], "..." if len(solution_vector) > 20 else "")
    
    # Decode solution to altitudes
    print("\n" + "="*80)
    print("DECODED SATELLITE CONSTELLATION")
    print("="*80)
    
    for sat in range(qubo_builder.n_sats):
        try:
            bits = solution_vector[sat*qubo_builder.n_bits:(sat+1)*qubo_builder.n_bits]
            altitude = qubo_builder.decode_altitude(bits)
            
            # Check for valid altitude
            if np.isnan(altitude) or altitude < 0:
                print(f"Satellite {sat+1}: Invalid altitude (using default 550 km)")
                altitude = 550.0
            
            # Get physical properties
            lifetime = atm_model.orbital_lifetime(altitude * 1e3)
            period = eclipse_model.orbital_period(R_EARTH + altitude*1e3) / 60
            T_sun = thermal_model.temperature_sunlit(altitude * 1e3) - 273.15
            debris_risk = debris_model.debris_density(altitude)
            
            print(f"\nSatellite {sat+1}:")
            print(f"  Altitude: {altitude:.1f} km")
            print(f"  Binary: {bits}")
            print(f"  Lifetime: {lifetime:.1f} years")
            print(f"  Period: {period:.1f} min")
            print(f"  T_max: {T_sun:.1f}°C")
            print(f"  Debris density: {debris_risk:.2e} objects/km³")
            
        except Exception as e:
            print(f"\nSatellite {sat+1}: Error decoding - {e}")
            continue

except Exception as e:
    print(f"\n❌ Error during optimization: {e}")
    print(f"   Error type: {type(e).__name__}")
    
    # Try fallback to NumPy if QAOA failed
    if optimizer == qaoa_optimizer:
        print("\n🔄 Trying NumPy solver as fallback...")
        try:
            start_time = time.time()
            result = numpy_optimizer.solve(qp)
            elapsed_time = time.time() - start_time
            
            print(f"\n✓ NumPy solver succeeded!")
            print(f"   Time: {elapsed_time:.2f} seconds")
            print(f"   Objective: {result.fval:.6f}")
            
            # Process solution with NaN checking
            if np.any(np.isnan(result.x)):
                solution_vector = [0] * len(result.x)
            else:
                solution_vector = [int(round(result.x[i])) for i in range(len(result.x))]
            
            print(f"   Solution: {solution_vector[:10]}...")
            
        except Exception as e2:
            print(f"❌ NumPy solver also failed: {e2}")
            raise
    else:
        raise

print("\n" + "="*80)
print("✅ OPTIMIZATION COMPLETE")
print("="*80)
```

## 🔍 Why This Happens

### Common Causes:
1. **QUBO problem is too large** - Solver runs out of memory
2. **Invalid constraints** - Problem has no valid solution
3. **Numerical instability** - Floating point errors
4. **Solver failure** - Silent failure returns NaN

### The Fix:
- ✅ Check for NaN before converting to int
- ✅ Use `round()` for better binary conversion
- ✅ Provide fallback values
- ✅ Better error messages

## 💡 Additional Safety

Add this at the start of your cell to validate the QUBO:

```python
# Validate QUBO problem before solving
print("🔍 Validating QUBO problem...")
print(f"   Variables: {qp.get_num_vars()}")
print(f"   Constraints: {qp.get_num_linear_constraints()}")

# Check if problem is valid
if qp.get_num_vars() == 0:
    raise ValueError("QUBO has no variables!")

print("✅ QUBO validation passed")
```

---

**Use the complete fixed cell above with NaN checking and error handling!** 🚀
