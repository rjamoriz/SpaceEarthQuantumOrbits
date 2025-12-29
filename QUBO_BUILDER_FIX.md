# 🔧 EnhancedQUBOBuilder Parameter Fix

## The Problem

The `EnhancedQUBOBuilder` is being called with incorrect parameter names. The class expects different parameter names than what's being passed.

## ✅ Quick Fix

Check what parameters `EnhancedQUBOBuilder` actually expects. Common parameter names are:
- `n_sats` (not `num_satellites`)
- `n_bits` (not `num_bits`)
- `altitude_range` (not `alt_range`)

## 🔍 Find the Correct Parameters

First, check the class definition to see what parameters it expects:

```python
# Check the class signature
import inspect
print(inspect.signature(EnhancedQUBOBuilder.__init__))
```

## 📋 Common Fix Options

### Option 1: Use Standard Parameter Names
```python
# Most likely correct usage
qubo_builder = EnhancedQUBOBuilder(
    n_sats=5,           # Number of satellites (not num_satellites)
    n_bits=3,           # Bits per altitude encoding
    altitude_range=(400, 600),  # Min and max altitude in km
    atm_model=atm_model,
    thermal_model=thermal_model,
    eclipse_model=eclipse_model,
    debris_model=debris_model
)
```

### Option 2: Check Class Definition
```python
# If EnhancedQUBOBuilder is defined in your notebook, find it and check __init__
# Look for the cell that defines the class and see what parameters it expects
```

## 🎯 Complete Working Example

Without seeing the exact class definition, here's a typical working version:

```python
# Cell: Create QUBO Builder

print("="*80)
print("CREATING ENHANCED QUBO BUILDER")
print("="*80)

# Define constellation parameters
n_satellites = 5        # Number of satellites
n_bits_altitude = 3     # Bits for altitude encoding (2^3 = 8 altitude levels)
min_altitude = 400      # km
max_altitude = 600      # km

print(f"\n📊 Constellation parameters:")
print(f"   Satellites: {n_satellites}")
print(f"   Altitude encoding: {n_bits_altitude} bits ({2**n_bits_altitude} levels)")
print(f"   Altitude range: {min_altitude}-{max_altitude} km")

# Create QUBO builder with correct parameter names
try:
    qubo_builder = EnhancedQUBOBuilder(
        n_sats=n_satellites,              # Use n_sats, not num_satellites
        n_bits=n_bits_altitude,           # Use n_bits, not num_bits
        altitude_range=(min_altitude, max_altitude),
        atm_model=atm_model,
        thermal_model=thermal_model,
        eclipse_model=eclipse_model,
        debris_model=debris_model
    )
    
    print(f"\n✅ EnhancedQUBOBuilder created successfully")
    print(f"   Total variables: {qubo_builder.n_vars}")
    print(f"   Problem size: 2^{qubo_builder.n_vars} = {2**qubo_builder.n_vars:,} configurations")
    
except TypeError as e:
    print(f"\n❌ Error creating QUBO builder: {e}")
    print(f"\n💡 Checking class signature...")
    import inspect
    sig = inspect.signature(EnhancedQUBOBuilder.__init__)
    print(f"   Expected parameters: {sig}")
    print(f"\n   Please use the correct parameter names shown above")
    raise

print("="*80)
```

## 🔍 Debug: Find the Class Definition

Run this to find where `EnhancedQUBOBuilder` is defined:

```python
# Find the class definition
import inspect

print("EnhancedQUBOBuilder location:")
print(inspect.getfile(EnhancedQUBOBuilder))

print("\nClass signature:")
print(inspect.signature(EnhancedQUBOBuilder.__init__))

print("\nClass docstring:")
print(EnhancedQUBOBuilder.__doc__)
```

## 📝 Common Parameter Name Patterns

Different implementations might use:

| Your Code | Possible Correct Name |
|-----------|----------------------|
| `num_satellites` | `n_sats` or `num_sats` |
| `num_bits` | `n_bits` or `bits_per_altitude` |
| `alt_range` | `altitude_range` |
| `min_alt, max_alt` | `altitude_range=(min, max)` |

## 🎯 Most Likely Fix

Based on common patterns, try this:

```python
# Replace your current call with this:
qubo_builder = EnhancedQUBOBuilder(
    n_sats=5,                      # Changed from num_satellites
    n_bits=3,                      # Changed from num_bits  
    altitude_range=(400, 600),     # Changed from separate min/max
    atm_model=atm_model,
    thermal_model=thermal_model,
    eclipse_model=eclipse_model,
    debris_model=debris_model
)
```

## 💡 If Still Not Working

If the above doesn't work, you need to find the class definition in your notebook:

1. Search for `class EnhancedQUBOBuilder` in your notebook
2. Look at the `__init__` method signature
3. Use the exact parameter names from that definition

Or share the error line and I can provide the exact fix!

---

**Try using `n_sats` instead of `num_satellites`!** 🚀
