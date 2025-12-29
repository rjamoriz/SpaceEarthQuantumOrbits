# 🔧 Propellant Budget Not Defined Fix

## The Problem

The `ScenarioAnalyzer` is trying to use `propellant_budget` but it hasn't been defined yet.

## ✅ Quick Fix

Add this code BEFORE initializing the `ScenarioAnalyzer`:

```python
# Define propellant budget (kg per satellite)
propellant_budget = 50.0  # kg - typical for small LEO satellites

print(f"✅ Propellant budget set: {propellant_budget} kg per satellite")
```

## 📋 Complete Fix with Context

Add this block before the `ScenarioAnalyzer` initialization:

```python
# Define mission parameters
propellant_budget = 50.0  # kg per satellite
mission_lifetime = 5.0    # years
max_altitude = 2000.0     # km

print("="*80)
print("MISSION PARAMETERS")
print("="*80)
print(f"  Propellant budget: {propellant_budget} kg/satellite")
print(f"  Mission lifetime: {mission_lifetime} years")
print(f"  Max altitude: {max_altitude} km")
print("="*80)

# Now initialize scenario analyzer
scenario_analyzer = ScenarioAnalyzer(
    atm_model, thermal_model, eclipse_model, debris_model, propellant_budget
)
```

## 🎯 Better: Check All Required Variables

Use this safer version that checks for all dependencies:

```python
# Check and define all required variables
print("🔍 Checking required variables...")

# Define propellant budget if not exists
if 'propellant_budget' not in dir():
    propellant_budget = 50.0  # kg
    print(f"✅ Defined propellant_budget: {propellant_budget} kg")
else:
    print(f"✅ propellant_budget already defined: {propellant_budget} kg")

# Check for required models
required_models = ['atm_model', 'thermal_model', 'eclipse_model', 'debris_model']
missing_models = [model for model in required_models if model not in dir()]

if missing_models:
    print(f"\n⚠️  WARNING: Missing models: {missing_models}")
    print("   Make sure you've run the model initialization cells first!")
    raise NameError(f"Required models not defined: {missing_models}")
else:
    print(f"✅ All required models are defined")

print("\n" + "="*80)
print("INITIALIZING SCENARIO ANALYZER")
print("="*80)

# Now safe to initialize
scenario_analyzer = ScenarioAnalyzer(
    atm_model, thermal_model, eclipse_model, debris_model, propellant_budget
)

print("✅ ScenarioAnalyzer initialized successfully")
print("="*80)
```

## 📊 Typical Propellant Budget Values

Choose based on your satellite class:

```python
# CubeSat (1-10 kg)
propellant_budget = 0.5  # kg

# Small satellite (10-100 kg)
propellant_budget = 5.0  # kg

# Medium satellite (100-500 kg) - Starlink class
propellant_budget = 50.0  # kg

# Large satellite (500+ kg)
propellant_budget = 200.0  # kg
```

## 🔧 Complete Working Cell

Replace your cell with this complete version:

```python
# Cell: Scenario Analysis Setup

import numpy as np

print("="*80)
print("SCENARIO ANALYSIS SETUP")
print("="*80)

# ========== DEFINE MISSION PARAMETERS ==========
print("\n📊 Setting mission parameters...")

# Propellant budget (kg per satellite)
propellant_budget = 50.0  # Typical for Starlink-class satellites

# Other mission parameters
mission_lifetime = 5.0    # years
max_altitude = 2000.0     # km
min_altitude = 300.0      # km

print(f"✅ Mission parameters defined:")
print(f"   Propellant budget: {propellant_budget} kg/satellite")
print(f"   Mission lifetime: {mission_lifetime} years")
print(f"   Altitude range: {min_altitude}-{max_altitude} km")

# ========== VERIFY REQUIRED MODELS ==========
print("\n🔍 Verifying required models...")

required_models = {
    'atm_model': 'Atmospheric drag model',
    'thermal_model': 'Thermal environment model',
    'eclipse_model': 'Eclipse/solar model',
    'debris_model': 'Space debris model'
}

all_models_ok = True
for model_name, description in required_models.items():
    if model_name in dir():
        print(f"   ✅ {description}: OK")
    else:
        print(f"   ❌ {description}: MISSING")
        all_models_ok = False

if not all_models_ok:
    print("\n⚠️  ERROR: Some required models are missing!")
    print("   Please run the model initialization cells first.")
    raise NameError("Required models not defined")

# ========== DEFINE SCENARIO ANALYZER CLASS ==========
print("\n🔧 Defining ScenarioAnalyzer class...")

class ScenarioAnalyzer:
    def __init__(self, atm_model, thermal_model, eclipse_model, debris_model, propellant_budget):
        self.atm_model = atm_model
        self.thermal_model = thermal_model
        self.eclipse_model = eclipse_model
        self.debris_model = debris_model
        self.propellant_budget = propellant_budget
        
        print(f"✅ ScenarioAnalyzer initialized")
        print(f"   Propellant budget: {propellant_budget} kg")
    
    def analyze_altitude_constraints(self, min_alt=300, max_alt=2000, step=50):
        """Analyze constraints across altitude range"""
        altitudes = np.arange(min_alt, max_alt + step, step)
        
        allowed_altitudes = []
        forbidden_altitudes = []
        
        for alt in altitudes:
            try:
                # Check various constraints
                lifetime = self.atm_model.orbital_lifetime(alt * 1e3)
                debris_risk = self.debris_model.collision_probability(alt)
                temp_sun = self.thermal_model.temperature_sunlit(alt * 1e3)
                
                # Define constraints
                is_allowed = (
                    lifetime >= 5.0 and  # At least 5 years
                    debris_risk < 1e-3 and  # Low debris risk
                    temp_sun < 400  # Below thermal limit (K)
                )
                
                if is_allowed:
                    allowed_altitudes.append(alt)
                else:
                    forbidden_altitudes.append(alt)
                    
            except Exception as e:
                print(f"⚠️  Error at {alt} km: {e}")
                forbidden_altitudes.append(alt)
        
        return {
            'allowed': allowed_altitudes,
            'forbidden': forbidden_altitudes,
            'total': len(altitudes)
        }

# ========== INITIALIZE SCENARIO ANALYZER ==========
print("\n🚀 Creating ScenarioAnalyzer instance...")

scenario_analyzer = ScenarioAnalyzer(
    atm_model, thermal_model, eclipse_model, debris_model, propellant_budget
)

print("\n" + "="*80)
print("✅ SCENARIO ANALYZER READY")
print("="*80)

# ========== RUN INITIAL ANALYSIS ==========
print("\n📊 Running initial altitude constraint analysis...")

try:
    constraints = scenario_analyzer.analyze_altitude_constraints(
        min_alt=300, max_alt=2000, step=100
    )
    
    print(f"\n✅ Constraint analysis complete:")
    print(f"   Total altitudes tested: {constraints['total']}")
    print(f"   Allowed altitudes: {len(constraints['allowed'])}")
    print(f"   Forbidden altitudes: {len(constraints['forbidden'])}")
    
    if constraints['allowed']:
        print(f"\n   Allowed altitude range:")
        print(f"   {min(constraints['allowed'])} - {max(constraints['allowed'])} km")
    
except Exception as e:
    print(f"⚠️  Error during analysis: {e}")

print("\n" + "="*80)
```

---

**Add `propellant_budget = 50.0` before initializing ScenarioAnalyzer!** 🚀
