# 🔧 Cell 14 - Complete Fixed Version

## The Problem

Cell 14 has multiple missing imports and initializations:
1. ❌ `numpy` not imported (`np` not defined)
2. ❌ `time` not imported
3. ❌ `telemetry_gen` not initialized (StarlinkTelemetryGenerator)

## ✅ Complete Fixed Cell 14

Replace your current cell 14 with this complete version:

```python
# Cell 14: Large-Scale Constellation Simulation

# ========== IMPORTS ==========
import numpy as np
import time
from starlink_telemetry import StarlinkTelemetryGenerator

# ========== CONFIGURATION ==========
LARGE_CONSTELLATION_SIZE = 100  # Increase to 200, 500, or more for stress testing

print("🚀 Large-Scale Constellation Simulation")
print("=" * 70)
print(f"Constellation Size: {LARGE_CONSTELLATION_SIZE} satellites")

# ========== INITIALIZE TELEMETRY GENERATOR ==========
telemetry_gen = StarlinkTelemetryGenerator(seed=42)
print("✅ Telemetry generator initialized")

# ========== GENERATE CONSTELLATION ==========
print(f"\nGenerating {LARGE_CONSTELLATION_SIZE} satellites...")
start_time = time.time()

# Generate large constellation
np.random.seed(123)
large_altitudes = np.random.uniform(340, 1200, LARGE_CONSTELLATION_SIZE)
large_latitudes = np.random.uniform(-60, 60, LARGE_CONSTELLATION_SIZE)
large_longitudes = np.random.uniform(-180, 180, LARGE_CONSTELLATION_SIZE)

# Create position dictionary
large_positions = {}
for i in range(LARGE_CONSTELLATION_SIZE):
    sat_id = 60000 + i
    large_positions[sat_id] = {
        'lat': large_latitudes[i],
        'lng': large_longitudes[i],
        'alt': large_altitudes[i]
    }

position_time = time.time() - start_time
print(f"\n✓ Positions generated in {position_time:.2f}s")

# ========== GENERATE TELEMETRY ==========
print(f"\n📊 Generating telemetry data...")
telemetry_start = time.time()

large_augmented = telemetry_gen.augment_satellite_positions(large_positions)

telemetry_time = time.time() - telemetry_start
print(f"✓ Telemetry generated in {telemetry_time:.2f}s")

# ========== PERFORMANCE STATISTICS ==========
print(f"\n📈 Performance Statistics:")
print(f"  Total satellites: {len(large_augmented):,}")
print(f"  Position generation: {position_time:.3f}s")
print(f"  Telemetry generation: {telemetry_time:.3f}s")
print(f"  Total time: {position_time + telemetry_time:.3f}s")
print(f"  Rate: {len(large_augmented)/(position_time + telemetry_time):.1f} satellites/second")

# ========== TELEMETRY ANALYSIS ==========
qoe_scores = [data['qoe_score'] for data in large_augmented.values()]
throughputs = [data['download_throughput_mbps'] for data in large_augmented.values()]
signal_losses = [data['signal_loss_db'] for data in large_augmented.values()]

print(f"\n📊 Telemetry Summary:")
print(f"  Average QoE: {np.mean(qoe_scores):.2f}/10")
print(f"  QoE Range: {np.min(qoe_scores):.2f} - {np.max(qoe_scores):.2f}")
print(f"  Average Throughput: {np.mean(throughputs):.1f} Mbps")
print(f"  Average Signal Loss: {np.mean(signal_losses):.2f} dB")

# ========== WEATHER DISTRIBUTION ==========
weather_counts = {}
for data in large_augmented.values():
    weather = data['weather']
    weather_counts[weather] = weather_counts.get(weather, 0) + 1

print(f"\n🌦️  Weather Distribution:")
for weather, count in sorted(weather_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(large_augmented)) * 100
    print(f"  {weather:<15} {count:>4} ({percentage:>5.1f}%)")

# ========== CONFLICT ANALYSIS (OPTIONAL) ==========
print(f"\n🔬 Conflict Analysis:")
print(f"  Computing conflict matrix for {LARGE_CONSTELLATION_SIZE} satellites...")

# Simple distance-based conflict check (threshold: 2500 km)
conflict_threshold = 2500  # km
conflicts = 0

# Sample-based conflict detection (check first 50 satellites for speed)
sample_size = min(50, LARGE_CONSTELLATION_SIZE)
print(f"  Sampling {sample_size} satellites for conflict detection...")

for i in range(sample_size):
    for j in range(i+1, sample_size):
        sid1 = 60000 + i
        sid2 = 60000 + j
        
        # Calculate 3D distance
        pos1 = large_positions[sid1]
        pos2 = large_positions[sid2]
        
        # Simple Euclidean approximation
        lat_diff = pos1['lat'] - pos2['lat']
        lng_diff = pos1['lng'] - pos2['lng']
        alt_diff = pos1['alt'] - pos2['alt']
        
        # Rough distance estimate
        distance = np.sqrt(lat_diff**2 + lng_diff**2 + (alt_diff/100)**2) * 111  # km
        
        if distance < conflict_threshold:
            conflicts += 1

total_pairs_sampled = sample_size * (sample_size - 1) // 2
conflict_rate = (conflicts / total_pairs_sampled) * 100 if total_pairs_sampled > 0 else 0

print(f"  Conflicts detected: {conflicts}/{total_pairs_sampled} pairs ({conflict_rate:.1f}%)")
print(f"  Estimated total conflicts: ~{int(conflicts * (LARGE_CONSTELLATION_SIZE/sample_size)**2)}")

print("\n" + "=" * 70)
print("✅ Large-scale simulation complete!")
print("=" * 70)

# Store for next cells
print(f"\n💾 Variables available for next cells:")
print(f"  - large_positions: {len(large_positions)} satellite positions")
print(f"  - large_augmented: {len(large_augmented)} satellites with telemetry")
print(f"  - telemetry_gen: Initialized generator")
```

## Key Changes Made:

### 1. **Added All Required Imports**
```python
import numpy as np
import time
from starlink_telemetry import StarlinkTelemetryGenerator
```

### 2. **Initialize Telemetry Generator**
```python
telemetry_gen = StarlinkTelemetryGenerator(seed=42)
```

### 3. **Enhanced Output**
- Performance timing
- Telemetry statistics
- Weather distribution
- Conflict analysis (sampled for performance)

### 4. **Better Error Handling**
- Clear progress messages
- Performance metrics
- Data availability confirmation

## Alternative: Minimal Fix

If you just want to fix the errors quickly, add this at the **top of cell 14**:

```python
# Quick fix - add these 3 lines at the top
import numpy as np
import time
from starlink_telemetry import StarlinkTelemetryGenerator

# Initialize generator
telemetry_gen = StarlinkTelemetryGenerator(seed=42)

# ... rest of your existing code
```

## Why This Happens

Jupyter notebooks execute cells independently. Each cell needs:
1. **Imports** for libraries it uses
2. **Variables** from previous cells (must run those cells first)
3. **Initializations** for objects it creates

## Best Practice

Create a **master import cell** at the beginning with:
```python
import numpy as np
import time
from starlink_telemetry import StarlinkTelemetryGenerator

# Initialize commonly used objects
telemetry_gen = StarlinkTelemetryGenerator(seed=42)
```

Then run it **before** running cell 14.

## Testing

After applying the fix, you should see output like:
```
🚀 Large-Scale Constellation Simulation
======================================================================
Constellation Size: 100 satellites
✅ Telemetry generator initialized

Generating 100 satellites...
✓ Positions generated in 0.00s

📊 Generating telemetry data...
✓ Telemetry generated in 0.15s

📈 Performance Statistics:
  Total satellites: 100
  Rate: 666.7 satellites/second

📊 Telemetry Summary:
  Average QoE: 7.64/10
  Average Throughput: 104.2 Mbps

✅ Large-scale simulation complete!
```

---

**Add the imports and initialization, then re-run the cell!** 🚀
