# 🔧 All Helper Functions for Notebook Cells

## Complete Set of Helper Functions

Add these functions to your notebook to fix all `NameError` issues. You can either:
1. **Add them to a master cell** at the beginning
2. **Add them to individual cells** that need them

---

## Function 1: `calculate_distance`

Calculates 3D distance between two satellites.

```python
import math

def calculate_distance(pos1, pos2):
    """
    Calculates the straight-line distance between two satellites in 3D space.
    
    Args:
        pos1: Dictionary with 'lat', 'lng', 'alt' keys
        pos2: Dictionary with 'lat', 'lng', 'alt' keys
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in km
    
    # Convert lat/lng/alt to Cartesian coordinates
    def to_cartesian(lat, lng, alt):
        lat_rad = math.radians(lat)
        lng_rad = math.radians(lng)
        x = (R + alt) * math.cos(lat_rad) * math.cos(lng_rad)
        y = (R + alt) * math.cos(lat_rad) * math.sin(lng_rad)
        z = (R + alt) * math.sin(lat_rad)
        return x, y, z
    
    x1, y1, z1 = to_cartesian(pos1['lat'], pos1['lng'], pos1['alt'])
    x2, y2, z2 = to_cartesian(pos2['lat'], pos2['lng'], pos2['alt'])
    
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    return distance
```

---

## Function 2: `create_conflict_matrix`

Creates performance-weighted conflict matrix.

```python
import numpy as np

def create_conflict_matrix(positions, augmented_data, threshold_km=3000, use_performance=True):
    """
    Creates an adjacency matrix representing conflicts between satellites.
    
    Args:
        positions: Dictionary of satellite positions {sat_id: {'lat', 'lng', 'alt'}}
        augmented_data: Dictionary of telemetry data {sat_id: {...}}
        threshold_km: Distance threshold for conflicts (default: 3000 km)
        use_performance: If True, weight conflicts by QoE performance
    
    Returns:
        tuple: (conflict_matrix, list_of_sat_ids)
    """
    sat_ids = list(positions.keys())
    num_sats = len(sat_ids)
    adj_matrix = np.zeros((num_sats, num_sats))
    
    for i in range(num_sats):
        for j in range(i + 1, num_sats):
            sat_id1 = sat_ids[i]
            sat_id2 = sat_ids[j]
            distance = calculate_distance(positions[sat_id1], positions[sat_id2])
            
            if distance < threshold_km:
                # Base conflict weight
                conflict_weight = 1.0
                
                # Weight by performance if telemetry data available
                if use_performance and augmented_data:
                    perf1 = augmented_data[sat_id1].get('qoe_score', 5.0) / 10.0
                    perf2 = augmented_data[sat_id2].get('qoe_score', 5.0) / 10.0
                    
                    # Lower performance = higher conflict weight
                    avg_perf = (perf1 + perf2) / 2.0
                    conflict_weight = 1.0 + (1.0 - avg_perf) * 2.0  # Range: 1.0 to 3.0
                
                adj_matrix[i, j] = conflict_weight
                adj_matrix[j, i] = conflict_weight  # Symmetric matrix
    
    return adj_matrix, sat_ids
```

---

## Function 3: `greedy_partition`

Greedy algorithm to partition satellites into two orbit sets.

```python
def greedy_partition(conflict_matrix, sat_ids):
    """
    Greedy algorithm to partition satellites into two orbit sets.
    Minimizes conflicts by assigning each satellite to the set with fewer conflicts.
    
    Args:
        conflict_matrix: NxN numpy array of conflict weights
        sat_ids: List of satellite IDs
    
    Returns:
        tuple: (indices_set_A, indices_set_B)
    """
    n = len(sat_ids)
    set_A = [0]  # Start with first satellite in set A
    set_B = []
    
    for i in range(1, n):
        # Calculate conflicts with each set
        conflict_A = sum(conflict_matrix[i][j] for j in set_A)
        conflict_B = sum(conflict_matrix[i][j] for j in set_B) if set_B else 0
        
        # Assign to set with fewer conflicts
        if conflict_A <= conflict_B or not set_B:
            set_B.append(i)
        else:
            set_A.append(i)
    
    return set_A, set_B
```

---

## Function 4: `calc_stats` (Optional)

Calculate performance statistics for a set of satellites.

```python
import numpy as np

def calc_stats(sat_ids, augmented_data):
    """
    Calculate performance statistics for a set of satellites.
    
    Args:
        sat_ids: List of satellite IDs
        augmented_data: Dictionary of telemetry data
    
    Returns:
        Dictionary with statistics
    """
    if not sat_ids:
        return {
            'qoe': 0,
            'throughput': 0,
            'packet_loss': 0,
            'signal_loss': 0
        }
    
    return {
        'qoe': np.mean([augmented_data[sid]['qoe_score'] for sid in sat_ids]),
        'throughput': np.mean([augmented_data[sid]['download_throughput_mbps'] for sid in sat_ids]),
        'packet_loss': np.mean([augmented_data[sid]['packet_loss_percent'] for sid in sat_ids]),
        'signal_loss': np.mean([augmented_data[sid]['signal_loss_db'] for sid in sat_ids])
    }
```

---

## 📦 Complete Master Function Cell

Copy this entire cell and place it **early in your notebook** (after imports):

```python
# ============================================================================
# HELPER FUNCTIONS FOR SATELLITE OPTIMIZATION
# ============================================================================

import math
import numpy as np

def calculate_distance(pos1, pos2):
    """Calculate 3D distance between two satellites"""
    R = 6371  # Earth's radius in km
    
    def to_cartesian(lat, lng, alt):
        lat_rad = math.radians(lat)
        lng_rad = math.radians(lng)
        x = (R + alt) * math.cos(lat_rad) * math.cos(lng_rad)
        y = (R + alt) * math.cos(lat_rad) * math.sin(lng_rad)
        z = (R + alt) * math.sin(lat_rad)
        return x, y, z
    
    x1, y1, z1 = to_cartesian(pos1['lat'], pos1['lng'], pos1['alt'])
    x2, y2, z2 = to_cartesian(pos2['lat'], pos2['lng'], pos2['alt'])
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def create_conflict_matrix(positions, augmented_data, threshold_km=3000, use_performance=True):
    """Create performance-weighted conflict matrix"""
    sat_ids = list(positions.keys())
    num_sats = len(sat_ids)
    adj_matrix = np.zeros((num_sats, num_sats))
    
    for i in range(num_sats):
        for j in range(i + 1, num_sats):
            sat_id1, sat_id2 = sat_ids[i], sat_ids[j]
            distance = calculate_distance(positions[sat_id1], positions[sat_id2])
            
            if distance < threshold_km:
                conflict_weight = 1.0
                if use_performance and augmented_data:
                    perf1 = augmented_data[sat_id1].get('qoe_score', 5.0) / 10.0
                    perf2 = augmented_data[sat_id2].get('qoe_score', 5.0) / 10.0
                    avg_perf = (perf1 + perf2) / 2.0
                    conflict_weight = 1.0 + (1.0 - avg_perf) * 2.0
                adj_matrix[i, j] = conflict_weight
                adj_matrix[j, i] = conflict_weight
    
    return adj_matrix, sat_ids


def greedy_partition(conflict_matrix, sat_ids):
    """Greedy algorithm to partition satellites into two orbit sets"""
    n = len(sat_ids)
    set_A = [0]  # Start with first satellite
    set_B = []
    
    for i in range(1, n):
        conflict_A = sum(conflict_matrix[i][j] for j in set_A)
        conflict_B = sum(conflict_matrix[i][j] for j in set_B) if set_B else 0
        
        if conflict_A <= conflict_B or not set_B:
            set_B.append(i)
        else:
            set_A.append(i)
    
    return set_A, set_B


def calc_stats(sat_ids, augmented_data):
    """Calculate performance statistics for a set of satellites"""
    if not sat_ids:
        return {'qoe': 0, 'throughput': 0, 'packet_loss': 0, 'signal_loss': 0}
    
    return {
        'qoe': np.mean([augmented_data[sid]['qoe_score'] for sid in sat_ids]),
        'throughput': np.mean([augmented_data[sid]['download_throughput_mbps'] for sid in sat_ids]),
        'packet_loss': np.mean([augmented_data[sid]['packet_loss_percent'] for sid in sat_ids]),
        'signal_loss': np.mean([augmented_data[sid]['signal_loss_db'] for sid in sat_ids])
    }


print("✅ All helper functions loaded!")
print("   - calculate_distance()")
print("   - create_conflict_matrix()")
print("   - greedy_partition()")
print("   - calc_stats()")
```

---

## 🎯 Quick Fix for Current Cell

Just add this **one function** at the top of your current cell:

```python
def greedy_partition(conflict_matrix, sat_ids):
    """Greedy algorithm to partition satellites"""
    n = len(sat_ids)
    set_A = [0]
    set_B = []
    for i in range(1, n):
        conflict_A = sum(conflict_matrix[i][j] for j in set_A)
        conflict_B = sum(conflict_matrix[i][j] for j in set_B) if set_B else 0
        if conflict_A <= conflict_B or not set_B:
            set_B.append(i)
        else:
            set_A.append(i)
    return set_A, set_B
```

---

## 📋 Function Dependencies

| Function | Depends On | Used By |
|----------|-----------|---------|
| `calculate_distance` | `math` | `create_conflict_matrix` |
| `create_conflict_matrix` | `calculate_distance`, `numpy` | Optimization cells |
| `greedy_partition` | `numpy` (for matrix) | Partitioning cells |
| `calc_stats` | `numpy` | Analysis cells |

---

## ✅ Recommended Notebook Structure

```
Cell 1: Title (Markdown)
Cell 2: Master Imports
Cell 3: Helper Functions (this file)
Cell 4: Configuration
Cell 5+: Your analysis cells
```

This ensures all functions are available for all subsequent cells!

---

**Add the `greedy_partition` function and your cell will work!** 🚀
