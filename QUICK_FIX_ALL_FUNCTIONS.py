"""
QUICK FIX: Copy this entire block to the TOP of your cell 14
This includes ALL 4 helper functions you need
"""

# ============================================================================
# ALL HELPER FUNCTIONS - ADD TO TOP OF CELL 14
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


print("✅ All 4 helper functions loaded!")
print("   - calculate_distance()")
print("   - create_conflict_matrix()")
print("   - greedy_partition()")
print("   - calc_stats()")

# ============================================================================
# NOW ADD YOUR CELL CODE BELOW THIS LINE
# ============================================================================
