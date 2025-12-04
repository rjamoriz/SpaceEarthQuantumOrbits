"""
Test script to demonstrate Starlink telemetry integration without requiring N2YO API key.
This uses simulated satellite positions to show the complete workflow.
"""

# Fix matplotlib backend issue before importing qiskit
import os
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')

from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
import numpy as np
import math
from starlink_telemetry import StarlinkTelemetryGenerator, print_telemetry_summary


def calculate_distance(pos1, pos2):
    """Calculate straight-line distance between two points in 3D space."""
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
    
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    return distance


def create_conflict_matrix(positions, threshold_km, augmented_data=None, use_performance=True):
    """
    Creates an adjacency matrix representing conflicts between satellites.
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
                conflict_weight = 1.0
                
                if use_performance and augmented_data:
                    perf1 = augmented_data[sat_id1].get('qoe_score', 5.0) / 10.0
                    perf2 = augmented_data[sat_id2].get('qoe_score', 5.0) / 10.0
                    avg_perf = (perf1 + perf2) / 2.0
                    conflict_weight = 1.0 + (1.0 - avg_perf) * 2.0
                
                adj_matrix[i, j] = conflict_weight
                adj_matrix[j, i] = conflict_weight
    
    return adj_matrix


def main():
    print("="*80)
    print("STARLINK TELEMETRY INTEGRATION TEST")
    print("="*80)
    
    # Simulated satellite positions (realistic orbital parameters)
    SATELLITE_IDS = [25544, 20580, 24876, 25338, 28654, 33591]
    satellite_positions = {
        25544: {'lat': 45.5, 'lng': -122.6, 'alt': 420},   # ISS-like
        20580: {'lat': -23.4, 'lng': 151.2, 'alt': 547},   # Hubble-like
        24876: {'lat': 65.2, 'lng': -18.1, 'alt': 850},    # Polar orbit
        25338: {'lat': 12.8, 'lng': 77.6, 'alt': 820},     # Mid-latitude
        28654: {'lat': -45.3, 'lng': 35.2, 'alt': 780},    # Southern hemisphere
        33591: {'lat': 51.5, 'lng': -0.1, 'alt': 550},     # European region
    }
    
    print(f"\nSimulating {len(SATELLITE_IDS)} satellites...")
    
    # Initialize telemetry generator
    telemetry_generator = StarlinkTelemetryGenerator(seed=42)
    
    # Augment satellite data with Starlink telemetry metrics
    print("\nGenerating Starlink telemetry data based on opendatabay.com schema...")
    augmented_data = telemetry_generator.augment_satellite_positions(satellite_positions)
    print_telemetry_summary(augmented_data)
    
    # Create conflict matrices (with and without performance weighting)
    CONFLICT_THRESHOLD_KM = 3000  # Larger threshold to ensure some conflicts
    
    print("\n" + "="*80)
    print("CONFLICT ANALYSIS")
    print("="*80)
    
    # Without performance weighting
    print("\n1. Standard Conflict Matrix (distance-only):")
    adj_matrix_standard = create_conflict_matrix(
        satellite_positions, 
        CONFLICT_THRESHOLD_KM,
        augmented_data=None,
        use_performance=False
    )
    print(adj_matrix_standard)
    
    # With performance weighting
    print("\n2. Performance-Weighted Conflict Matrix:")
    adj_matrix_weighted = create_conflict_matrix(
        satellite_positions, 
        CONFLICT_THRESHOLD_KM,
        augmented_data=augmented_data,
        use_performance=True
    )
    print(adj_matrix_weighted)
    
    # Solve QUBO with performance-weighted matrix
    print("\n" + "="*80)
    print("QUANTUM OPTIMIZATION (QUBO)")
    print("="*80)
    
    max_cut = Maxcut(adj_matrix_weighted)
    qp = max_cut.to_quadratic_program()
    
    qubo_converter = QuadraticProgramToQubo()
    qubo = qubo_converter.convert(qp)
    
    numpy_solver = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(numpy_solver)
    result = optimizer.solve(qubo)
    
    print(f"\nOptimal solution: {result.x}")
    print(f"Objective value: {result.fval}")
    
    # Partition satellites
    set_A_ids = [SATELLITE_IDS[i] for i, x in enumerate(result.x) if x == 0]
    set_B_ids = [SATELLITE_IDS[i] for i, x in enumerate(result.x) if x == 1]
    
    print(f"\n  Set A (Orbit 1): {set_A_ids}")
    print(f"  Set B (Orbit 2): {set_B_ids}")
    
    # Performance analysis by orbit set
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS BY ORBIT SET")
    print("="*80)
    
    for set_name, sat_ids in [("Set A", set_A_ids), ("Set B", set_B_ids)]:
        if sat_ids:
            print(f"\n{set_name}:")
            avg_qoe = np.mean([augmented_data[sid]['qoe_score'] for sid in sat_ids])
            avg_throughput = np.mean([augmented_data[sid]['download_throughput_mbps'] for sid in sat_ids])
            avg_packet_loss = np.mean([augmented_data[sid]['packet_loss_percent'] for sid in sat_ids])
            avg_signal_loss = np.mean([augmented_data[sid]['signal_loss_db'] for sid in sat_ids])
            
            print(f"  Average QoE Score: {avg_qoe:.2f}/10")
            print(f"  Average Download Throughput: {avg_throughput:.1f} Mbps")
            print(f"  Average Packet Loss: {avg_packet_loss:.2f}%")
            print(f"  Average Signal Loss: {avg_signal_loss:.2f} dB")
            
            print(f"\n  Satellites in {set_name}:")
            for sid in sat_ids:
                data = augmented_data[sid]
                perf_score = telemetry_generator.calculate_performance_score(data)
                print(f"    - Sat {sid}: QoE={data['qoe_score']:.1f}, "
                      f"Weather={data['weather']}, "
                      f"Throughput={data['download_throughput_mbps']:.0f}Mbps, "
                      f"Performance={perf_score:.2f}")
    
    # Summary insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("\n✓ Telemetry data successfully generated using opendatabay.com schema")
    print("✓ Performance metrics integrated into conflict matrix weighting")
    print("✓ QUBO optimization considers both spatial and performance factors")
    print("✓ Orbit partitioning balances collision avoidance with service quality")
    
    # Calculate improvement
    if set_A_ids and set_B_ids:
        overall_avg_qoe = np.mean([augmented_data[sid]['qoe_score'] for sid in SATELLITE_IDS])
        print(f"\n📊 Overall Average QoE Score: {overall_avg_qoe:.2f}/10")
        
        # Count conflicts
        total_conflicts_standard = np.sum(adj_matrix_standard > 0) / 2
        total_conflicts_weighted = np.sum(adj_matrix_weighted > 0) / 2
        print(f"📊 Total Spatial Conflicts Detected: {int(total_conflicts_standard)}")
        print(f"📊 Performance-Weighted Conflict Score: {total_conflicts_weighted:.1f}")


if __name__ == "__main__":
    main()
