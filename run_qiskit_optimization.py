from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
import numpy as np
import requests
import math
from starlink_telemetry import StarlinkTelemetryGenerator, print_telemetry_summary

# --- 1. Fetch real-time satellite data ---
def get_satellite_positions(api_key, satellite_ids):
    """
    Fetches the current latitude, longitude, and altitude of a list of satellites.
    """
    positions = {}
    base_url = "https://api.n2yo.com/rest/v1/satellite/positions"
    for sat_id in satellite_ids:
        # For this example, we'll use a dummy observer location.
        # A real application might use a specific ground station.
        url = f"{base_url}/{sat_id}/34.0522/-118.2437/0/1/&apiKey={api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            if 'positions' in data and data['positions']:
                positions[sat_id] = {
                    'lat': data['positions'][0]['satlatitude'],
                    'lng': data['positions'][0]['satlongitude'],
                    'alt': data['positions'][0]['sataltitude']
                }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for satellite {sat_id}: {e}")
    return positions

def calculate_distance(pos1, pos2):
    """
    Calculates the straight-line distance between two points in 3D space.
    This is a simplification; for real orbital mechanics, you'd use more complex models.
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

def create_conflict_matrix(positions, threshold_km, augmented_data=None, use_performance=True):
    """
    Creates an adjacency matrix representing conflicts between satellites based on distance
    and optionally performance metrics.
    
    Args:
        positions: Dictionary of satellite positions
        threshold_km: Distance threshold for conflicts
        augmented_data: Optional telemetry data for performance-based weighting
        use_performance: If True, weight conflicts by performance degradation
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
                
                # If telemetry data available, weight by performance impact
                if use_performance and augmented_data:
                    # Higher conflict weight if either satellite has poor performance
                    # (we want to separate poorly performing satellites more)
                    perf1 = augmented_data[sat_id1].get('qoe_score', 5.0) / 10.0
                    perf2 = augmented_data[sat_id2].get('qoe_score', 5.0) / 10.0
                    
                    # Inverse performance: lower performance = higher conflict weight
                    avg_perf = (perf1 + perf2) / 2.0
                    conflict_weight = 1.0 + (1.0 - avg_perf) * 2.0  # Range: 1.0 to 3.0
                
                adj_matrix[i, j] = conflict_weight
                adj_matrix[j, i] = conflict_weight  # The matrix is symmetric
    
    return adj_matrix

# --- 2. Define the problem from real data ---
# IMPORTANT: You need a free API key from https://www.n2yo.com/
API_KEY = "YOUR_API_KEY_HERE"  # <-- REPLACE WITH YOUR ACTUAL API KEY

# A small selection of satellites (NORAD IDs). 
# You can find more at https://celestrak.com/
SATELLITE_IDS = [
    25544,  # INTERNATIONAL SPACE STATION
    20580,  # HUBBLE SPACE TELESCOPE
    24876,  # NOAA 15
    25338,  # NOAA 16
]

print("Fetching satellite data...")
satellite_positions = get_satellite_positions(API_KEY, SATELLITE_IDS)

# Initialize telemetry generator
telemetry_generator = StarlinkTelemetryGenerator(seed=42)

if not satellite_positions or len(satellite_positions) < len(SATELLITE_IDS):
    print("\nCould not fetch all satellite positions. Using fallback data.")
    print("This might be due to an invalid API key or network issues.")
    print("Get a free key from https://www.n2yo.com/ and place it in the API_KEY variable.")
    # Fallback to the original hardcoded problem if the API fails
    num_slots = 4
    adj_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])
else:
    print("Satellite data fetched successfully!")
    
    # Augment satellite data with Starlink telemetry metrics
    print("\nGenerating Starlink telemetry data...")
    augmented_data = telemetry_generator.augment_satellite_positions(satellite_positions)
    print_telemetry_summary(augmented_data)
    
    # Define a conflict if satellites are within this distance of each other
    CONFLICT_THRESHOLD_KM = 1000
    
    # Create conflict matrix with performance-based weighting
    print("\nCreating performance-weighted conflict matrix...")
    adj_matrix = create_conflict_matrix(
        satellite_positions, 
        CONFLICT_THRESHOLD_KM,
        augmented_data=augmented_data,
        use_performance=True
    )

print("\nAdjacency Matrix (Conflict Graph):")
print(adj_matrix)

# This is a Max-Cut problem, which is equivalent to a QUBO.
# We want to partition the graph to maximize the number of edges between the two sets.
max_cut = Maxcut(adj_matrix)
qp = max_cut.to_quadratic_program()

# --- 3. Convert to QUBO ---
qubo_converter = QuadraticProgramToQubo()
qubo = qubo_converter.convert(qp)

# --- 4. Solve the QUBO ---
# We use a classical solver here for simplicity.
# For a real quantum approach, you would use a quantum algorithm like QAOA or VQE.
numpy_solver = NumPyMinimumEigensolver()
optimizer = MinimumEigenOptimizer(numpy_solver)
result = optimizer.solve(qubo)

# --- 5. Interpret the results ---
print(f"\nOptimal solution: {result.x}")
print(f"This means we can partition the satellites into two sets to minimize conflicts:")

# Map the result back to the satellite IDs
set_A_ids = [SATELLITE_IDS[i] for i, x in enumerate(result.x) if x == 0]
set_B_ids = [SATELLITE_IDS[i] for i, x in enumerate(result.x) if x == 1]

print(f"  Set A (e.g., Orbit 1): {set_A_ids}")
print(f"  Set B (e.g., Orbit 2): {set_B_ids}")
print("\nThis partitioning minimizes conflicts weighted by performance degradation.")

# If we have augmented data, show performance analysis
if 'augmented_data' in locals():
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS BY ORBIT SET")
    print("="*80)
    
    for set_name, sat_ids in [("Set A", set_A_ids), ("Set B", set_B_ids)]:
        if sat_ids:
            print(f"\n{set_name}:")
            avg_qoe = np.mean([augmented_data[sid]['qoe_score'] for sid in sat_ids])
            avg_throughput = np.mean([augmented_data[sid]['download_throughput_mbps'] for sid in sat_ids])
            avg_packet_loss = np.mean([augmented_data[sid]['packet_loss_percent'] for sid in sat_ids])
            
            print(f"  Average QoE Score: {avg_qoe:.2f}/10")
            print(f"  Average Download Throughput: {avg_throughput:.1f} Mbps")
            print(f"  Average Packet Loss: {avg_packet_loss:.2f}%")
            
            # List satellites with their performance
            for sid in sat_ids:
                data = augmented_data[sid]
                print(f"    - Satellite {sid}: QoE={data['qoe_score']}, Weather={data['weather']}")
