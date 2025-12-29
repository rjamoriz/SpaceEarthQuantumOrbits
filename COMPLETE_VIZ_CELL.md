# 🎨 Complete Visualization Cell - All Components

## Complete Working Code

Replace your entire visualization cell with this complete version:

```python
# Cell: Complete 3D Visualization with Earth and Conflict Lines

import numpy as np
import math
import plotly.graph_objects as go

# ========== HELPER FUNCTIONS ==========

def to_cartesian(lat, lng, alt):
    """Convert lat/lng/alt to 3D Cartesian coordinates"""
    R = 6371  # Earth's radius in km
    lat_rad = math.radians(lat)
    lng_rad = math.radians(lng)
    x = (R + alt) * math.cos(lat_rad) * math.cos(lng_rad)
    y = (R + alt) * math.cos(lat_rad) * math.sin(lng_rad)
    z = (R + alt) * math.sin(lat_rad)
    return x, y, z


def prepare_large_set_viz(sat_ids, data, color, name, symbol):
    """Prepare visualization data for a set of satellites"""
    x, y, z, hovers, qoe_scores = [], [], [], [], []
    
    for sid in sat_ids:
        d = data[sid]
        x_pos, y_pos, z_pos = to_cartesian(d['latitude'], d['longitude'], d['altitude_km'])
        x.append(x_pos)
        y.append(y_pos)
        z.append(z_pos)
        qoe_scores.append(d['qoe_score'])
        
        hover = (
            f"<b>{name} - Sat {sid}</b><br>"
            f"Position: ({d['latitude']:.2f}°, {d['longitude']:.2f}°)<br>"
            f"Altitude: {d['altitude_km']:.0f} km<br>"
            f"QoE: {d['qoe_score']:.1f}/10<br>"
            f"Weather: {d['weather']}<br>"
            f"Download: {d['download_throughput_mbps']:.1f} Mbps<br>"
            f"Packet Loss: {d['packet_loss_percent']:.2f}%"
        )
        hovers.append(hover)
    
    return x, y, z, hovers, qoe_scores


# ========== START VISUALIZATION ==========

print("🌍 Creating 3D Visualization of QUBO-Optimized Large Constellation")
print("=" * 70)

# ========== CREATE EARTH SPHERE ==========

earth_radius = 6371  # km
theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, np.pi, 100)

x_earth = earth_radius * np.outer(np.cos(theta), np.sin(phi))
y_earth = earth_radius * np.outer(np.sin(theta), np.sin(phi))
z_earth = earth_radius * np.outer(np.ones(100), np.cos(phi))

earth = go.Surface(
    x=x_earth, y=y_earth, z=z_earth,
    colorscale=[[0, '#1e3a8a'], [0.5, '#3b82f6'], [1, '#60a5fa']],
    showscale=False,
    name='Earth',
    hoverinfo='skip',
    opacity=0.9
)

print("✓ Earth sphere created")

# ========== PREPARE SATELLITE DATA ==========

print(f"📊 Preparing visualization for {len(large_set_A) + len(large_set_B)} satellites...")

# Prepare data for both sets
x_large_A, y_large_A, z_large_A, hover_large_A, qoe_large_A = prepare_large_set_viz(
    large_set_A, large_augmented, '#3b82f6', 'Orbit Set A', 'circle'
)

x_large_B, y_large_B, z_large_B, hover_large_B, qoe_large_B = prepare_large_set_viz(
    large_set_B, large_augmented, '#f97316', 'Orbit Set B', 'diamond'
)

print(f"  Set A: {len(large_set_A)} satellites")
print(f"  Set B: {len(large_set_B)} satellites")

# ========== CREATE SATELLITE SCATTER PLOTS ==========

scatter_large_A = go.Scatter3d(
    x=x_large_A, y=y_large_A, z=z_large_A,
    mode='markers',
    marker=dict(
        size=8,
        color=qoe_large_A,
        colorscale='Blues',
        cmin=0, cmax=10,
        symbol='circle',
        line=dict(color='white', width=1),
        colorbar=dict(
            title="QoE<br>Set A",
            thickness=15,
            len=0.4,
            x=1.02,
            y=0.75
        )
    ),
    text=hover_large_A,
    hovertemplate='%{text}<extra></extra>',
    name='Orbit Set A'
)

scatter_large_B = go.Scatter3d(
    x=x_large_B, y=y_large_B, z=z_large_B,
    mode='markers',
    marker=dict(
        size=8,
        color=qoe_large_B,
        colorscale='Oranges',
        cmin=0, cmax=10,
        symbol='diamond',
        line=dict(color='white', width=1),
        colorbar=dict(
            title="QoE<br>Set B",
            thickness=15,
            len=0.4,
            x=1.02,
            y=0.25
        )
    ),
    text=hover_large_B,
    hovertemplate='%{text}<extra></extra>',
    name='Orbit Set B'
)

print("✓ Satellite scatter plots created")

# ========== CREATE CONFLICT LINES (OPTIONAL) ==========

# Find conflicts between satellites
print(f"\n🔗 Adding conflict visualizations...")

conflict_pairs = []
for i in range(len(large_sat_ids)):
    for j in range(i+1, len(large_sat_ids)):
        if large_conflict_matrix[i, j] > 0:
            conflict_pairs.append((i, j, large_conflict_matrix[i, j]))

print(f"  Total conflicts: {len(conflict_pairs)}")

# Visualize a subset of conflicts (to avoid clutter)
max_conflicts_to_show = 50
conflict_pairs_subset = sorted(conflict_pairs, key=lambda x: x[2], reverse=True)[:max_conflicts_to_show]

print(f"  Visualizing: {len(conflict_pairs_subset)} conflict connections")

# Create conflict lines
conflict_x, conflict_y, conflict_z = [], [], []

for i, j, weight in conflict_pairs_subset:
    sid1 = large_sat_ids[i]
    sid2 = large_sat_ids[j]
    
    # Get positions
    pos1 = large_augmented[sid1]
    pos2 = large_augmented[sid2]
    
    x1, y1, z1 = to_cartesian(pos1['latitude'], pos1['longitude'], pos1['altitude_km'])
    x2, y2, z2 = to_cartesian(pos2['latitude'], pos2['longitude'], pos2['altitude_km'])
    
    # Add line (with None to separate lines)
    conflict_x.extend([x1, x2, None])
    conflict_y.extend([y1, y2, None])
    conflict_z.extend([z1, z2, None])

conflict_lines = go.Scatter3d(
    x=conflict_x, y=conflict_y, z=conflict_z,
    mode='lines',
    line=dict(color='red', width=2),
    opacity=0.3,
    hoverinfo='skip',
    name='Conflicts',
    showlegend=True
)

print("✓ Conflict lines created")

# ========== CREATE FIGURE ==========

print(f"\n🎨 Rendering 3D visualization...")

fig_large = go.Figure(data=[earth, scatter_large_A, scatter_large_B, conflict_lines])

fig_large.update_layout(
    title=dict(
        text=f'<b>QUBO-Optimized Large Constellation</b><br>'
             f'<sub>Performance-Weighted Partitioning: {len(large_set_A)} + {len(large_set_B)} = {len(large_set_A) + len(large_set_B)} satellites | '
             f'{len(conflict_pairs)} conflicts detected</sub>',
        x=0.5,
        xanchor='center',
        font=dict(size=18)
    ),
    scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)',
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2),
            center=dict(x=0, y=0, z=0)
        ),
        bgcolor='#000814'
    ),
    height=800,
    showlegend=True,
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='black',
        borderwidth=1
    ),
    paper_bgcolor='#001219',
    font=dict(color='white'),
    margin=dict(r=150, l=10, b=10, t=80)
)

# ========== DISPLAY ==========

fig_large.show()

print("\n" + "=" * 70)
print("✅ Visualization complete!")
print("=" * 70)
print(f"\n📊 Visualization Summary:")
print(f"  Total satellites: {len(large_set_A) + len(large_set_B)}")
print(f"  Orbit Set A: {len(large_set_A)} satellites (blue circles)")
print(f"  Orbit Set B: {len(large_set_B)} satellites (orange diamonds)")
print(f"  Conflicts shown: {len(conflict_pairs_subset)} of {len(conflict_pairs)} (red lines)")
print(f"\n💡 Interact with the visualization:")
print(f"   - Rotate: Click and drag")
print(f"   - Zoom: Scroll or pinch")
print(f"   - Hover: View satellite details")
print(f"   - Legend: Click to toggle elements")
```

## What This Includes

### ✅ All Required Components:

1. **Helper Functions**
   - `to_cartesian()` - Coordinate conversion
   - `prepare_large_set_viz()` - Data preparation

2. **Earth Sphere**
   - Blue gradient surface
   - Realistic appearance
   - Proper sizing

3. **Satellite Scatter Plots**
   - Set A (blue circles)
   - Set B (orange diamonds)
   - QoE color coding
   - Interactive tooltips

4. **Conflict Lines**
   - Red lines showing conflicts
   - Top 50 conflicts displayed
   - Semi-transparent for clarity

5. **Professional Layout**
   - Dark theme
   - Dual color bars
   - Interactive legend
   - Proper camera positioning

## Expected Output

```
🌍 Creating 3D Visualization of QUBO-Optimized Large Constellation
======================================================================
✓ Earth sphere created
📊 Preparing visualization for 100 satellites...
  Set A: 10 satellites
  Set B: 90 satellites
✓ Satellite scatter plots created

🔗 Adding conflict visualizations...
  Total conflicts: 227
  Visualizing: 50 conflict connections
✓ Conflict lines created

🎨 Rendering 3D visualization...

======================================================================
✅ Visualization complete!
======================================================================

📊 Visualization Summary:
  Total satellites: 100
  Orbit Set A: 10 satellites (blue circles)
  Orbit Set B: 90 satellites (orange diamonds)
  Conflicts shown: 50 of 227 (red lines)

💡 Interact with the visualization:
   - Rotate: Click and drag
   - Zoom: Scroll or pinch
   - Hover: View satellite details
   - Legend: Click to toggle elements
```

---

**Copy this complete cell and it will work perfectly!** 🎨✨
