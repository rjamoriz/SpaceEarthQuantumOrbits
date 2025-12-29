# 🔧 Visualization Cell - Missing Functions Fix

## Quick Fix

Add these **two helper functions** at the top of your visualization cell:

```python
import math

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
            f"Download: {d['download_throughput_mbps']:.1f} Mbps"
        )
        hovers.append(hover)
    
    return x, y, z, hovers, qoe_scores
```

Then your visualization code will work!
