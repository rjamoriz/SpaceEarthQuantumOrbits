#!/usr/bin/env python3
"""
Photorealistic Starlink Earth Visualization
Creates high-quality Earth visualization matching professional Starlink imagery.
Inspired by the official Starlink constellation visualization style.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# Set dark theme
plt.style.use('dark_background')
matplotlib.rcParams['figure.facecolor'] = '#000000'
matplotlib.rcParams['axes.facecolor'] = '#000000'

print("=" * 80)
print("🌍 PHOTOREALISTIC STARLINK EARTH VISUALIZATION")
print("=" * 80)

def to_cartesian(lat, lon, alt, earth_radius=6371):
    """Convert geodetic coordinates to Cartesian."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    r = earth_radius + alt
    
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    
    return x, y, z

def create_detailed_earth_texture():
    """Create a detailed textured Earth with realistic continents and oceans."""
    
    print("\n🎨 Generating detailed Earth texture...")
    
    # High resolution grid
    u = np.linspace(0, 2 * np.pi, 300)
    v = np.linspace(0, np.pi, 150)
    
    earth_radius = 6371
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create realistic color map
    colors_map = np.zeros((len(u), len(v), 3))
    
    for i, longitude in enumerate(u):
        for j, latitude in enumerate(v):
            lat_deg = np.degrees(latitude) - 90
            lon_deg = np.degrees(longitude) - 180
            
            # Deep ocean color (dark blue)
            ocean_color = np.array([0.02, 0.08, 0.20])
            
            # Shallow ocean (lighter blue)
            shallow_ocean = np.array([0.05, 0.15, 0.35])
            
            # Default to ocean
            color = ocean_color
            is_land = False
            
            # NORTH AMERICA
            if -170 < lon_deg < -50 and 15 < lat_deg < 75:
                # USA and Canada
                if -130 < lon_deg < -65 and 25 < lat_deg < 50:
                    color = np.array([0.25, 0.35, 0.15])  # Green/brown
                    is_land = True
                # Western mountains
                elif -125 < lon_deg < -105 and 30 < lat_deg < 50:
                    color = np.array([0.35, 0.30, 0.20])  # Brown
                    is_land = True
                # Canada
                elif -140 < lon_deg < -60 and 45 < lat_deg < 70:
                    color = np.array([0.20, 0.35, 0.15])  # Green
                    is_land = True
                # Mexico
                elif -115 < lon_deg < -85 and 15 < lat_deg < 32:
                    color = np.array([0.35, 0.30, 0.15])  # Desert brown
                    is_land = True
                # Central America
                elif -95 < lon_deg < -75 and 7 < lat_deg < 20:
                    color = np.array([0.20, 0.40, 0.15])  # Tropical green
                    is_land = True
            
            # SOUTH AMERICA
            elif -85 < lon_deg < -35 and -55 < lat_deg < 15:
                # Amazon rainforest
                if -75 < lon_deg < -50 and -10 < lat_deg < 5:
                    color = np.array([0.15, 0.45, 0.20])  # Deep green
                    is_land = True
                # Andes mountains
                elif -80 < lon_deg < -65 and -40 < lat_deg < 10:
                    color = np.array([0.40, 0.35, 0.25])  # Mountain brown
                    is_land = True
                # Rest of South America
                else:
                    color = np.array([0.20, 0.40, 0.15])
                    is_land = True
            
            # EUROPE
            elif -10 < lon_deg < 40 and 35 < lat_deg < 71:
                color = np.array([0.25, 0.38, 0.18])  # Temperate green
                is_land = True
            
            # AFRICA
            elif -20 < lon_deg < 52 and -35 < lat_deg < 37:
                # Sahara Desert
                if -15 < lon_deg < 40 and 15 < lat_deg < 30:
                    color = np.array([0.55, 0.45, 0.25])  # Sandy
                    is_land = True
                # Sub-Saharan Africa
                elif -15 < lon_deg < 45 and -10 < lat_deg < 15:
                    color = np.array([0.25, 0.40, 0.15])  # Green
                    is_land = True
                # Southern Africa
                elif 10 < lon_deg < 35 and -35 < lat_deg < -10:
                    color = np.array([0.35, 0.35, 0.20])  # Savanna
                    is_land = True
                else:
                    color = np.array([0.30, 0.35, 0.18])
                    is_land = True
            
            # ASIA
            elif 40 < lon_deg < 180 and -10 < lat_deg < 75:
                # Middle East deserts
                if 35 < lon_deg < 65 and 15 < lat_deg < 40:
                    color = np.array([0.50, 0.40, 0.25])  # Desert
                    is_land = True
                # Siberia
                elif 60 < lon_deg < 180 and 50 < lat_deg < 75:
                    color = np.array([0.20, 0.30, 0.15])  # Tundra
                    is_land = True
                # Southeast Asia
                elif 95 < lon_deg < 140 and -10 < lat_deg < 25:
                    color = np.array([0.20, 0.45, 0.20])  # Tropical
                    is_land = True
                # Rest of Asia
                else:
                    color = np.array([0.25, 0.35, 0.18])
                    is_land = True
            
            # AUSTRALIA
            elif 110 < lon_deg < 155 and -45 < lat_deg < -10:
                # Outback
                if 115 < lon_deg < 145 and -30 < lat_deg < -15:
                    color = np.array([0.50, 0.35, 0.20])  # Red desert
                    is_land = True
                # Coastal areas
                else:
                    color = np.array([0.30, 0.38, 0.18])
                    is_land = True
            
            # ANTARCTICA
            elif lat_deg < -60:
                color = np.array([0.92, 0.94, 0.98])  # Ice white
                is_land = True
            
            # GREENLAND
            elif -75 < lon_deg < -15 and 60 < lat_deg < 85:
                color = np.array([0.88, 0.92, 0.96])  # Ice
                is_land = True
            
            # Add natural variation
            if is_land:
                variation = np.random.uniform(0.90, 1.10)
                color = color * variation
            else:
                # Ocean depth variation
                variation = np.random.uniform(0.85, 1.05)
                color = ocean_color * variation
            
            colors_map[i, j] = np.clip(color, 0, 1)
    
    print("   ✓ Earth texture complete (300x150 resolution)")
    return x, y, z, colors_map

def add_atmosphere_glow(ax, earth_radius=6371):
    """Add realistic atmospheric glow."""
    
    print("   💫 Adding atmospheric glow...")
    
    # Multiple layers for realistic atmosphere
    glow_layers = [
        (6500, 0.12, '#4a9eff'),  # Outer atmosphere
        (6450, 0.15, '#5aa8ff'),  # Mid atmosphere
        (6420, 0.10, '#6ab5ff'),  # Inner atmosphere
    ]
    
    for glow_radius, alpha, color in glow_layers:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        
        x_glow = glow_radius * np.outer(np.cos(u), np.sin(v))
        y_glow = glow_radius * np.outer(np.sin(u), np.sin(v))
        z_glow = glow_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_glow, y_glow, z_glow, 
                       color=color, alpha=alpha, shade=False)

def create_starfield(ax, n_stars=800):
    """Add realistic background stars."""
    
    print("   ⭐ Adding starfield...")
    
    star_distance = 18000
    
    # Generate star positions
    star_x = np.random.uniform(-star_distance, star_distance, n_stars)
    star_y = np.random.uniform(-star_distance, star_distance, n_stars)
    star_z = np.random.uniform(-star_distance, star_distance, n_stars)
    
    # Use uniform size and alpha to avoid array mismatch
    ax.scatter(star_x, star_y, star_z, 
              c='white', s=0.5, alpha=0.6)

def generate_starlink_constellation(n_satellites=100000):
    """Generate realistic Starlink constellation."""
    
    print(f"\n🛰️  Generating {n_satellites:,} satellite constellation...")
    
    # Shell 1: 340-550 km (60% of satellites) - Main constellation
    n_shell1 = int(n_satellites * 0.60)
    alt_shell1 = np.random.uniform(340, 550, n_shell1)
    lat_shell1 = np.random.uniform(-53, 53, n_shell1)
    lon_shell1 = np.random.uniform(-180, 180, n_shell1)
    
    # Shell 2: 550-1150 km (30% of satellites) - Extended coverage
    n_shell2 = int(n_satellites * 0.30)
    alt_shell2 = np.random.uniform(550, 1150, n_shell2)
    lat_shell2 = np.random.uniform(-70, 70, n_shell2)
    lon_shell2 = np.random.uniform(-180, 180, n_shell2)
    
    # Shell 3: 1150-1325 km (10% of satellites) - Polar coverage
    n_shell3 = n_satellites - n_shell1 - n_shell2
    alt_shell3 = np.random.uniform(1150, 1325, n_shell3)
    lat_shell3 = np.random.uniform(-85, 85, n_shell3)
    lon_shell3 = np.random.uniform(-180, 180, n_shell3)
    
    # Combine all shells
    altitudes = np.concatenate([alt_shell1, alt_shell2, alt_shell3])
    latitudes = np.concatenate([lat_shell1, lat_shell2, lat_shell3])
    longitudes = np.concatenate([lon_shell1, lon_shell2, lon_shell3])
    
    print(f"   ✓ Shell 1 (340-550 km): {n_shell1:,} satellites")
    print(f"   ✓ Shell 2 (550-1150 km): {n_shell2:,} satellites")
    print(f"   ✓ Shell 3 (1150-1325 km): {n_shell3:,} satellites")
    
    return altitudes, latitudes, longitudes

def create_photorealistic_visualization(view='north_america', n_display=12000):
    """
    Create photorealistic Earth with Starlink constellation.
    
    Parameters:
    -----------
    view : str
        Camera view angle: 'north_america', 'europe', 'asia', 'global'
    n_display : int
        Number of satellites to display (sampled from full constellation)
    """
    
    print(f"\n📸 Creating photorealistic view: {view.upper()}")
    print(f"   Display satellites: {n_display:,}")
    
    # Create figure
    fig = plt.figure(figsize=(24, 24), facecolor='#000000')
    ax = fig.add_subplot(111, projection='3d', facecolor='#000000')
    
    # Add starfield
    create_starfield(ax, n_stars=600)
    
    # Create Earth
    x_earth, y_earth, z_earth, earth_colors = create_detailed_earth_texture()
    
    print("\n🌍 Rendering Earth surface...")
    ax.plot_surface(x_earth, y_earth, z_earth, 
                   facecolors=earth_colors,
                   shade=True,
                   lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=50),
                   antialiased=True,
                   rcount=300, ccount=150)
    
    # Add atmosphere
    add_atmosphere_glow(ax)
    
    # Generate constellation
    altitudes, latitudes, longitudes = generate_starlink_constellation(100000)
    
    # Sample satellites for display
    sample_indices = np.random.choice(len(altitudes), n_display, replace=False)
    
    print(f"\n🛰️  Positioning {n_display:,} satellites...")
    sat_x, sat_y, sat_z = [], [], []
    
    for idx in sample_indices:
        x, y, z = to_cartesian(latitudes[idx], longitudes[idx], altitudes[idx])
        sat_x.append(x)
        sat_y.append(y)
        sat_z.append(z)
    
    # Plot satellites with golden/orange color (like reference image)
    # Use single color array to avoid alpha mismatch
    sat_colors = ['#ffb347'] * len(sat_x)
    ax.scatter(sat_x, sat_y, sat_z,
              c=sat_colors,
              s=2.5,
              alpha=0.85)
    
    # Set camera view
    views = {
        'north_america': (20, -100),
        'europe': (25, 10),
        'asia': (25, 85),
        'global': (15, -60),
        'pacific': (20, -160)
    }
    
    elev, azim = views.get(view, (20, -100))
    ax.view_init(elev=elev, azim=azim)
    
    # Styling
    limit = 9000
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    # Hide axes for clean look
    ax.set_axis_off()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    
    plt.tight_layout(pad=0)
    
    return fig

# Main execution
if __name__ == "__main__":
    print("\n🎬 Starting visualization rendering...")
    print("   This may take 30-60 seconds for high quality...\n")
    
    # Create visualization
    fig = create_photorealistic_visualization(
        view='north_america',
        n_display=12000
    )
    
    # Save high-resolution image
    output_file = 'starlink_photorealistic_earth.png'
    print(f"\n💾 Saving high-resolution image...")
    plt.savefig(output_file, 
                dpi=300, 
                facecolor='#000000', 
                bbox_inches='tight',
                pad_inches=0.1)
    
    print(f"   ✓ Saved: {output_file}")
    print(f"   Resolution: ~7200x7200 pixels (300 DPI)")
    
    # Display
    print("\n📺 Displaying visualization...")
    plt.show()
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   • Total constellation: 100,000 satellites")
    print(f"   • Displayed: 12,000 satellites")
    print(f"   • Earth texture: 300x150 resolution")
    print(f"   • Atmosphere: Multi-layer glow effect")
    print(f"   • Background: 600 stars")
    print(f"   • Output: {output_file}")
    print("\n🎨 Style: Professional Starlink visualization")
    print("=" * 80)
