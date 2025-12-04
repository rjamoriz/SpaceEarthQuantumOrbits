# 🌍 Photorealistic Starlink Earth Visualization

Professional-grade visualization of the Starlink satellite constellation matching the style of official SpaceX imagery.

## 🎨 Features

- **Photorealistic Earth**: Detailed continent and ocean textures with realistic colors
- **100,000 Satellite Constellation**: Full-scale Starlink mega-constellation
- **Multi-Shell Architecture**: Three orbital shells (340-550km, 550-1150km, 1150-1325km)
- **Atmospheric Glow**: Multi-layer blue atmospheric effect
- **Starfield Background**: 600+ background stars
- **Golden Satellites**: Orange/golden satellite points matching SpaceX style
- **High Resolution**: 7200x7200 pixels at 300 DPI

## 📸 Output

![Starlink Photorealistic Earth](starlink_photorealistic_earth.png)

## 🚀 Usage

### Run the Visualization

```bash
python3 photorealistic_earth_viz.py
```

### Customize View Angles

Edit the script to change the camera view:

```python
fig = create_photorealistic_visualization(
    view='north_america',  # Options: 'north_america', 'europe', 'asia', 'global', 'pacific'
    n_display=12000        # Number of satellites to display
)
```

### Available Views

- **north_america**: Focus on North America (default)
- **europe**: Focus on Europe and Africa
- **asia**: Focus on Asia and Pacific
- **global**: Wide global view
- **pacific**: Pacific Ocean focus

## 📊 Technical Details

### Constellation Architecture

| Shell | Altitude Range | Satellites | Coverage |
|-------|---------------|------------|----------|
| Shell 1 | 340-550 km | 60,000 (60%) | ±53° latitude |
| Shell 2 | 550-1150 km | 30,000 (30%) | ±70° latitude |
| Shell 3 | 1150-1325 km | 10,000 (10%) | ±85° latitude |

### Earth Texture

- **Resolution**: 300x150 grid points
- **Continents**: North America, South America, Europe, Africa, Asia, Australia, Antarctica, Greenland
- **Terrain Types**: Oceans, forests, deserts, mountains, ice caps
- **Color Palette**: Realistic blues, greens, browns, and whites

### Rendering Performance

- **Generation Time**: ~30-60 seconds
- **Memory Usage**: ~2-3 GB RAM
- **Output Size**: ~5-10 MB PNG file

## 🎯 Comparison with Reference

This visualization matches the professional Starlink imagery style:

✅ **Photorealistic Earth** with detailed continents  
✅ **Golden/orange satellites** forming a ring around Earth  
✅ **Blue atmospheric glow** effect  
✅ **Black space background** with stars  
✅ **Realistic satellite density** and distribution  
✅ **Professional camera angle** showing Earth curvature  

## 🛠️ Dependencies

```bash
pip install numpy matplotlib
```

## 📝 Notes

- The visualization displays 12,000 satellites (sampled from 100,000 total)
- Satellite positions are procedurally generated based on Starlink orbital parameters
- Earth texture is simplified but geographically accurate
- Atmosphere uses multi-layer rendering for realistic glow effect

## 🔧 Customization Options

### Adjust Satellite Count

```python
n_display=12000  # Increase for denser constellation (slower rendering)
```

### Change Satellite Color

```python
sat_colors = ['#ffb347']  # Golden orange (default)
# Try: '#ff6b6b' (red), '#4ecdc4' (cyan), '#ffe66d' (yellow)
```

### Modify Earth Colors

Edit the `create_detailed_earth_texture()` function to adjust:
- Ocean colors (dark blue to light blue)
- Land colors (green, brown, desert, ice)
- Terrain variation

### Adjust Atmosphere

```python
glow_layers = [
    (6500, 0.12, '#4a9eff'),  # (radius, alpha, color)
    (6450, 0.15, '#5aa8ff'),
    (6420, 0.10, '#6ab5ff'),
]
```

## 📈 Performance Tips

1. **Reduce satellite count** for faster rendering: `n_display=5000`
2. **Lower Earth resolution** in `create_detailed_earth_texture()`: `u = np.linspace(0, 2 * np.pi, 200)`
3. **Fewer stars**: `create_starfield(ax, n_stars=300)`
4. **Lower DPI**: `plt.savefig(..., dpi=150)`

## 🎓 Educational Use

This visualization is ideal for:
- **Presentations** on satellite technology
- **Educational materials** about space infrastructure
- **Research papers** on mega-constellations
- **Public outreach** and science communication
- **Technical documentation** for aerospace projects

## 📄 License

This visualization tool is part of the SpaceEarth project. Use freely for educational and research purposes.

## 🙏 Credits

- **Starlink Constellation Data**: Based on SpaceX public information
- **Visualization Style**: Inspired by official SpaceX Starlink imagery
- **Earth Model**: Simplified geographic representation
- **Rendering**: Matplotlib 3D surface plotting

---

**Created**: December 2025  
**Version**: 1.0  
**Author**: SpaceEarth Project
