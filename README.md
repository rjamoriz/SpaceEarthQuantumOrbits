# 🛰️ SpaceEarth Quantum Orbits

<div align="center">

![Starlink Photorealistic Earth](starlink_photorealistic_earth.png)

**Advanced Satellite Constellation Optimization using Quantum-Inspired Algorithms**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-purple.svg)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

[Quick Start](#-quick-start) • [Features](#-features) • [Documentation](#-documentation) • [Examples](#-examples) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

**SpaceEarth Quantum Orbits** is a cutting-edge project that combines **quantum-inspired optimization algorithms** with **real-time satellite telemetry** to solve complex orbital partitioning problems. The system leverages Qiskit's quantum optimization framework to minimize conflicts in satellite constellations while maximizing network performance.

### 🎯 What Does This Project Do?

- **Fetches real-time satellite positions** from N2YO API or uses simulated data
- **Generates realistic telemetry metrics** based on Starlink performance patterns
- **Optimizes satellite orbit partitioning** using QUBO (Quadratic Unconstrained Binary Optimization)
- **Visualizes 100,000+ satellite constellations** with photorealistic Earth rendering
- **Analyzes performance metrics** including QoE scores, throughput, and packet loss
- **Models weather-dependent performance** degradation for realistic simulations

---

## ✨ Features

### 🔬 Quantum Optimization
- **QUBO Formulation**: Converts satellite conflict problems into quantum-solvable format
- **Max-Cut Algorithm**: Optimal graph partitioning using Qiskit
- **Performance Weighting**: Integrates QoE scores into optimization objective
- **Classical & Quantum Solvers**: Support for both NumPy and quantum backends

### 📡 Satellite Telemetry
- **13-Column Schema**: Comprehensive metrics based on real Starlink datasets
- **Weather Modeling**: Realistic performance degradation (clear, rain, snow, etc.)
- **Altitude Adjustments**: Signal quality varies with orbital height
- **QoE Scoring**: 0-10 scale quality assessment based on multiple factors

### 🌍 Photorealistic Visualization
- **100,000 Satellite Constellation**: Full-scale mega-constellation rendering
- **Multi-Shell Architecture**: Three orbital shells (340-1325 km)
- **Atmospheric Effects**: Multi-layer blue glow and starfield background
- **High Resolution**: 7200x7200 pixels at 300 DPI
- **Multiple Views**: North America, Europe, Asia, Global, Pacific

### 📊 Performance Analytics
- **Real-Time Metrics**: Download/upload speeds, packet loss, signal quality
- **Orbit Set Comparison**: Performance analysis per partition
- **Conflict Matrix**: Distance and performance-weighted conflict detection
- **Statistical Insights**: Average QoE, throughput, and reliability metrics

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python3 --version

# Virtual environment (recommended)
python3 -m venv qiskit_env
source qiskit_env/bin/activate  # On Windows: qiskit_env\Scripts\activate
```

### Installation

```bash
# Clone the repository
git clone https://github.com/rjamoriz/SpaceEarthQuantumOrbits.git
cd SpaceEarthQuantumOrbits

# Install dependencies
pip install qiskit qiskit-optimization numpy matplotlib requests
```

### Run the Demo (No API Key Required)

```bash
python test_telemetry_integration.py
```

This demonstrates the complete workflow with simulated satellite data:
- ✅ Telemetry generation for 6 satellites
- ✅ Performance-weighted conflict analysis
- ✅ QUBO optimization
- ✅ Detailed performance analytics

### Run with Real Satellite Data

1. **Get a free API key** from [N2YO.com](https://www.n2yo.com/)

2. **Configure the API key** in `run_qiskit_optimization.py`:
   ```python
   API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
   ```

3. **Run the optimization**:
   ```bash
   python run_qiskit_optimization.py
   ```

### Generate Photorealistic Visualization

```bash
python photorealistic_earth_viz.py
```

Output: `starlink_photorealistic_earth.png` (7200x7200 px, ~6 MB)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Real-Time Satellite Data                      │
│              (N2YO API or Simulated Positions)               │
│            Position: latitude, longitude, altitude           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Starlink Telemetry Generator Module                 │
│         (starlink_telemetry.py - Statistical Models)         │
│                                                              │
│  • Weather-dependent performance modeling                    │
│  • Altitude-based signal quality adjustments                 │
│  • Realistic throughput and packet loss simulation           │
│  • QoE score calculation (0-10 scale)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             Augmented Satellite Data                         │
│        Position + Telemetry (13 metrics per satellite)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│        Performance-Weighted Conflict Matrix                  │
│                                                              │
│  • Spatial conflicts (distance < threshold)                  │
│  • Weighted by QoE score and performance metrics             │
│  • Poor performance = higher conflict weight                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             QUBO Optimization (Qiskit)                       │
│                                                              │
│  • Max-Cut problem formulation                               │
│  • Quantum-inspired optimization                             │
│  • Partitions satellites into optimal orbit sets            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Optimized Orbit Partitioning + Analytics            │
│                                                              │
│  • Set A & Set B satellite assignments                       │
│  • Performance analysis per orbit set                        │
│  • QoE, throughput, packet loss statistics                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
SpaceEarthQuantumOrbits/
│
├── 📊 Core Modules
│   ├── starlink_telemetry.py              # Telemetry generation engine
│   ├── run_qiskit_optimization.py         # Main optimization script
│   └── test_telemetry_integration.py      # Demo script (no API key)
│
├── 🎨 Visualization
│   ├── photorealistic_earth_viz.py        # 3D Earth rendering
│   ├── satellite_visualization.ipynb      # Interactive Jupyter notebook
│   └── starlink_3d_quantum_optimization.ipynb  # Advanced analysis
│
├── 🖼️ Assets
│   ├── starlink_photorealistic_earth.png  # Main visualization (6.8 MB)
│   └── starlink_reference_image.png       # Reference comparison
│
├── 📚 Documentation
│   ├── README.md                          # This file
│   ├── QUICKSTART.md                      # Quick start guide
│   ├── README_TELEMETRY.md                # Telemetry documentation
│   ├── README_PHOTOREALISTIC_VIZ.md       # Visualization guide
│   └── SAVE_REFERENCE_IMAGE.md            # Image saving instructions
│
└── 🛠️ Utilities
    └── add_header_image.py                # Image processing utility
```

---

## 🔬 Technical Details

### Telemetry Schema (13 Columns)

| Column | Description | Unit/Range |
|--------|-------------|------------|
| `S_ID` | Satellite identifier | Integer |
| `latitude` | Geographic latitude | -90° to 90° |
| `longitude` | Geographic longitude | -180° to 180° |
| `altitude_km` | Orbital altitude | 340-1325 km |
| `season` | Observation season | Spring/Summer/Fall/Winter |
| `weather` | Weather conditions | clear/rain/snow/cloudy |
| `visible_satellites` | Satellites in view | 4-20 |
| `serving_satellites` | Active connections | 1-4 |
| `signal_loss_db` | Signal attenuation | 0.5-15 dB |
| `download_throughput_mbps` | Download speed | 30-150 Mbps |
| `upload_throughput_mbps` | Upload speed | 5-20 Mbps |
| `packet_loss_percent` | Data loss rate | 0.05-5% |
| `qoe_score` | Quality of Experience | 0-10 |

### Weather Impact on Performance

| Weather | Signal Loss | Download Speed | Packet Loss | QoE Score |
|---------|-------------|----------------|-------------|-----------|
| ☀️ Clear | 2.5 dB | 150 Mbps | 0.1% | 9.5/10 |
| 🌤️ Partly Cloudy | 3.2 dB | 130 Mbps | 0.2% | 8.8/10 |
| ☁️ Cloudy | 4.5 dB | 100 Mbps | 0.5% | 7.5/10 |
| 🌧️ Rain | 6.8 dB | 70 Mbps | 1.5% | 6.2/10 |
| ⛈️ Heavy Rain | 12.5 dB | 30 Mbps | 5.0% | 3.5/10 |
| ❄️ Snow | 8.5 dB | 50 Mbps | 2.5% | 5.0/10 |

### Constellation Architecture

| Shell | Altitude Range | Satellites | Coverage | Inclination |
|-------|---------------|------------|----------|-------------|
| Shell 1 | 340-550 km | 60,000 (60%) | ±53° latitude | 53° |
| Shell 2 | 550-1150 km | 30,000 (30%) | ±70° latitude | 70° |
| Shell 3 | 1150-1325 km | 10,000 (10%) | ±85° latitude | 85° |

---

## 💡 Examples

### Example 1: Basic Telemetry Generation

```python
from starlink_telemetry import StarlinkTelemetryGenerator

# Initialize generator
generator = StarlinkTelemetryGenerator(seed=42)

# Define satellite positions
positions = {
    25544: {'lat': 45.5, 'lng': -122.6, 'alt': 420},  # ISS
    20580: {'lat': 28.5, 'lng': -80.5, 'alt': 547},   # Hubble
}

# Generate telemetry
augmented_data = generator.augment_satellite_positions(positions)

# Access metrics
for sat_id, data in augmented_data.items():
    print(f"Satellite {sat_id}:")
    print(f"  QoE Score: {data['qoe_score']}/10")
    print(f"  Download: {data['download_throughput_mbps']} Mbps")
    print(f"  Weather: {data['weather']}")
```

### Example 2: Custom Visualization

```python
from photorealistic_earth_viz import create_photorealistic_visualization

# Create visualization with custom view
fig = create_photorealistic_visualization(
    view='europe',        # Focus on Europe
    n_display=15000,      # Display 15,000 satellites
    figsize=(20, 20)      # Larger figure size
)

# Save high-resolution image
fig.savefig('custom_view.png', dpi=300, bbox_inches='tight')
```

### Example 3: Performance Analysis

```python
from starlink_telemetry import StarlinkTelemetryGenerator, print_telemetry_summary

generator = StarlinkTelemetryGenerator()
positions = {...}  # Your satellite positions
augmented_data = generator.augment_satellite_positions(positions)

# Print detailed summary
print_telemetry_summary(augmented_data)

# Calculate average metrics
avg_qoe = sum(d['qoe_score'] for d in augmented_data.values()) / len(augmented_data)
avg_throughput = sum(d['download_throughput_mbps'] for d in augmented_data.values()) / len(augmented_data)

print(f"Average QoE: {avg_qoe:.2f}/10")
print(f"Average Throughput: {avg_throughput:.1f} Mbps")
```

---

## 📊 Sample Output

```
================================================================================
STARLINK TELEMETRY SUMMARY
================================================================================

Satellite ID: 25544 (INTERNATIONAL SPACE STATION)
  Position: (45.50°, -122.60°) @ 420 km
  Weather: clear | Season: Summer
  Satellites: 3/16 serving/visible
  Signal Loss: 2.23 dB
  Throughput: ↓152.7 Mbps / ↑25.4 Mbps
  Packet Loss: 0.10%
  QoE Score: 9.5/10

Satellite ID: 20580 (HUBBLE SPACE TELESCOPE)
  Position: (28.50°, -80.50°) @ 547 km
  Weather: rain | Season: Fall
  Satellites: 2/14 serving/visible
  Signal Loss: 7.12 dB
  Throughput: ↓68.3 Mbps / ↑12.1 Mbps
  Packet Loss: 1.65%
  QoE Score: 6.1/10

================================================================================
PERFORMANCE-WEIGHTED CONFLICT MATRIX
================================================================================
[[0.    0.    0.    1.216]
 [0.    0.    1.892 0.   ]
 [0.    1.892 0.    2.145]
 [1.216 0.    2.145 0.   ]]

Optimal solution: [0 1 0 1]

================================================================================
PERFORMANCE ANALYSIS BY ORBIT SET
================================================================================

Set A (Orbit 1):
  Average QoE Score: 9.05/10
  Average Download Throughput: 142.3 Mbps
  Average Packet Loss: 0.28%
    - Satellite 25544: QoE=9.5, Weather=clear
    - Satellite 24876: QoE=8.6, Weather=partly_cloudy

Set B (Orbit 2):
  Average QoE Score: 6.82/10
  Average Download Throughput: 75.1 Mbps
  Average Packet Loss: 1.42%
    - Satellite 20580: QoE=6.1, Weather=rain
    - Satellite 25338: QoE=7.5, Weather=cloudy
```

---

## 🎓 Use Cases

### 🛰️ Satellite Network Optimization
- Minimize collision risks in dense constellations
- Optimize orbital slot assignments
- Balance network performance across regions

### 📡 Telecommunications
- Predict service quality under various weather conditions
- Plan ground station locations
- Optimize satellite handover strategies

### 🔬 Research & Education
- Study quantum optimization algorithms
- Analyze satellite constellation dynamics
- Model atmospheric effects on RF signals

### 🏢 Commercial Applications
- Satellite internet service planning
- Network capacity forecasting
- Performance benchmarking

---

## 🛠️ Advanced Configuration

### Customize Conflict Threshold

```python
# In run_qiskit_optimization.py
CONFLICT_THRESHOLD_KM = 1000  # Adjust distance threshold (km)
```

### Modify Weather Probabilities

```python
# In starlink_telemetry.py
weather_probs = {
    'clear': 0.50,         # 50% clear weather
    'partly_cloudy': 0.20,
    'cloudy': 0.15,
    'rain': 0.10,
    'heavy_rain': 0.03,
    'snow': 0.02
}
```

### Change Visualization Settings

```python
# In photorealistic_earth_viz.py
fig = create_photorealistic_visualization(
    view='north_america',
    n_display=12000,
    figsize=(24, 24),
    dpi=300
)
```

---

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get up and running in 5 minutes
- **[Telemetry Documentation](README_TELEMETRY.md)** - Detailed telemetry system guide
- **[Visualization Guide](README_PHOTOREALISTIC_VIZ.md)** - 3D rendering documentation
- **[Qiskit Documentation](https://qiskit.org/documentation/)** - Quantum computing framework

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution
- 🔬 Additional quantum algorithms (QAOA, VQE)
- 📊 Enhanced visualization features
- 🌍 Real-world dataset integration
- 📝 Documentation improvements
- 🐛 Bug fixes and optimizations

---

## 🔮 Future Enhancements

- [ ] **Real-time tracking** integration with multiple satellite APIs
- [ ] **Machine learning** models for performance prediction
- [ ] **Multi-objective optimization** (cost, performance, coverage)
- [ ] **Time-series analysis** of orbital dynamics
- [ ] **Interactive web dashboard** for visualization
- [ ] **GPU acceleration** for large-scale simulations
- [ ] **Quantum hardware** backend support (IBM Quantum, AWS Braket)
- [ ] **Geographic clustering** for regional optimization

---

## 📖 References

### Datasets
- [Starlink Telemetry Dataset](https://www.opendatabay.com/data/science-research/ce2cc978-11e7-462c-ab7e-3c06e6841ea3) - 100,000 rows of performance data
- [N2YO Satellite API](https://www.n2yo.com/api/) - Real-time satellite tracking
- [CelesTrak](https://celestrak.com/) - NORAD satellite catalog

### Technologies
- [Qiskit](https://qiskit.org/) - Quantum computing framework
- [Qiskit Optimization](https://qiskit.org/documentation/optimization/) - QUBO solver
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Visualization library

### Research Papers
- [QUBO Formulations](https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization)
- [Max-Cut Problem](https://en.wikipedia.org/wiki/Maximum_cut)
- [Satellite Constellation Design](https://www.sciencedirect.com/topics/engineering/satellite-constellation)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Ruben Amoriz**
- GitHub: [@rjamoriz](https://github.com/rjamoriz)
- Project: [SpaceEarthQuantumOrbits](https://github.com/rjamoriz/SpaceEarthQuantumOrbits)

---

## 🙏 Acknowledgments

- **SpaceX Starlink** - Inspiration for constellation architecture
- **Qiskit Team** - Quantum optimization framework
- **OpenDataBay** - Telemetry dataset patterns
- **N2YO** - Real-time satellite tracking API
- **Open Source Community** - Tools and libraries

---

## 📞 Support

- 📧 **Issues**: [GitHub Issues](https://github.com/rjamoriz/SpaceEarthQuantumOrbits/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/rjamoriz/SpaceEarthQuantumOrbits/discussions)
- 📖 **Documentation**: [Wiki](https://github.com/rjamoriz/SpaceEarthQuantumOrbits/wiki)

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ and ⚛️ quantum computing

![Satellite](https://img.shields.io/badge/🛰️-Satellites-blue) ![Quantum](https://img.shields.io/badge/⚛️-Quantum-purple) ![Earth](https://img.shields.io/badge/🌍-Earth-green)

</div>
