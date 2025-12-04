# Quick Start Guide - Starlink Telemetry Integration

## 🚀 Run the Demo (No API Key Required)

```bash
cd "/Users/Ruben_MACPRO/Desktop/IA DevOps/SpaceEarth"
./qiskit_env/bin/python test_telemetry_integration.py
```

This will demonstrate the complete workflow with simulated satellite data.

## 📊 What You'll See

1. **Telemetry Generation**: Synthetic Starlink performance metrics for 6 satellites
   - Weather conditions (clear, cloudy, rain, etc.)
   - Signal loss, throughput, packet loss
   - Quality of Experience (QoE) scores

2. **Conflict Analysis**: Two types of conflict matrices
   - Standard (distance-only)
   - Performance-weighted (considers QoE and service quality)

3. **Quantum Optimization**: QUBO solution using Qiskit
   - Optimal satellite partitioning into two orbit sets
   - Minimizes conflicts while balancing performance

4. **Performance Analytics**: Detailed metrics per orbit set
   - Average QoE, throughput, packet loss
   - Individual satellite performance breakdown

## 🔧 Use with Real Satellite Data

### Step 1: Get N2YO API Key
Visit [https://www.n2yo.com/](https://www.n2yo.com/) and sign up for a free API key.

### Step 2: Configure API Key
Edit `run_qiskit_optimization.py`:
```python
API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
```

### Step 3: Run with Real Data
```bash
./qiskit_env/bin/python run_qiskit_optimization.py
```

## 📁 Project Structure

```
SpaceEarth/
├── starlink_telemetry.py          # Core telemetry generation module
├── run_qiskit_optimization.py     # Main optimization script (needs API key)
├── test_telemetry_integration.py  # Demo script (no API key needed)
├── satellite_visualization.ipynb  # 3D visualization notebook
├── README_TELEMETRY.md            # Full documentation
└── QUICKSTART.md                  # This file
```

## 🎯 Key Features Implemented

### ✅ Option 3: Augment Real-Time Data with Synthetic Telemetry

- **13-column schema** from opendatabay.com Starlink dataset
- **Weather-dependent performance** modeling
- **Altitude-based adjustments** for signal quality
- **Performance-weighted optimization** in QUBO formulation
- **QoE scoring** (0-10 scale) based on multiple metrics

### 📈 Telemetry Metrics Generated

| Metric | Description | Realistic Range |
|--------|-------------|-----------------|
| Signal Loss | Atmospheric interference | 0.5-15 dB |
| Download Speed | Downlink throughput | 30-150 Mbps |
| Upload Speed | Uplink throughput | 5-20 Mbps |
| Packet Loss | Data loss rate | 0.05-5% |
| QoE Score | Overall quality | 0-10 |
| Visible Satellites | In view | 4-20 |
| Serving Satellites | Active connection | 1-4 |

### 🌦️ Weather Impact Modeling

Performance degrades realistically based on weather:
- **Clear**: Best performance (150 Mbps, 0.1% loss)
- **Partly Cloudy**: Slight degradation (130 Mbps, 0.2% loss)
- **Cloudy**: Moderate impact (100 Mbps, 0.5% loss)
- **Rain**: Significant degradation (70 Mbps, 1.5% loss)
- **Heavy Rain**: Severe impact (30 Mbps, 5% loss)
- **Snow**: Cold weather effects (50 Mbps, 2.5% loss)

## 🔬 How It Works

```
Real Satellite Positions (N2YO API or Simulated)
            ↓
Telemetry Generator (starlink_telemetry.py)
  • Applies weather-based performance models
  • Adjusts for altitude and latitude
  • Generates 13 metrics per satellite
            ↓
Augmented Data (Position + Performance)
            ↓
Conflict Matrix Builder
  • Spatial conflicts (distance threshold)
  • Performance weighting (QoE-based)
            ↓
QUBO Optimization (Qiskit)
  • Max-Cut problem formulation
  • Quantum-inspired solver
            ↓
Optimal Orbit Partitioning + Analytics
```

## 💡 Example Output Snippet

```
Satellite ID: 25544
  Position: (45.50°, -122.60°) @ 420 km
  Weather: clear | Season: Spring
  Satellites: 3/16 serving/visible
  Signal Loss: 2.23 dB
  Throughput: ↓152.7 Mbps / ↑25.4 Mbps
  Packet Loss: 0.10%
  QoE Score: 9.5/10

Performance-Weighted Conflict Matrix:
[[0.    0.    0.   ]
 [0.    0.    1.216]
 [0.    1.216 0.   ]]

Set A: Average QoE Score: 9.05/10
Set B: Average QoE Score: 9.11/10
```

## 🐛 Troubleshooting

### ImportError: Qiskit version conflict
Use the virtual environment:
```bash
./qiskit_env/bin/python <script_name>.py
```

### API Key Issues
- Verify key is correct in `run_qiskit_optimization.py`
- Check N2YO API rate limits (free tier: 1000 requests/hour)
- Use `test_telemetry_integration.py` for testing without API

### Module Not Found
Install dependencies in virtual environment:
```bash
./qiskit_env/bin/pip install qiskit qiskit-optimization numpy requests
```

## 📚 Learn More

- **Full Documentation**: See `README_TELEMETRY.md`
- **Dataset Source**: [opendatabay.com](https://www.opendatabay.com/data/science-research/ce2cc978-11e7-462c-ab7e-3c06e6841ea3)
- **Qiskit Docs**: [qiskit.org](https://qiskit.org/documentation/)

## 🎓 Next Steps

1. **Experiment with parameters**:
   - Adjust `CONFLICT_THRESHOLD_KM` in scripts
   - Modify weather probability distributions
   - Change number of satellites

2. **Visualize results**:
   - Open `satellite_visualization.ipynb`
   - Add telemetry overlays to 3D plots

3. **Extend functionality**:
   - Download actual CSV from opendatabay.com
   - Add time-series analysis
   - Implement multi-objective optimization

## ✨ Summary

You now have a complete system that:
- ✅ Generates realistic Starlink telemetry data
- ✅ Augments real-time satellite positions with performance metrics
- ✅ Optimizes satellite orbits using quantum-inspired algorithms
- ✅ Balances spatial conflicts with service quality
- ✅ Provides detailed performance analytics

**Ready to run? Execute the demo command at the top of this file!** 🚀
