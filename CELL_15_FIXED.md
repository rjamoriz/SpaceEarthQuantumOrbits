# 🔧 Cell 15 - Fixed Version

## The Problem

Your cell 15 is trying to use `go.Figure()` and `go.Histogram()` but **plotly is not imported** in that cell.

The error message shows:
```
NameError: name 'go' is not defined
```

## The Solution

Add the plotly import at the **beginning of cell 15**, before the dataset loading code.

## ✅ Fixed Cell 15 Code

Replace your current cell 15 with this:

```python
# Cell 15: Starlink Telemetry Dataset Analysis with Plotly

# ========== IMPORTS (ADD THIS!) ==========
import plotly.graph_objects as go
import pandas as pd
import os

# ========== DATASET CONFIGURATION ==========
DATASET_URL = "https://www.opendatabay.com/data/science-research/ce2cc978-11e7-462c-ab7e-3c06e6841ea3"
LOCAL_CSV_PATH = "starlink_telemetry_dataset.csv"

print("📥 Starlink Telemetry Dataset Analysis")
print("=" * 70)

# Check if CSV exists locally
if os.path.exists(LOCAL_CSV_PATH):
    print(f"✓ Loading dataset from {LOCAL_CSV_PATH}...\n")
    
    try:
        # Load the dataset
        df = pd.read_csv(LOCAL_CSV_PATH)
        
        print(f"📊 Dataset Overview:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"\n  Column Names:")
        for col in df.columns:
            print(f"    - {col}")
        
        # Display first few rows
        print(f"\n📋 Sample Data:")
        print(df.head())
        
        # Statistical summary
        print(f"\n📈 Statistical Summary:")
        print(df.describe())
        
        # Weather distribution
        if 'weather' in df.columns:
            print(f"\n🌦️  Weather Distribution:")
            weather_counts = df['weather'].value_counts()
            for weather, count in weather_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {weather}: {count:,} ({percentage:.1f}%)")
        
        # QoE analysis
        if 'qoe_score' in df.columns:
            print(f"\n⭐ QoE Score Analysis:")
            print(f"  Mean: {df['qoe_score'].mean():.2f}")
            print(f"  Median: {df['qoe_score'].median():.2f}")
            print(f"  Std Dev: {df['qoe_score'].std():.2f}")
            print(f"  Min: {df['qoe_score'].min():.2f}")
            print(f"  Max: {df['qoe_score'].max():.2f}")
        
        # Throughput analysis
        if 'download_throughput_bps' in df.columns:
            print(f"\n📶 Throughput Analysis:")
            df['download_mbps'] = df['download_throughput_bps'] / 1e6
            print(f"  Mean Download: {df['download_mbps'].mean():.1f} Mbps")
            print(f"  Median Download: {df['download_mbps'].median():.1f} Mbps")
            print(f"  Max Download: {df['download_mbps'].max():.1f} Mbps")
        
        # Visualize distributions
        print(f"\n📊 Creating visualizations...")
        
        # QoE distribution by weather
        if 'qoe_score' in df.columns and 'weather' in df.columns:
            fig_hist = go.Figure()
            
            for weather in df['weather'].unique():
                weather_data = df[df['weather'] == weather]['qoe_score']
                fig_hist.add_trace(go.Histogram(
                    x=weather_data,
                    name=weather,
                    opacity=0.7,
                    nbinsx=20
                ))
            
            fig_hist.update_layout(
                title='QoE Score Distribution by Weather Condition',
                xaxis_title='QoE Score',
                yaxis_title='Frequency',
                barmode='overlay',
                height=500
            )
            fig_hist.show()
            
            print("✅ Visualization created successfully!")
    
    except Exception as e:
        print(f"\n⚠️  Error loading dataset: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Show detailed error for debugging
        import traceback
        print("\n🔍 Full error details:")
        traceback.print_exc()
        
        print("\n💡 Troubleshooting:")
        print("   1. Make sure plotly is installed: pip install plotly")
        print("   2. Restart kernel: Kernel → Restart Kernel")
        print("   3. Re-run all cells from the beginning")
    
else:
    print(f"⚠️  Dataset not found at {LOCAL_CSV_PATH}")
    print(f"\n📥 To download the dataset:")
    print(f"  1. Visit: {DATASET_URL}")
    print(f"  2. Download the CSV file")
    print(f"  3. Save it as '{LOCAL_CSV_PATH}' in this directory")
    print(f"\n💡 Dataset Details:")
    print(f"  - Volume: 100,000 rows × 13 columns")
    print(f"  - Size: 9.34 MB")
    print(f"  - Format: CSV")
    print(f"  - Quality: 5/5 (UDQS certified)")
    print(f"\n📝 Note: The dataset file is not included in this repository.")
    print(f"   This cell will work once you download and place the CSV file.")

print("\n" + "=" * 70)
```

## Key Changes:

### ✅ Added at the top:
```python
import plotly.graph_objects as go
import pandas as pd
import os
```

### ✅ Added success message:
```python
print("✅ Visualization created successfully!")
```

### ✅ Enhanced error handling:
```python
import traceback
traceback.print_exc()
```

## Alternative: Check Earlier Cells

If you want to keep imports in a separate cell, make sure you have this **before cell 15**:

```python
# Cell (before 15): Import Libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os

print("✅ All libraries imported successfully!")
```

## What's Happening Now

Based on your output:
- ✅ **pandas is working** - Dataset loads successfully
- ✅ **100,000 rows loaded**
- ✅ **All statistics calculated**
- ❌ **plotly not imported** - Causing NameError when creating visualization

## Quick Fix Steps:

1. **Add this at the top of cell 15:**
   ```python
   import plotly.graph_objects as go
   ```

2. **Or run this in a new cell before cell 15:**
   ```python
   import plotly.graph_objects as go
   print("✅ Plotly imported!")
   ```

3. **Then re-run cell 15**

The visualization will work perfectly! 🎉

## Note About the Warning

The warning about `numexpr` is harmless:
```
UserWarning: Pandas requires version '2.8.4' or newer of 'numexpr' (version '2.8.3' currently installed).
```

You can ignore it or fix it with:
```bash
pip install --upgrade numexpr
```

But it won't affect your analysis!
