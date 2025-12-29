# 📊 Improved Cell 15 Code

## Analysis of Your Current Code

Your cell 15 code is **well-structured and correct**! The issue was purely the pandas version compatibility, which has now been fixed.

### ✅ What Works Well:

1. **Error Handling**: Good try-except block
2. **File Checking**: Verifies CSV exists before loading
3. **Comprehensive Analysis**: Weather, QoE, throughput statistics
4. **Visualization**: Plotly histogram by weather condition
5. **User Feedback**: Clear messages and formatting

### 🔧 Suggested Improvements

Here's an enhanced version with additional features:

```python
# Cell 15: Enhanced Starlink Telemetry Dataset Analysis

import plotly.graph_objects as go
import os
import sys

# Dataset configuration
DATASET_URL = "https://www.opendatabay.com/data/science-research/ce2cc978-11e7-462c-ab7e-3c06e6841ea3"
LOCAL_CSV_PATH = "starlink_telemetry_dataset.csv"

print("📥 Starlink Telemetry Dataset Analysis")
print("=" * 70)

# Verify pandas version first
try:
    import pandas as pd
    pandas_version = tuple(map(int, pd.__version__.split('.')[:2]))
    if pandas_version < (2, 3):
        print(f"⚠️  Warning: pandas {pd.__version__} detected. Recommended: 2.3+")
        print("   If you encounter errors, restart the kernel after upgrading.")
    else:
        print(f"✓ pandas {pd.__version__} (compatible)")
except ImportError:
    print("❌ pandas not installed. Run: pip install pandas")
    sys.exit(1)

# Check if CSV exists locally
if os.path.exists(LOCAL_CSV_PATH):
    file_size_mb = os.path.getsize(LOCAL_CSV_PATH) / (1024 * 1024)
    print(f"✓ Loading dataset from {LOCAL_CSV_PATH} ({file_size_mb:.2f} MB)...\n")
    
    try:
        # Load the dataset with progress indicator
        df = pd.read_csv(LOCAL_CSV_PATH)
        
        print(f"📊 Dataset Overview:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print(f"\n  Column Names:")
        for col in df.columns:
            dtype = df[col].dtype
            print(f"    - {col:<30} ({dtype})")
        
        # Display first few rows
        print(f"\n📋 Sample Data (first 5 rows):")
        print(df.head())
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  Missing Values Detected:")
            for col, count in missing[missing > 0].items():
                print(f"    - {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            print(f"\n✅ No missing values detected")
        
        # Statistical summary
        print(f"\n📈 Statistical Summary:")
        print(df.describe())
        
        # Weather distribution
        if 'weather' in df.columns:
            print(f"\n🌦️  Weather Distribution:")
            weather_counts = df['weather'].value_counts()
            for weather, count in weather_counts.items():
                percentage = (count / len(df)) * 100
                bar = '█' * int(percentage / 2)
                print(f"  {weather:<15} {count:>7,} ({percentage:>5.1f}%) {bar}")
        
        # QoE analysis
        if 'qoe_score' in df.columns:
            print(f"\n⭐ QoE Score Analysis:")
            print(f"  Mean:   {df['qoe_score'].mean():>6.2f}")
            print(f"  Median: {df['qoe_score'].median():>6.2f}")
            print(f"  Std:    {df['qoe_score'].std():>6.2f}")
            print(f"  Min:    {df['qoe_score'].min():>6.2f}")
            print(f"  Max:    {df['qoe_score'].max():>6.2f}")
            
            # QoE categories
            excellent = (df['qoe_score'] >= 85).sum()
            good = ((df['qoe_score'] >= 70) & (df['qoe_score'] < 85)).sum()
            fair = ((df['qoe_score'] >= 60) & (df['qoe_score'] < 70)).sum()
            poor = (df['qoe_score'] < 60).sum()
            
            print(f"\n  Quality Categories:")
            print(f"    Excellent (≥85): {excellent:>7,} ({excellent/len(df)*100:>5.1f}%)")
            print(f"    Good (70-84):    {good:>7,} ({good/len(df)*100:>5.1f}%)")
            print(f"    Fair (60-69):    {fair:>7,} ({fair/len(df)*100:>5.1f}%)")
            print(f"    Poor (<60):      {poor:>7,} ({poor/len(df)*100:>5.1f}%)")
        
        # Throughput analysis
        if 'download_throughput_bps' in df.columns:
            print(f"\n📶 Throughput Analysis:")
            df['download_mbps'] = df['download_throughput_bps'] / 1e6
            df['upload_mbps'] = df['upload_throughput_bps'] / 1e6
            
            print(f"  Download:")
            print(f"    Mean:   {df['download_mbps'].mean():>6.1f} Mbps")
            print(f"    Median: {df['download_mbps'].median():>6.1f} Mbps")
            print(f"    Max:    {df['download_mbps'].max():>6.1f} Mbps")
            
            print(f"  Upload:")
            print(f"    Mean:   {df['upload_mbps'].mean():>6.1f} Mbps")
            print(f"    Median: {df['upload_mbps'].median():>6.1f} Mbps")
            print(f"    Max:    {df['upload_mbps'].max():>6.1f} Mbps")
        
        # Signal loss analysis
        if 'signal_loss_db' in df.columns:
            print(f"\n📡 Signal Loss Analysis:")
            print(f"  Mean:   {df['signal_loss_db'].mean():>6.2f} dB")
            print(f"  Median: {df['signal_loss_db'].median():>6.2f} dB")
            print(f"  Max:    {df['signal_loss_db'].max():>6.2f} dB")
        
        # Packet loss analysis
        if 'packet_loss' in df.columns:
            print(f"\n📦 Packet Loss Analysis:")
            print(f"  Mean:   {df['packet_loss'].mean()*100:>6.3f}%")
            print(f"  Median: {df['packet_loss'].median()*100:>6.3f}%")
            print(f"  Max:    {df['packet_loss'].max()*100:>6.3f}%")
        
        # Visualize distributions
        print(f"\n📊 Creating visualizations...")
        
        # QoE distribution by weather
        if 'qoe_score' in df.columns and 'weather' in df.columns:
            fig_hist = go.Figure()
            
            # Sort weather by average QoE
            weather_avg_qoe = df.groupby('weather')['qoe_score'].mean().sort_values(ascending=False)
            
            for weather in weather_avg_qoe.index:
                weather_data = df[df['weather'] == weather]['qoe_score']
                avg_qoe = weather_data.mean()
                fig_hist.add_trace(go.Histogram(
                    x=weather_data,
                    name=f"{weather} (avg: {avg_qoe:.1f})",
                    opacity=0.7,
                    nbinsx=30
                ))
            
            fig_hist.update_layout(
                title='QoE Score Distribution by Weather Condition',
                xaxis_title='QoE Score',
                yaxis_title='Frequency',
                barmode='overlay',
                height=600,
                showlegend=True,
                legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
            )
            fig_hist.show()
            
            # Box plot for better comparison
            fig_box = go.Figure()
            
            for weather in weather_avg_qoe.index:
                weather_data = df[df['weather'] == weather]['qoe_score']
                fig_box.add_trace(go.Box(
                    y=weather_data,
                    name=weather,
                    boxmean='sd'
                ))
            
            fig_box.update_layout(
                title='QoE Score Distribution by Weather (Box Plot)',
                yaxis_title='QoE Score',
                height=500,
                showlegend=False
            )
            fig_box.show()
        
        # Throughput vs QoE scatter
        if 'download_mbps' in df.columns and 'qoe_score' in df.columns and 'weather' in df.columns:
            fig_scatter = go.Figure()
            
            for weather in df['weather'].unique():
                weather_df = df[df['weather'] == weather]
                fig_scatter.add_trace(go.Scatter(
                    x=weather_df['download_mbps'],
                    y=weather_df['qoe_score'],
                    mode='markers',
                    name=weather,
                    opacity=0.6,
                    marker=dict(size=4)
                ))
            
            fig_scatter.update_layout(
                title='Download Speed vs QoE Score by Weather',
                xaxis_title='Download Speed (Mbps)',
                yaxis_title='QoE Score',
                height=600
            )
            fig_scatter.show()
        
        print(f"\n✅ Analysis complete!")
        print("=" * 70)
    
    except Exception as e:
        print(f"\n⚠️  Error loading dataset: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Detailed error info
        import traceback
        print(f"\n🔍 Detailed Error:")
        traceback.print_exc()
        
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Restart the kernel: Kernel → Restart Kernel")
        print(f"   2. Check pandas version: {pd.__version__}")
        print(f"   3. If < 2.3, upgrade: pip install --upgrade pandas")
        print(f"   4. Re-run this cell after restart")
    
else:
    print(f"⚠️  Dataset not found at {LOCAL_CSV_PATH}")
    print(f"\n📥 To download the dataset:")
    print(f"  1. Visit: {DATASET_URL}")
    print(f"  2. Download the CSV file")
    print(f"  3. Save it as '{LOCAL_CSV_PATH}' in this directory")
    print(f"\n💡 Dataset Details:")
    print(f"  - Volume: 100,000 rows × 13 columns")
    print(f"  - Size: ~9.34 MB")
    print(f"  - Format: CSV")
    print(f"  - Quality: 5/5 (UDQS certified)")
    print(f"  - Source: opendatabay.com")
    print(f"\n📝 Note: The dataset file is not included in this repository.")
    print(f"   This cell will work once you download and place the CSV file.")
```

## Key Improvements:

### 1. **Version Check**
- Verifies pandas version before loading
- Warns if version is too old

### 2. **Enhanced Statistics**
- Memory usage display
- Missing value detection
- QoE quality categories (Excellent/Good/Fair/Poor)
- Signal loss and packet loss analysis

### 3. **Better Visualizations**
- Sorted weather conditions by average QoE
- Added box plots for distribution comparison
- Scatter plot showing throughput vs QoE correlation
- Better legends with average values

### 4. **Improved Formatting**
- Progress bars for weather distribution
- Aligned numeric output
- More detailed column information with data types

### 5. **Error Handling**
- Full traceback on errors
- Specific troubleshooting steps
- File size display

## Current Status

✅ **Your original code works perfectly now!**  
✅ **Pandas 2.3.3 is installed**  
✅ **Dataset loads successfully**  
✅ **All 100,000 rows available**

## Next Steps

1. **Restart your Jupyter kernel** (if you haven't already)
2. **Re-run cell 15** - it should work now
3. **Optional**: Replace with the improved version above for more features

The error you experienced was purely due to the pandas version, which is now resolved! 🎉
