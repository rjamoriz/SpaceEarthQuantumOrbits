#!/usr/bin/env python3
"""
Test script to replicate cell 15 behavior and identify issues
"""

import plotly.graph_objects as go

# Dataset URL (you may need to download manually from opendatabay.com)
DATASET_URL = "https://www.opendatabay.com/data/science-research/ce2cc978-11e7-462c-ab7e-3c06e6841ea3"
LOCAL_CSV_PATH = "starlink_telemetry_dataset.csv"  # Path to downloaded CSV

print("📥 Starlink Telemetry Dataset Analysis")
print("=" * 70)

# Check if CSV exists locally
import os
if os.path.exists(LOCAL_CSV_PATH):
    print(f"✓ Loading dataset from {LOCAL_CSV_PATH}...\n")
    
    try:
        import pandas as pd
        
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
            # Don't show in script, just create
            print("✅ Visualization created successfully!")
    
    except Exception as e:
        print(f"\n⚠️  Error loading dataset: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        print("\n💡 This might be a pandas version compatibility issue.")
        print("   Try upgrading pandas: pip install --upgrade pandas")
    
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
print("✅ Test completed!")
