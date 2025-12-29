#!/usr/bin/env python3
"""
Test script to verify pandas CSV loading works correctly
"""

import pandas as pd
import sys

print("="*70)
print("📥 Starlink Telemetry Dataset Analysis")
print("="*70)

try:
    # Test pandas version
    print(f"✓ Pandas version: {pd.__version__}")
    
    # Load dataset
    print("✓ Loading dataset from starlink_telemetry_dataset.csv...")
    df = pd.read_csv('starlink_telemetry_dataset.csv')
    
    print(f"\n✅ SUCCESS! Dataset loaded successfully")
    print(f"\n📊 Dataset Information:")
    print(f"   - Rows: {len(df):,}")
    print(f"   - Columns: {len(df.columns)}")
    print(f"   - Size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    print(f"\n📋 Columns:")
    for col in df.columns:
        print(f"   - {col}")
    
    print(f"\n📈 Sample Statistics:")
    print(f"   - Average QoE Score: {df['qoe_score'].mean():.2f}/10")
    print(f"   - Average Download: {df['download_throughput_bps'].mean() / 1_000_000:.1f} Mbps")
    print(f"   - Average Signal Loss: {df['signal_loss_db'].mean():.2f} dB")
    
    print(f"\n🌦️  Weather Distribution:")
    weather_counts = df['weather'].value_counts()
    for weather, count in weather_counts.items():
        print(f"   - {weather}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n✅ All tests passed! The pandas issue is resolved.")
    
except Exception as e:
    print(f"\n⚠️  Error: {e}")
    print(f"   Error type: {type(e).__name__}")
    print(f"\n💡 Solution: The pandas version has been upgraded to fix the TypeVar issue.")
    sys.exit(1)

print("="*70)
