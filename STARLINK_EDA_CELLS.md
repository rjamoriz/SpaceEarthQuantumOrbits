# 📊 Complete Exploratory Data Analysis - Starlink Telemetry Dataset

## Add these cells at the end of your notebook for comprehensive EDA

---

## Cell 1: Load Data and Initial Overview

```python
# Cell: Load and Initial Data Overview

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("STARLINK TELEMETRY DATASET - EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load dataset
df = pd.read_csv('starlink_telemetry_dataset.csv')

print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "="*80)
print("COLUMN INFORMATION")
print("="*80)
print(df.info())

print("\n" + "="*80)
print("FIRST 10 ROWS")
print("="*80)
print(df.head(10))

print("\n" + "="*80)
print("LAST 10 ROWS")
print("="*80)
print(df.tail(10))

print("\n" + "="*80)
print("RANDOM SAMPLE (10 ROWS)")
print("="*80)
print(df.sample(10, random_state=42))
```

---

## Cell 2: Data Quality Assessment

```python
# Cell: Data Quality Assessment

print("="*80)
print("DATA QUALITY ASSESSMENT")
print("="*80)

# Missing values
print("\n📋 Missing Values Analysis:")
print("-" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percentage': missing_pct.values
}).sort_values('Missing_Count', ascending=False)

print(missing_df)

if missing.sum() == 0:
    print("\n✅ No missing values found!")
else:
    print(f"\n⚠️  Total missing values: {missing.sum():,}")

# Duplicate rows
print("\n" + "="*80)
print("DUPLICATE ANALYSIS")
print("="*80)
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")

if duplicates > 0:
    print("\nSample of duplicate rows:")
    print(df[df.duplicated(keep=False)].head(10))

# Data types
print("\n" + "="*80)
print("DATA TYPES")
print("="*80)
dtype_counts = df.dtypes.value_counts()
print(dtype_counts)

# Unique values per column
print("\n" + "="*80)
print("UNIQUE VALUES PER COLUMN")
print("="*80)
unique_counts = df.nunique().sort_values(ascending=False)
for col, count in unique_counts.items():
    pct = (count / len(df)) * 100
    print(f"{col:30s}: {count:8,} unique ({pct:6.2f}%)")

# Check for constant columns
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
if constant_cols:
    print(f"\n⚠️  Constant columns (only 1 unique value): {constant_cols}")
else:
    print("\n✅ No constant columns found")
```

---

## Cell 3: Statistical Summary

```python
# Cell: Statistical Summary

print("="*80)
print("STATISTICAL SUMMARY - NUMERICAL FEATURES")
print("="*80)

# Numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}\n")

# Descriptive statistics
desc_stats = df[numerical_cols].describe()
print(desc_stats)

# Additional statistics
print("\n" + "="*80)
print("ADDITIONAL STATISTICS")
print("="*80)

additional_stats = pd.DataFrame({
    'skewness': df[numerical_cols].skew(),
    'kurtosis': df[numerical_cols].kurtosis(),
    'variance': df[numerical_cols].var(),
    'range': df[numerical_cols].max() - df[numerical_cols].min(),
    'IQR': df[numerical_cols].quantile(0.75) - df[numerical_cols].quantile(0.25)
})

print(additional_stats)

# Categorical columns
print("\n" + "="*80)
print("CATEGORICAL FEATURES SUMMARY")
print("="*80)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}\n")

for col in categorical_cols:
    print(f"\n{col}:")
    print("-" * 40)
    value_counts = df[col].value_counts()
    value_pcts = df[col].value_counts(normalize=True) * 100
    
    summary = pd.DataFrame({
        'Count': value_counts,
        'Percentage': value_pcts
    })
    print(summary)
```

---

## Cell 4: Distribution Analysis and Visualizations

```python
# Cell: Distribution Analysis

print("="*80)
print("DISTRIBUTION ANALYSIS")
print("="*80)

# Create comprehensive distribution plots
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    if idx < len(axes):
        ax = axes[idx]
        
        # Histogram with KDE
        df[col].hist(bins=50, ax=ax, alpha=0.7, edgecolor='black')
        df[col].plot(kind='kde', ax=ax, secondary_y=True, color='red', linewidth=2)
        
        ax.set_title(f'{col}\nMean: {df[col].mean():.2f}, Median: {df[col].median():.2f}',
                    fontweight='bold', fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        ax.text(0.02, 0.98, f'Skew: {skew:.2f}\nKurt: {kurt:.2f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=8)

plt.tight_layout()
plt.suptitle('Distribution Analysis - All Numerical Features', 
            fontsize=16, fontweight='bold', y=1.001)
plt.show()

# Box plots for outlier detection
print("\n" + "="*80)
print("OUTLIER DETECTION - BOX PLOTS")
print("="*80)

fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    if idx < len(axes):
        ax = axes[idx]
        
        # Box plot
        bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        
        ax.set_title(f'{col}', fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Calculate outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
        
        ax.text(0.5, 0.02, f'Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)',
               transform=ax.transAxes, ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
               fontsize=8)

plt.tight_layout()
plt.suptitle('Outlier Detection - Box Plots', 
            fontsize=16, fontweight='bold', y=1.001)
plt.show()
```

---

## Cell 5: Correlation Analysis

```python
# Cell: Correlation Analysis

print("="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Calculate correlation matrix
corr_matrix = df[numerical_cols].corr()

print("\nCorrelation Matrix:")
print(corr_matrix)

# Visualize correlation matrix
fig, ax = plt.subplots(figsize=(14, 12))

# Create heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)

ax.set_title('Correlation Matrix - All Numerical Features', 
            fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Find highly correlated pairs
print("\n" + "="*80)
print("HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.7)")
print("="*80)

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', 
                                                              key=abs, 
                                                              ascending=False)
    print(high_corr_df.to_string(index=False))
else:
    print("No highly correlated pairs found (|r| > 0.7)")

# Correlation with target variable (QoE score)
if 'qoe_score' in df.columns:
    print("\n" + "="*80)
    print("CORRELATION WITH QoE SCORE (Target Variable)")
    print("="*80)
    
    qoe_corr = df[numerical_cols].corr()['qoe_score'].sort_values(ascending=False)
    print(qoe_corr)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    qoe_corr.drop('qoe_score').plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Feature Correlation with QoE Score', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Correlation Coefficient')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
```

---

## Cell 6: Feature Relationships and Scatter Plots

```python
# Cell: Feature Relationships

print("="*80)
print("FEATURE RELATIONSHIPS - SCATTER PLOTS")
print("="*80)

# Key feature pairs for analysis
feature_pairs = [
    ('download_throughput_bps', 'upload_throughput_bps'),
    ('signal_loss_db', 'packet_loss'),
    ('visible_satellites', 'serving_satellites'),
    ('elevation_m', 'signal_loss_db'),
    ('download_throughput_bps', 'qoe_score'),
    ('packet_loss', 'qoe_score')
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (x_col, y_col) in enumerate(feature_pairs):
    if idx < len(axes) and x_col in df.columns and y_col in df.columns:
        ax = axes[idx]
        
        # Scatter plot with color by QoE if available
        if 'qoe_score' in df.columns:
            scatter = ax.scatter(df[x_col], df[y_col], 
                               c=df['qoe_score'], cmap='RdYlGn',
                               alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='QoE Score')
        else:
            ax.scatter(df[x_col], df[y_col], alpha=0.6, s=20)
        
        # Add regression line
        z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(df[x_col].sort_values(), p(df[x_col].sort_values()), 
               "r--", linewidth=2, label=f'y={z[0]:.2e}x+{z[1]:.2f}')
        
        # Calculate correlation
        corr = df[[x_col, y_col]].corr().iloc[0, 1]
        
        ax.set_xlabel(x_col, fontweight='bold')
        ax.set_ylabel(y_col, fontweight='bold')
        ax.set_title(f'{x_col} vs {y_col}\nCorrelation: {corr:.3f}',
                    fontsize=10, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Feature Relationships - Scatter Plots', 
            fontsize=16, fontweight='bold', y=1.001)
plt.show()
```

---

## Cell 7: Categorical Feature Analysis

```python
# Cell: Categorical Feature Analysis

print("="*80)
print("CATEGORICAL FEATURE ANALYSIS")
print("="*80)

# Analyze weather impact
if 'weather' in df.columns:
    print("\n📊 Weather Condition Analysis:")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Weather distribution
    ax = axes[0, 0]
    weather_counts = df['weather'].value_counts()
    weather_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title('Weather Condition Distribution', fontweight='bold', fontsize=12)
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentages on bars
    for i, v in enumerate(weather_counts.values):
        ax.text(i, v, f'{v:,}\n({v/len(df)*100:.1f}%)', 
               ha='center', va='bottom', fontweight='bold')
    
    # Weather vs QoE Score
    if 'qoe_score' in df.columns:
        ax = axes[0, 1]
        df.boxplot(column='qoe_score', by='weather', ax=ax)
        ax.set_title('QoE Score by Weather Condition', fontweight='bold', fontsize=12)
        ax.set_xlabel('Weather Condition')
        ax.set_ylabel('QoE Score')
        plt.sca(ax)
        plt.xticks(rotation=45)
        
        # Weather vs Download Throughput
        ax = axes[1, 0]
        df.boxplot(column='download_throughput_bps', by='weather', ax=ax)
        ax.set_title('Download Throughput by Weather', fontweight='bold', fontsize=12)
        ax.set_xlabel('Weather Condition')
        ax.set_ylabel('Download Throughput (bps)')
        plt.sca(ax)
        plt.xticks(rotation=45)
        
        # Weather vs Packet Loss
        ax = axes[1, 1]
        df.boxplot(column='packet_loss', by='weather', ax=ax)
        ax.set_title('Packet Loss by Weather', fontweight='bold', fontsize=12)
        ax.set_xlabel('Weather Condition')
        ax.set_ylabel('Packet Loss (%)')
        plt.sca(ax)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    print("\n📈 Statistical Tests - Weather Impact:")
    print("-" * 60)
    
    weather_groups = [group['qoe_score'].values for name, group in df.groupby('weather')]
    f_stat, p_value = stats.f_oneway(*weather_groups)
    
    print(f"ANOVA Test (Weather vs QoE Score):")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("  ✅ Significant difference between weather conditions (p < 0.05)")
    else:
        print("  ❌ No significant difference between weather conditions (p >= 0.05)")

# Analyze season impact
if 'season' in df.columns:
    print("\n" + "="*80)
    print("SEASON ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Season distribution
    ax = axes[0]
    season_counts = df['season'].value_counts()
    season_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
    ax.set_title('Season Distribution', fontweight='bold', fontsize=12)
    ax.set_xlabel('Season')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Season vs QoE
    if 'qoe_score' in df.columns:
        ax = axes[1]
        df.boxplot(column='qoe_score', by='season', ax=ax)
        ax.set_title('QoE Score by Season', fontweight='bold', fontsize=12)
        ax.set_xlabel('Season')
        ax.set_ylabel('QoE Score')
    
    plt.tight_layout()
    plt.show()
```

---

## Cell 8: Geographic Analysis

```python
# Cell: Geographic Analysis

print("="*80)
print("GEOGRAPHIC ANALYSIS")
print("="*80)

if 'latitude' in df.columns and 'longitude' in df.columns:
    
    # Geographic distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Latitude distribution
    ax = axes[0, 0]
    df['latitude'].hist(bins=50, ax=ax, color='lightblue', edgecolor='black')
    ax.set_title('Latitude Distribution', fontweight='bold', fontsize=12)
    ax.set_xlabel('Latitude (degrees)')
    ax.set_ylabel('Frequency')
    ax.axvline(df['latitude'].mean(), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {df["latitude"].mean():.2f}°')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Longitude distribution
    ax = axes[0, 1]
    df['longitude'].hist(bins=50, ax=ax, color='lightgreen', edgecolor='black')
    ax.set_title('Longitude Distribution', fontweight='bold', fontsize=12)
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Frequency')
    ax.axvline(df['longitude'].mean(), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {df["longitude"].mean():.2f}°')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Geographic scatter plot
    ax = axes[1, 0]
    if 'qoe_score' in df.columns:
        scatter = ax.scatter(df['longitude'], df['latitude'], 
                           c=df['qoe_score'], cmap='RdYlGn',
                           alpha=0.6, s=10, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='QoE Score')
    else:
        ax.scatter(df['longitude'], df['latitude'], alpha=0.6, s=10)
    
    ax.set_title('Geographic Distribution of Measurements', 
                fontweight='bold', fontsize=12)
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.grid(True, alpha=0.3)
    
    # Elevation analysis
    if 'elevation_m' in df.columns:
        ax = axes[1, 1]
        df['elevation_m'].hist(bins=50, ax=ax, color='tan', edgecolor='black')
        ax.set_title('Elevation Distribution', fontweight='bold', fontsize=12)
        ax.set_xlabel('Elevation (meters)')
        ax.set_ylabel('Frequency')
        ax.axvline(df['elevation_m'].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {df["elevation_m"].mean():.0f}m')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Geographic statistics
    print("\n📍 Geographic Statistics:")
    print("-" * 60)
    print(f"Latitude range: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
    print(f"Longitude range: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
    if 'elevation_m' in df.columns:
        print(f"Elevation range: {df['elevation_m'].min():.0f}m to {df['elevation_m'].max():.0f}m")
```

---

## Cell 9: Performance Metrics Analysis

```python
# Cell: Performance Metrics Deep Dive

print("="*80)
print("PERFORMANCE METRICS ANALYSIS")
print("="*80)

# QoE Score analysis
if 'qoe_score' in df.columns:
    print("\n🎯 QoE Score Analysis:")
    print("-" * 60)
    
    qoe_stats = df['qoe_score'].describe()
    print(qoe_stats)
    
    # QoE categories
    df['qoe_category'] = pd.cut(df['qoe_score'], 
                                bins=[0, 3, 6, 8, 10],
                                labels=['Poor', 'Fair', 'Good', 'Excellent'])
    
    print("\n📊 QoE Categories:")
    print(df['qoe_category'].value_counts().sort_index())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # QoE distribution
    ax = axes[0, 0]
    df['qoe_score'].hist(bins=50, ax=ax, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.axvline(df['qoe_score'].mean(), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {df["qoe_score"].mean():.2f}')
    ax.axvline(df['qoe_score'].median(), color='green', linestyle='--', 
              linewidth=2, label=f'Median: {df["qoe_score"].median():.2f}')
    ax.set_title('QoE Score Distribution', fontweight='bold')
    ax.set_xlabel('QoE Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # QoE categories pie chart
    ax = axes[0, 1]
    qoe_cat_counts = df['qoe_category'].value_counts()
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4ecdc4']
    ax.pie(qoe_cat_counts.values, labels=qoe_cat_counts.index, autopct='%1.1f%%',
          colors=colors, startangle=90)
    ax.set_title('QoE Categories Distribution', fontweight='bold')
    
    # Throughput analysis
    ax = axes[0, 2]
    if 'download_throughput_bps' in df.columns:
        df['download_throughput_mbps'] = df['download_throughput_bps'] / 1e6
        df['download_throughput_mbps'].hist(bins=50, ax=ax, 
                                           color='steelblue', edgecolor='black')
        ax.set_title('Download Throughput Distribution', fontweight='bold')
        ax.set_xlabel('Throughput (Mbps)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Signal loss analysis
    ax = axes[1, 0]
    if 'signal_loss_db' in df.columns:
        df['signal_loss_db'].hist(bins=50, ax=ax, color='orange', edgecolor='black')
        ax.set_title('Signal Loss Distribution', fontweight='bold')
        ax.set_xlabel('Signal Loss (dB)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Packet loss analysis
    ax = axes[1, 1]
    if 'packet_loss' in df.columns:
        df['packet_loss'].hist(bins=50, ax=ax, color='salmon', edgecolor='black')
        ax.set_title('Packet Loss Distribution', fontweight='bold')
        ax.set_xlabel('Packet Loss (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Satellite count analysis
    ax = axes[1, 2]
    if 'visible_satellites' in df.columns and 'serving_satellites' in df.columns:
        x = np.arange(2)
        means = [df['visible_satellites'].mean(), df['serving_satellites'].mean()]
        stds = [df['visible_satellites'].std(), df['serving_satellites'].std()]
        
        ax.bar(x, means, yerr=stds, capsize=10, color=['lightblue', 'lightgreen'],
              edgecolor='black', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['Visible\nSatellites', 'Serving\nSatellites'])
        ax.set_ylabel('Count')
        ax.set_title('Average Satellite Counts', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
```

---

## Cell 10: Machine Learning Preparation

```python
# Cell: Machine Learning Preparation

print("="*80)
print("MACHINE LEARNING PREPARATION")
print("="*80)

# Feature engineering suggestions
print("\n🔧 Feature Engineering Recommendations:")
print("-" * 60)

# 1. Check for skewness
print("\n1. Skewness Analysis (for transformation):")
skewed_features = []
for col in numerical_cols:
    skew = df[col].skew()
    if abs(skew) > 1:
        skewed_features.append((col, skew))
        print(f"   {col}: {skew:.2f} (highly skewed)")

if skewed_features:
    print("\n   💡 Recommendation: Apply log or Box-Cox transformation")
else:
    print("   ✅ No highly skewed features")

# 2. Outlier summary
print("\n2. Outlier Summary:")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    outlier_pct = len(outliers) / len(df) * 100
    
    if outlier_pct > 5:
        print(f"   {col}: {len(outliers):,} outliers ({outlier_pct:.2f}%)")

# 3. Feature scaling recommendations
print("\n3. Feature Scaling Recommendations:")
print("   Features with different scales:")
for col in numerical_cols:
    print(f"   {col}: [{df[col].min():.2e}, {df[col].max():.2e}]")

print("\n   💡 Recommendation: Apply StandardScaler or MinMaxScaler")

# 4. Categorical encoding
print("\n4. Categorical Encoding:")
for col in categorical_cols:
    n_unique = df[col].nunique()
    print(f"   {col}: {n_unique} unique values")
    if n_unique == 2:
        print(f"      → Use Label Encoding")
    elif n_unique <= 10:
        print(f"      → Use One-Hot Encoding")
    else:
        print(f"      → Use Target Encoding or Frequency Encoding")

# 5. Feature importance (if target exists)
if 'qoe_score' in df.columns:
    print("\n5. Feature Importance (Correlation with Target):")
    print("-" * 60)
    
    feature_importance = df[numerical_cols].corr()['qoe_score'].abs().sort_values(ascending=False)
    feature_importance = feature_importance.drop('qoe_score')
    
    print(feature_importance)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    feature_importance.plot(kind='barh', ax=ax, color='teal')
    ax.set_title('Feature Importance (Absolute Correlation with QoE)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Absolute Correlation')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

# 6. Train-test split recommendation
print("\n6. Train-Test Split Recommendation:")
print("-" * 60)
print(f"   Total samples: {len(df):,}")
print(f"   Recommended split: 80-20 (Train-Test)")
print(f"   Training samples: {int(len(df)*0.8):,}")
print(f"   Testing samples: {int(len(df)*0.2):,}")
print(f"\n   💡 Use stratified split if target is imbalanced")

# 7. Missing value strategy
print("\n7. Missing Value Strategy:")
if df.isnull().sum().sum() > 0:
    print("   Numerical features: Use median imputation")
    print("   Categorical features: Use mode imputation or 'Unknown' category")
else:
    print("   ✅ No missing values to handle")

# 8. Create sample preprocessed dataset
print("\n8. Sample Preprocessing Pipeline:")
print("-" * 60)

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Example preprocessing
df_ml = df.copy()

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    df_ml[f'{col}_encoded'] = le.fit_transform(df_ml[col])

# Scale numerical features
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(df_ml[numerical_cols])
df_ml_scaled = pd.DataFrame(numerical_scaled, 
                            columns=[f'{col}_scaled' for col in numerical_cols])

print("✅ Sample preprocessing complete")
print(f"   Original features: {len(df.columns)}")
print(f"   After encoding: {len(df_ml.columns)}")
print(f"   Scaled features: {len(df_ml_scaled.columns)}")

print("\n" + "="*80)
print("✅ EDA COMPLETE - READY FOR MACHINE LEARNING")
print("="*80)
```

---

## Cell 11: Final Summary and Recommendations

```python
# Cell: Final Summary and ML Recommendations

print("="*80)
print("FINAL SUMMARY & MACHINE LEARNING RECOMMENDATIONS")
print("="*80)

print("\n📊 DATASET SUMMARY:")
print("-" * 60)
print(f"✅ Total Records: {len(df):,}")
print(f"✅ Total Features: {len(df.columns)}")
print(f"✅ Numerical Features: {len(numerical_cols)}")
print(f"✅ Categorical Features: {len(categorical_cols)}")
print(f"✅ Missing Values: {df.isnull().sum().sum()}")
print(f"✅ Duplicate Rows: {df.duplicated().sum()}")

print("\n🎯 TARGET VARIABLE (QoE Score):")
print("-" * 60)
if 'qoe_score' in df.columns:
    print(f"Mean: {df['qoe_score'].mean():.2f}")
    print(f"Median: {df['qoe_score'].median():.2f}")
    print(f"Std Dev: {df['qoe_score'].std():.2f}")
    print(f"Range: [{df['qoe_score'].min():.2f}, {df['qoe_score'].max():.2f}]")

print("\n🔬 RECOMMENDED ML MODELS:")
print("-" * 60)
print("1. Regression Models (for QoE prediction):")
print("   • Linear Regression (baseline)")
print("   • Random Forest Regressor")
print("   • Gradient Boosting (XGBoost, LightGBM)")
print("   • Neural Networks (MLP)")
print("\n2. Classification Models (for QoE categories):")
print("   • Logistic Regression")
print("   • Random Forest Classifier")
print("   • Support Vector Machines")
print("   • Deep Learning (CNN/LSTM for time series)")

print("\n🛠️ PREPROCESSING STEPS:")
print("-" * 60)
print("1. Handle outliers (if needed)")
print("2. Scale numerical features (StandardScaler)")
print("3. Encode categorical features (One-Hot or Label Encoding)")
print("4. Feature selection (Remove low-importance features)")
print("5. Train-test split (80-20, stratified)")
print("6. Cross-validation (5-fold)")

print("\n📈 FEATURE ENGINEERING IDEAS:")
print("-" * 60)
print("• Throughput ratio: download/upload")
print("• Satellite efficiency: serving/visible")
print("• Signal quality index: combine signal_loss and packet_loss")
print("• Geographic clusters: group by lat/long regions")
print("• Weather severity score: encode weather conditions")
print("• Time-based features: if timestamps available")

print("\n⚠️  IMPORTANT CONSIDERATIONS:")
print("-" * 60)
print("• Check for data leakage")
print("• Validate model on unseen data")
print("• Monitor for overfitting")
print("• Consider ensemble methods")
print("• Track model performance metrics (RMSE, MAE, R²)")

print("\n" + "="*80)
print("🎉 EXPLORATORY DATA ANALYSIS COMPLETE!")
print("="*80)
print("\nDataset is ready for machine learning modeling.")
print("Next steps: Feature engineering → Model training → Evaluation")
```

---

## Summary

These 11 cells provide a **comprehensive EDA** covering:

1. ✅ Data loading and overview
2. ✅ Data quality assessment
3. ✅ Statistical summaries
4. ✅ Distribution analysis
5. ✅ Correlation analysis
6. ✅ Feature relationships
7. ✅ Categorical analysis
8. ✅ Geographic analysis
9. ✅ Performance metrics
10. ✅ ML preparation
11. ✅ Final recommendations

**Add these cells at the end of your notebook for complete analysis!** 📊🚀
