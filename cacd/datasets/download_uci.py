"""
Download and prepare UCI regression datasets for uncertainty quantification.

We'll download:
1. Concrete Compressive Strength
2. Energy Efficiency
3. Wine Quality (Red)
4. Yacht Hydrodynamics
5. Power Plant

These are standard benchmarks used in UQ literature.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import os
from pathlib import Path

# Set base path
BASE_PATH = Path('/ssd_4TB/divake/temporal_uncertainty/cacd/datasets')
BASE_PATH.mkdir(exist_ok=True, parents=True)

print("="*70)
print("DOWNLOADING UCI DATASETS FOR UNCERTAINTY QUANTIFICATION")
print("="*70)

# ============================================================
# Dataset 1: Concrete Compressive Strength
# ============================================================
print("\n1. Concrete Compressive Strength")
print("-" * 50)
try:
    # Try OpenML
    concrete = fetch_openml(name='concrete', version=1, as_frame=True, parser='auto')
    df_concrete = concrete.frame
    print(f"   ✅ Downloaded from OpenML")
    print(f"   Shape: {df_concrete.shape}")
    print(f"   Features: {list(df_concrete.columns[:-1])}")
    print(f"   Target: {df_concrete.columns[-1]}")

    # Save
    df_concrete.to_csv(BASE_PATH / 'concrete.csv', index=False)
    print(f"   💾 Saved to concrete.csv")

except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   Will try manual download from UCI repository...")

    # Fallback: Download from UCI directly
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    try:
        df_concrete = pd.read_excel(url)
        df_concrete.to_csv(BASE_PATH / 'concrete.csv', index=False)
        print(f"   ✅ Downloaded from UCI repository")
        print(f"   Shape: {df_concrete.shape}")
    except Exception as e2:
        print(f"   ❌ Manual download also failed: {e2}")
        df_concrete = None

# ============================================================
# Dataset 2: Energy Efficiency
# ============================================================
print("\n2. Energy Efficiency")
print("-" * 50)
try:
    energy = fetch_openml(name='energy-efficiency', version=1, as_frame=True, parser='auto')
    df_energy = energy.frame
    print(f"   ✅ Downloaded from OpenML")
    print(f"   Shape: {df_energy.shape}")
    print(f"   Features: {list(df_energy.columns[:-2])}")
    print(f"   Targets: Heating Load, Cooling Load")

    # Save both targets separately
    df_energy_heating = df_energy.iloc[:, :-1]  # All features + heating
    df_energy_heating.to_csv(BASE_PATH / 'energy_heating.csv', index=False)

    df_energy_cooling = df_energy.copy()
    df_energy_cooling.to_csv(BASE_PATH / 'energy_cooling.csv', index=False)

    print(f"   💾 Saved to energy_heating.csv and energy_cooling.csv")

except Exception as e:
    print(f"   ❌ Error: {e}")
    df_energy = None

# ============================================================
# Dataset 3: Wine Quality (Red)
# ============================================================
print("\n3. Wine Quality (Red)")
print("-" * 50)
try:
    wine = fetch_openml(name='wine-quality-red', version=1, as_frame=True, parser='auto')
    df_wine = wine.frame
    print(f"   ✅ Downloaded from OpenML")
    print(f"   Shape: {df_wine.shape}")
    print(f"   Features: {list(df_wine.columns[:-1])}")
    print(f"   Target: quality")

    df_wine.to_csv(BASE_PATH / 'wine_quality_red.csv', index=False)
    print(f"   💾 Saved to wine_quality_red.csv")

except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   Trying alternative name...")
    try:
        # Try direct URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df_wine = pd.read_csv(url, sep=';')
        df_wine.to_csv(BASE_PATH / 'wine_quality_red.csv', index=False)
        print(f"   ✅ Downloaded from UCI repository")
        print(f"   Shape: {df_wine.shape}")
    except Exception as e2:
        print(f"   ❌ Both attempts failed: {e2}")
        df_wine = None

# ============================================================
# Dataset 4: Yacht Hydrodynamics
# ============================================================
print("\n4. Yacht Hydrodynamics")
print("-" * 50)
try:
    yacht = fetch_openml(name='yacht_hydrodynamics', version=1, as_frame=True, parser='auto')
    df_yacht = yacht.frame
    print(f"   ✅ Downloaded from OpenML")
    print(f"   Shape: {df_yacht.shape}")

    df_yacht.to_csv(BASE_PATH / 'yacht.csv', index=False)
    print(f"   💾 Saved to yacht.csv")

except Exception as e:
    print(f"   ❌ Error: {e}")
    df_yacht = None

# ============================================================
# Dataset 5: Power Plant
# ============================================================
print("\n5. Combined Cycle Power Plant")
print("-" * 50)
try:
    # Try alternative name
    power = fetch_openml(name='CCPP', version=1, as_frame=True, parser='auto')
    df_power = power.frame
    print(f"   ✅ Downloaded from OpenML")
    print(f"   Shape: {df_power.shape}")

    df_power.to_csv(BASE_PATH / 'power_plant.csv', index=False)
    print(f"   💾 Saved to power_plant.csv")

except Exception as e:
    print(f"   ❌ Error: {e}")
    # Try direct download
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
        import urllib.request
        import zipfile

        zip_path = BASE_PATH / 'CCPP.zip'
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_PATH)

        # Find and rename the extracted file
        for file in (BASE_PATH).glob('*.xlsx'):
            df_power = pd.read_excel(file)
            df_power.to_csv(BASE_PATH / 'power_plant.csv', index=False)
            file.unlink()  # Remove xlsx

        zip_path.unlink()  # Remove zip
        print(f"   ✅ Downloaded and extracted from UCI")
        print(f"   Shape: {df_power.shape}")

    except Exception as e2:
        print(f"   ❌ Both attempts failed: {e2}")
        df_power = None

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("DOWNLOAD SUMMARY")
print("="*70)

datasets_info = {
    'concrete.csv': 'Concrete Compressive Strength',
    'energy_heating.csv': 'Energy Efficiency (Heating)',
    'energy_cooling.csv': 'Energy Efficiency (Cooling)',
    'wine_quality_red.csv': 'Wine Quality (Red)',
    'yacht.csv': 'Yacht Hydrodynamics',
    'power_plant.csv': 'Power Plant'
}

available = []
for filename, name in datasets_info.items():
    path = BASE_PATH / filename
    if path.exists():
        size = path.stat().st_size / 1024  # KB
        available.append(filename)
        print(f"✅ {name:40s} ({size:.1f} KB)")
    else:
        print(f"❌ {name:40s} (FAILED)")

print(f"\n📊 Successfully downloaded: {len(available)}/{len(datasets_info)} datasets")
print(f"📁 Location: {BASE_PATH}")
print("="*70)
