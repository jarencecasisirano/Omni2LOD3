#!/usr/bin/env python3
"""
Copy Pipeline A's clipped file to match the EXACT path of the working test
This will determine if it's a path/location issue
"""
import shutil
import os

source = r"C:\Projects\Omni2LOD3\outputs\02_clipped\NIMBB 112025_05_clipped.las"
dest = r"C:\Projects\Omni2LOD3\outputs\00_archive\03_pipeline_A_copied.las"

print("="*60)
print("PATH/LOCATION TEST")
print("="*60)
print(f"\nCopying Pipeline A's file to 00_archive directory:")
print(f"  FROM: {source}")
print(f"  TO:   {dest}")

try:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(source, dest)
    
    print(f"\n✓ File copied successfully!")
    print(f"\nNow try this in CityForge:")
    print(f"  {dest}")
    print(f"\nThis is Pipeline A's exact file, just in a different location.")
    print(f"")
    print(f"Expected result:")
    print(f"  - If this WORKS: The issue is the file path or directory")
    print(f"  - If this FAILS: The issue is in Pipeline A's file itself")
    
except Exception as e:
    print(f"\n✗ Error: {e}")

print(f"\n" + "="*60)
print("SUMMARY OF FINDINGS SO FAR")
print("="*60)
print(f"\n1. test_exact_B_process.py created a file that WORKS")
print(f"   Location: C:\\Projects\\Omni2LOD3\\outputs\\00_archive\\02_test_B_process_clipped.las")
print(f"")
print(f"2. Pipeline A creates a file that DOESN'T WORK")
print(f"   Location: C:\\Projects\\Omni2LOD3\\outputs\\02_clipped\\NIMBB 112025_05_clipped.las")
print(f"")
print(f"3. Deep inspection shows files are IDENTICAL")
print(f"")
print(f"4. The only code difference is:")
print(f"   - Pipeline B uses: from progress.bar import ChargingBar")
print(f"   - Pipeline A uses: from utils.loading import create_bar")
print(f"")
print(f"NEXT STEP: Check what create_bar() actually does!")