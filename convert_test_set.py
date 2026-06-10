import os
import glob
import pandas as pd
import numpy as np

def main():
    csv_files = glob.glob('test_set/**/*.csv', recursive=True)
    if not csv_files:
        print("No CSV files found in test_set/")
        return
        
    print(f"Found {len(csv_files)} CSV files. Converting to .npy...")
    for i, f in enumerate(csv_files):
        print(f"[{i+1}/{len(csv_files)}] Converting {f}...")
        try:
            arr = pd.read_csv(f, header=None).values.astype(np.float32)
            npy_path = f.replace('.csv', '.npy')
            np.save(npy_path, arr)
            os.remove(f)  # delete original to save space
        except Exception as e:
            print(f"  Error converting {f}: {e}")
            
    print("Done! All test files are now in fast .npy format.")

if __name__ == "__main__":
    main()
