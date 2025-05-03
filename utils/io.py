# main.py (or utils/io.py)
import os
from typing import List, Tuple

def load_file_list(
    file_path: str,
    audio_dir:  str,
    landmark_dir: str,
    strict: bool = False  # New parameter to control strictness
) -> Tuple[List[str], List[str]]:
    mfcc_paths, landmark_paths = [], []
    missing = 0
    total = 0

    with open(file_path, "r") as f:
        for line in f:
            total += 1
            base = line.strip()
            mfcc = os.path.join(audio_dir, base + ".npy")
            lmk  = os.path.join(landmark_dir, base + ".npy")

            if os.path.isfile(mfcc) and os.path.isfile(lmk):
                mfcc_paths.append(mfcc)
                landmark_paths.append(lmk)
            else:
                missing += 1
                print(f"⚠️  Missing for `{base}`:", 
                      f"mfcc={os.path.isfile(mfcc)},", 
                      f"landmark={os.path.isfile(lmk)}")

    if missing > 0:
        missing_percentage = (missing / total) * 100
        warning_message = f"{missing}/{total} entries missing ({missing_percentage:.1f}%)! Check your train.txt vs folders."
        
        if strict or missing == total:
            # If strict mode or ALL files are missing, raise error
            raise RuntimeError(warning_message)
        else:
            # Otherwise just print a warning and continue with available files
            print(f"⚠️  Warning: {warning_message}")
            print(f"✅ Continuing with {len(mfcc_paths)} available files.")
    
    return mfcc_paths, landmark_paths
