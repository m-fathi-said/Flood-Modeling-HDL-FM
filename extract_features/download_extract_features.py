"""
This script automatically downloads the datasets stored for this repository on Zenodo, extracts
the features and saves them in the 'data' folder, which will be automatically created if nonexistent.
WARNING: Due to the size of the data, you will need up to 10GB of memory while extracting and 
up to 15.5GB of disk space for storage or it will crash.
You will also need a fast and stable internet connection. Expect this script to take between 5-25mins.
"""

def extract_FML_data():
    """Wrapper function for extracting features from the FML dataset."""
    # ==== Import libraries ====
    # Global admin libraries
    import time
    start_time = time.time()
    import gc
    import traceback
    import requests
    from io import BytesIO
    from pathlib import Path
    from tqdm.auto import tqdm

    # Data manipulation and visualization libraries
    import numpy as np
    import torch

    # Get inputs from config file
    from config import (
        FSTORE_DIR, X_TRAIN_URL, X_TEST_URL,
        Y_TRAIN_URL, Y_TEST_URL, EXTRACT_X_TRAIN,
        EXTRACT_Y_TRAIN, EXTRACT_X_TEST,
        EXTRACT_Y_TEST, COMPRESSED, DOWNLOAD_CHUNK_SIZE
    )


    # ==== Global settings ====
    # Make path cross-platform compatible with pathlib
    fstore_dir = Path(FSTORE_DIR)

    # Create directory to save data if it doesn't exist
    if fstore_dir.exists():
        print(f"Directory for saving final dataset found: {fstore_dir}")
    else:
        fstore_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists, create if not
        print(f"WARNING: Directory for saving preprocessed features not found at {fstore_dir}. Created directory.")


    # ==== Download and extract features ====
    FEATURES = ["X_Qb", "X_elev", "X_h_lag", "X_h"]
    TARGETS  = ["Y_h", "Y_v", "Y_flowdir"]

    def downloadt_extractf(url, feature_names, prefix, fstore_dir, compressed, download_mbs=1):
        """Function to download tensor, extract and save features."""

        try:
            print(f'Downloading large {prefix} tensor from Zenodo')
            
            # Stream the response to download in chunks
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 1024 * 1024 * download_mbs  # multiple of 1MB chunks
            bytes_io = BytesIO()
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading {prefix}') as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        bytes_io.write(chunk)
                        pbar.update(len(chunk))
            bytes_io.seek(0)
            tensor = torch.load(bytes_io, map_location='cpu')
            print(f'{prefix} tensor occupies: {tensor.nbytes / (1024 ** 3):.2f} GB')
            print(f'Finished downloading {prefix} tensor. Extracting and saving features')
            for i, name in enumerate(feature_names):
                fname = f"{name}_{prefix}.npz" if compressed else f"{name}_{prefix}.npy"
                if compressed:
                    np.savez_compressed(fstore_dir / fname, **{f"{name}_{prefix}": tensor[:, i, :, :].numpy()})
                else:
                    np.save(fstore_dir / fname, tensor[:, i, :, :].numpy())
            del tensor, response
            gc.collect()
            print(f'Saved data to {fstore_dir} and cleared from memory.')
        except Exception as e:
            print(f"Error processing {prefix}: {e}")
            traceback.print_exc()

    if EXTRACT_X_TRAIN:
        downloadt_extractf(X_TRAIN_URL,
                           FEATURES,
                           "train",
                           fstore_dir,
                           COMPRESSED, 
                           DOWNLOAD_CHUNK_SIZE)
    if EXTRACT_Y_TRAIN:
        downloadt_extractf(Y_TRAIN_URL,
                           TARGETS,
                           "train",
                           fstore_dir,
                           COMPRESSED,
                           DOWNLOAD_CHUNK_SIZE)
    if EXTRACT_X_TEST:
        downloadt_extractf(X_TEST_URL,
                           FEATURES,
                           "test",
                           fstore_dir,
                           COMPRESSED,
                           DOWNLOAD_CHUNK_SIZE)
    if EXTRACT_Y_TEST:
        downloadt_extractf(Y_TEST_URL,
                           TARGETS,
                           "test",
                           fstore_dir,
                           COMPRESSED,
                           DOWNLOAD_CHUNK_SIZE)
    print('\nFinished downloading and extracting all features.')

    # ==== Time taken to run script ====
    end_time = time.time()
    print(f'This script took {(end_time - start_time)/60:.2f} minutes to run.')


# Run script only if called directly
if __name__ == "__main__":
    extract_FML_data()
