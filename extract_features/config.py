"""This file contains the user inputs for the script download_extract_features.py."""

# Path of directory where preprocessed features should be saved (this can be changed)
import os
FSTORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# URL from which to download the data
X_TRAIN_URL = "https://zenodo.org/records/15223719/files/train_x.pt?download=1"  # occupies ~6.6GB
Y_TRAIN_URL = "https://zenodo.org/records/15223719/files/train_y.pt?download=1"  # occupies ~5.0GB
X_TEST_URL  = "https://zenodo.org/records/15223719/files/test_x.pt?download=1"   # occupies ~2.6GB
Y_TEST_URL  = "https://zenodo.org/records/15223719/files/test_y.pt?download=1"   # occupies ~2.0GB

# Choose to leave out data e.g. if limited disk space
EXTRACT_X_TRAIN = True  # Set to False to skip extracting X_train features
EXTRACT_Y_TRAIN = True  # Set to False to skip extracting Y_train features
EXTRACT_X_TEST  = True  # Set to False to skip extracting X_test features
EXTRACT_Y_TEST  = True  # Set to False to skip extracting Y_test features

# Save features as compressed or uncompressed numpy arrays
COMPRESSED = True  # Set to True to save as compressed .npz files, False for uncompressed .npy files

# Specify the chunk size (in MB) for downloading data (higher => faster but needs more memory)
DOWNLOAD_CHUNK_SIZE = 4
