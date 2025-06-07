# Complementary module - download big data / extract features from pytorch tensors
This complementary module to the Flood-Modeling-HDL-FM repository aims to facilitate the download of the HDL-FM project data from Zenodo and the extraction of its features and targets into a feature store for reproducibility and ease of use.

## Contents
It contains 4 files:
- `download_extract_features.py` can be used to download the features from zenodo and extract them to a folder of choice.
- `config.py` contains the user inputs for `download_extract_features.py`.
- `extract_visualize_features.ipynb` allows the user to extract the features and visualize them in the process if the user has already downloaded the files from Zenodo manually.
- `requirements.txt` contains the python library requirements to run the python script or the jupyter notebook.
- `requirements_ipynb.txt` contains the python library requirements to use only the jupyter notebook.

## Use
There are various ways to use this module.

### Local: Python virtual environment

In your CLI:
1. Create a virtual environment (venv) to isolate dependencies:
    ```sh
    python -m venv venv
    ```

2. Activate the virtual environment:
    - On Windows: 
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux: 
        ```sh
        source venv/bin/activate
        ```

3. Install required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Navigate to this folder and run the extraction script with: 
    ```sh
    python download_extract_features.py
    ```


### Local: Conda virtual environment
In your CLI:
1. Create a new conda environment:
    ```sh
    conda create -n hdl_fm_extract python=3.10
    ```

2. Activate the environment:
    ```sh
    conda activate hdl_fm_extract
    ```

3. Install required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Navigate to this folder and run the extraction script:
    ```sh
    python download_extract_features.py
    ```


### Local: Docker container
A docker image has been built and published to Docker Hub under jllovell/HDL-FM_extract_features to ensure maximum reproducibility, so this module works even if there are issues with the library requirements on your local machine.

You can use the pre-built Docker image from Docker Hub by following the steps below.

In your CLI:
1. Navigate to this folder
2. Pull the image from Docker Hub:
    ```sh
    docker pull <your-dockerhub-username>/myhdl-extract-features:latest
    ```
3. Run the image with the extraction script:
    ```sh
    docker run --rm -v /path/to/your/code:/workspace -w /workspace <your-dockerhub-username>/myhdl-extract-features:latest download_extract_features.py
    ```


### Local: On your system (NOT recommended)
This is the quickest way but not at all recommended. Using a virtual environment or container ensures that dependencies required by this module do not interfere with other Python projects on your system.

In your CLI:
1. Install required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Navigate to this folder and run the extraction script with: 
    ```sh
    python download_extract_features.py
    ```


### Cloud: e.g. Google Colab (only for .ipynb file)
The .ipynb file can be uploaded to your Google Drive alongside the data downloaded from Zenodo. The notebook can then be opened in Google Colab and used with one of Google's runtimes to extract the features from the data. Bear in mind this will require at least 30GB of free space on your Google Drive.
