"""Utilities for checking if running on Google Colab and setting up the environment accordingly."""

from __future__ import annotations


def is_running_on_colab() -> bool:
    """
    Checks if the code is running on Google Colab.
    
    Returns:
        bool: True if running on Google Colab, False otherwise.
    """
    try:
        if 'google.colab' in str(get_ipython()): # type: ignore
            return True
        else:
            return False
    except NameError:
        return False
    

def setup_environment():
    """
    Sets up the environment based on the running platform.
    
    Returns:
        str: The root path for the project.
    """
    # If running on Google Colab, set up the environment and the root path to use Google Drive for data storage and processing
    if is_running_on_colab():
        import tensorflow as tf
        print("Running on Google Colab")

        # Check for available GPUs and, if found, configure TensorFlow to use memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            # Ensure at least one GPU is available
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("GPU memory growth enabled.")
            else:
                print("No GPU devices found.")
        except Exception as e:
            print(f"Error setting GPU memory growth: {e}") # type: ignore
            pass

        # Get GPU infos
        import shutil
        import subprocess

        if shutil.which("nvidia-smi") is None:
            print("nvidia-smi not found (likely no NVIDIA GPU runtime).")
        else:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                msg = (result.stderr or result.stdout).strip()
                print(f"nvidia-smi failed: {msg}" if msg else "nvidia-smi failed")
            else:
                print(result.stdout)

        # Connect to Google Drive
        from google.colab import drive # type: ignore
        drive.mount('/content/drive')

        # Setup the Google root
        root_path = "./drive/MyDrive/Project/Python_parallelism"

        print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    else:
        print("Running on a local machine")
        # Setup the local root
        root_path = ".."
        print("TensorFlow will be imported in Section 3 (after multiprocessing benchmarks).")
        
    return root_path
