import logging
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def log_printer():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(formatter)
    return console_handler


def scaleImage(uimg):
    scaled_comb = uimg 
    a_min = scaled_comb.min()
    a_max = scaled_comb.max()
    a_scaled = (scaled_comb - a_min) / (a_max - a_min)
    return a_scaled

def getPCAVisual(emb, channels=3):
    """
    Perform PCA on the embedding and return an image output in RGB format.
    
    Parameters:
    emb (np.ndarray): The input embedding of shape (H, W, D).
    channels (int): The number of channels for the output image (default is 3).
    
    Returns:
    np.ndarray: The output image of shape (H, W, channels).
    """
    # Check if the embedding has the correct shape
    if len(emb.shape) != 3:
        raise ValueError("The input embedding must have 3 dimensions (H, W, D).")
    
    H, W, D = emb.shape
    
    # Flatten the embedding to a 2D array (H*W, D)
    emb_flattened = emb.reshape(-1, D)
    
    # Perform PCA to reduce the dimensionality to 'channels' components
    pca = PCA(n_components=channels)
    emb_pca = pca.fit_transform(emb_flattened)
    
    # Normalize the PCA output to the range [0, 1]
    scaler = MinMaxScaler()
    emb_pca_normalized = scaler.fit_transform(emb_pca)
    
    # Reshape the result back to the original spatial dimensions with 'channels' channels
    emb_pca_image = emb_pca_normalized.reshape(H, W, channels)
    
    return emb_pca_image

def get_binary_image(output_image):
    """
    Converts a 2-class output image of shape (2, H, W) into a binary image of shape (H, W).

    Args:
        output_image (numpy.ndarray): Array of shape (2, H, W) with values representing the output
                                      for each class at each pixel.

    Returns:
        numpy.ndarray: Binary image of shape (H, W) with values 0 or 1.
    """
    if output_image.shape[0] != 2:
        raise ValueError("Input image must have shape (2, H, W), with two classes.")

    # Get the class with the maximum value at each pixel
    binary_image = np.argmax(output_image, axis=0)  # Shape: (H, W)

    return binary_image


def save_metrics(exp_dir, mode, epoch, print_metrics):
    """
    Args:
        exp_dir (str): Directory where 'metrics.txt' should be saved.
        mode (str): Training or validation mode.
        epoch (int): Current training epoch.
        print_metrics (dict or str): Metrics to be written.
    """

    # If `print_metrics` is a dictionary, turn it into a string or JSON
    if isinstance(print_metrics, dict):
        import json
        metrics_str =  str(print_metrics)
    else:
        # Otherwise assume it's already a string
        metrics_str = str(print_metrics)

    file_path = os.path.join(exp_dir, "metrics.txt")

    # Safely open + append
    with open(file_path, "a") as file_h:
        file_h.write(f"Epoch {epoch} - {mode} Metrics: {metrics_str}\n")
    
    #print(f"Metrics saved at '{file_path}'")
