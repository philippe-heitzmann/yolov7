from typing import Optional
import requests
from PIL import Image
import numpy as np
import io

def download_image_convert_np(url: str) -> Optional[np.ndarray]:
    """
    Download an image from a URL and convert it to a numpy array.

    Args:
        url (str): The URL of the image.

    Returns:
        np.ndarray: The image as a numpy array.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching image: {e}")
        return None

    image = Image.open(io.BytesIO(response.content))
    image_np = np.array(image)

    return image_np
