import numpy as np
from PIL import Image

def process_image(uploaded_file):
    # Convert the uploaded image to RGB format
    image = Image.open(uploaded_file).convert('RGB')
    return np.array(image)
