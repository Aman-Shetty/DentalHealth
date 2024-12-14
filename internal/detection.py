import torch

def detect_disease(image, model):
    # Perform inference on the input image
    results = model(image)
    return results.xyxy[0].cpu().numpy()  # Get detection results as a numpy array
