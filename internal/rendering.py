import cv2
import numpy as np

def draw_boxes(image, detections, class_names):
    """
    Draw bounding boxes and labels on the image.
    :param image: Original image
    :param detections: Detections from the model (array of xyxy coordinates)
    :param class_names: Class names corresponding to the model
    :return: Image with bounding boxes drawn
    """
    # Draw each detection
    for *box, conf, cls in detections:
        # Convert bounding box coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        label = f'{class_names[int(cls)]}: {conf:.2f}'
        
        # Draw the rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # BGR color
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    return image
