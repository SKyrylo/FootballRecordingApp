import cv2


def draw_bbox(frame, bbox, color):
    """
    Draws a bounding box on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the triangle.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the triangle in BGR format.

    Returns:
        numpy.ndarray: The frame with the triangle drawn on it.
    """
    x = int(bbox[0])
    y = int(bbox[1])
    x2 = int(x + int(bbox[2]))
    y2 = int(y + int(bbox[3]))
    
    return cv2.rectangle(frame, (x, y), (x2, y2), color, 2)