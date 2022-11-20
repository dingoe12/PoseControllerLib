
class ICompletePoseDetector:
    """Interface for the complete pose detector, which provides no functionality,
    but is used to define the interface for the complete pose detector."""

    def __init__(self, source, model_complexity: int = 2, min_point_confidence: float = 0.5,
                 min_detection_confidence: float = 0.5):
        """Initializes the complete pose detector based on the video source provided using the provided parameters."""
        pass

    def detect(self, image):
        """Detects the pose in the provided image and returns the detection results as well as the original image."""
        pass

    def __iter__(self):
        """Returns the generator object, which is used to iterate over the poses of the video."""
        pass