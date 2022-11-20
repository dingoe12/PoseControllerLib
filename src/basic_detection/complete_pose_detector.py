from basic_detection import I_complete_pose_detector, video_source
import mediapipe as mp

class CompletePoseDetection(I_complete_pose_detector.ICompletePoseDetector):
    """An abstraction for the complete pose detector, which can be used to detect poses in a video."""

    def __init__(self, source, model_complexity: int = 2, min_point_confidence: float = 0.5,
                 min_detection_confidence: float = 0.5):
        """Initializes the complete pose detector based on the video source provided using the provided parameters."""
        super().__init__(source, model_complexity, min_point_confidence, min_detection_confidence)
        self.video = video_source.VideoSource(source)
        self.mpPose = mp.solutions.pose.Pose(model_complexity=model_complexity,
                                             min_detection_confidence=min_detection_confidence,
                                             min_tracking_confidence=min_point_confidence)

    def detect(self, image):
        """Detects the pose in the provided image and returns the detection results as well as the original image."""
        image.flags.writeable = False
        results = self.mpPose.process(image)
        image.flags.writeable = True
        return image, results

    def __iter__(self):
        """Returns the generator object, which is used to iterate over the poses of the video.
        Each element of the generator is a tuple containing the image and the pose detection results.
        The pose detection results are structured as a tuple of lists of the detected points,
        The first list (pose_landmarks) containing the two-dimensional results
        and the second list (pose_world_landmarks) containing the three-dimensional results
        with the origin being the midpoint between the hips, and the scale being in meters."""
        for im in self.video:
            yield self.detect(im)
