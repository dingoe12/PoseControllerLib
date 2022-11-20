import mediapipe as mp
import video_source


class PoseDetector:
    """An abstraction for the pose detector, which can be used to detect poses in a video."""

    def __init__(self, source: video_source.VideoSource, model_complexity: int = 2):
        """Initializes the pose detector based on the video source provided."""
        self.video = source
        self.mpPose = mp.solutions.pose.Pose(model_complexity=model_complexity)

    def __iter__(self):
        """Returns the generator object, which is used to iterate over the poses of the video."""
        for im in self.video:
            im.flags.writeable = False
            results = self.mpPose.process(im)
            im.flags.writeable = True
            yield im, results
