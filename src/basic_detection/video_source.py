import cv2
from pathlib import Path


class VideoSource:
    """An abstraction for the video source, which can be either a webcam or a video file"""

    def __init__(self, source):
        """Initializes the video source based on the type of source provided."""
        if source is None:
            self.video = cv2.VideoCapture(0)
            self.webcam = True
        elif isinstance(source, int):
            self.video = cv2.VideoCapture(source)
            self.webcam = True
        elif isinstance(source, str):
            self.video = cv2.VideoCapture(source)
            self.webcam = False
        elif isinstance(source, Path) & source.is_file():
            self.video = cv2.VideoCapture(str(source))
            self.webcam = False
        else:
            raise ValueError('Invalid source')

    def __iter__(self):
        """Returns the generator object, which is used to iterate over the frames of the video."""
        vidcap = self.video
        while vidcap.isOpened():
            success, frame = vidcap.read()
            if success:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # If webcam is used, then we simply need to wait for the next frame, as it might not yet be ready
            elif self.webcam:
                continue
            # Otherwise we need to break when no more frames are found, as we have found the end of the video file
            else:
                break
