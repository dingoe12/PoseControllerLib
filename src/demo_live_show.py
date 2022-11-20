import mediapipe as mp
from basic_detection.complete_pose_detector import CompletePoseDetection
import cv2
import argparse


def main(source, complexity=2, point_min_conf=0.5, det_min_conf=0.5):
    """Main function for the demo,
    which uses the complete pose detector to detect poses in the provided video source and shows them.
    Pressing q will exit the demo."""
    detector = CompletePoseDetection(source, complexity, point_min_conf, det_min_conf)
    for image, results in detector:
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Pose Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for the complete pose detector.")
    parser.add_argument("--source", type=str, default="0", help="The source of the video.")
    parser.add_argument("--complexity", type=int, default=2, help="The model complexity of the pose detector.")
    parser.add_argument("--point-min-conf", type=float, default=0.5,
                        help="The minimum confidence for a point to be detected.")
    parser.add_argument("--det-min-conf", type=float, default=0.5,
                        help="The minimum confidence for a detection to be detected.")
    args = parser.parse_args()
    main(args.source, args.complexity, args.point_min_conf, args.det_min_conf)
