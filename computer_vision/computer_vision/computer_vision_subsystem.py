import cv2
import numpy as np
import mediapipe as mp
from hsemotion.facial_emotions import HSEmotionRecognizer
from .gaze_tracking.gaze_tracking import GazeTracking
from .clothes import ClothesEvaluator
from .emotions import VideoEmotions
from .gestures import Gestures


class ComputerVisionSubsystem:
    """
    Class for analysing non-verbal components of the video.
    """
    device = 'cpu'
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    model_name = 'enet_b0_8_best_afew'
    fer = HSEmotionRecognizer(model_name=model_name, device=device)
    face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.3)

    def __init__(self, video_path, seconds=1):
        """
        Initializes video.
        @param video_path: path of the vidio that will be analysed.
        @param seconds: time between frames that will be processed.
        """
        self.video_path = video_path
        self.frames = self.get_frames(seconds)

    def get_frames(self, seconds):
        """
        Reads video frames.
        @param seconds: time between read frames.
        @return: array with frames for processing.
        """
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        count = 0
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        while count * seconds * fps < video_length:
            cap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000 * seconds))
            count += 1
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return frames

    def uncorrect_angle(self):
        """
        Count proportion of frames with incorrect angle of speaker.
        @return: proportion of incorrect frames.
        """
        imgs = self.frames
        width = imgs[0].shape[1]
        height = imgs[0].shape[0]
        results = list(map(ComputerVisionSubsystem.pose.process, imgs))
        count = 0
        correct = 0
        for result in results:
            if result.pose_landmarks:
                # Get head coordinates.
                head_landmark = result.pose_landmarks.landmark[0]
                head_x = head_landmark.x * width
                head_y = head_landmark.y * height
                count += 1
                # Check head position.
                if 0.2 * height <= head_y <= 0.6 * height and \
                        0.2 * width <= head_x <= 0.8 * width:
                    correct += 1
        return 1 - correct / count

    def eye_tracking(self):
        """
        Count proportion of frames with incorrect eye direction of speaker.
        @return: proportion of incorrect frames.
        """
        gaze = GazeTracking()
        central_direction = 0
        for img in self.frames:
            gaze.refresh(img)
            if gaze.is_center():
                central_direction += 1
        return 1 - central_direction / len(self.frames)

    def get_gestures(self):
        """
        Count velocity of speaker movement.
        @return: velocity.
        """
        result = Gestures.count_mean_velocity(self.frames)
        if result <= 15:
            return 0
        elif result <= 30:
            return 1
        return 2

    def get_emotions_scores(self):
        """
        Predict speaker emotions.
        @return: array of probabilities of 6 emotions for every frame.
        """
        scores = VideoEmotions.process_frame_every_n(self.frames, 1)
        modified_list = None
        if scores is not None:
            modified_list = [np.concatenate((inner_list[:4], inner_list[5:-1]), axis=0) for inner_list in scores]
        return modified_list

    def get_clothes_estimation(self):
        """
        Estimate if clothes is appropriate for performance.
        @return: boolean value if clothes is appropriate.
        """
        return ClothesEvaluator.is_outfit_appropriate(self.frames[0])
