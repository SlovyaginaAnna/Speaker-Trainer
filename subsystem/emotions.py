import mediapipe as mp
from hsemotion.facial_emotions import HSEmotionRecognizer
import cv2


class VideoEmotions:
    face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.3)
    model_name = 'enet_b0_8_best_afew'
    face_mesh = mp.solutions.face_mesh.FaceMesh()

    def __init__(self, device='cpu', model='HSEmotion'):
        """
        Initialize model and device.
        :param device: cpu or gpu.
        :param model: HSEmotion or deepFace.
        """
        self.device = device
        self.model = model
        if model == 'HSEmotion':
            self.predictor = HSEmotionRecognizer(model_name=VideoEmotions.model_name, device=device)

    @staticmethod
    def get_main_face(frame):
        """
        Crop image to get main face on the frame.
        @return: cropped image.
        """
        results = VideoEmotions.face_detection.process(frame)
        main_face = None
        max_score = 0
        if results.detections is not None:
            for detection in results.detections:
                if detection.score[0] > max_score:
                    main_face = detection
                    max_score = detection.score[0]
            if main_face is not None:
                bbox = main_face.location_data.relative_bounding_box
                image_height, image_width, _ = frame.shape
                x, y, w, h = int(bbox.xmin * image_width), int(bbox.ymin * image_height), \
                    int(bbox.width * image_width), int(bbox.height * image_height)
                main_face = frame[y:y + h, x:x + w]
        return main_face

    @staticmethod
    def calculate_emotion_percentage(emotion_list):
        """
        Calculate percentage of each element in the list.
        :param emotion_list: list for calculation.
        :return: dictionary with percentages of each element.
        """
        total_frames = len(emotion_list)
        emotion_percentage = {}
        for emotion in emotion_list:
            if emotion in emotion_percentage.keys():
                emotion_percentage[emotion] += 1
            else:
                emotion_percentage[emotion] = 1
        for emotion in emotion_percentage.keys():
            emotion_percentage[emotion] = (emotion_percentage[emotion] / total_frames) * 100
        return emotion_percentage

    @staticmethod
    def calculate_emotion_change_frequency(emotion_list):
        """
        calculate the percentage of changing emotions between two seconds.
        :param emotion_list: list to calculate changes in it.
        :return: frequency of changing emotions.
        """
        total_frames = len(emotion_list)
        emotion_changes = 0
        for i in range(1, total_frames):
            if emotion_list[i] != emotion_list[i - 1]:
                emotion_changes += 1
        emotion_change_frequency = emotion_changes / total_frames
        return emotion_change_frequency

    def process_frames(self, frames):
        """
        Predict emotions on each frame.
        :param frames: frames for processing.
        :return: main emotions and probabilities for each frame.
        """
        imgs = frames
        faces = list(map(VideoEmotions.get_main_face, imgs))
        emotions, scores = [], []
        for face in faces:
            if face is not None:
                try:
                    emotion, score = self.predictor.predict_emotions(face, logits=False)
                    emotions.append(emotion)
                    scores.append(score)
                except Exception:
                    continue
        return emotions, scores
