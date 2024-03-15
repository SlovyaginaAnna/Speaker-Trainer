import mediapipe as mp
from hsemotion.facial_emotions import HSEmotionRecognizer

class VideoEmotions:
    """
    Class estimates human emotions on frames.
    """
    face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.3)
    device = 'cpu'
    model_name = 'enet_b0_8_best_afew'
    fer = HSEmotionRecognizer(model_name=model_name, device=device)

    # Функция возвращает обрезанное изображение лица основного человека на изображении
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

    def process_frame_every_n(frames, n):
        """
        Process frames and get emotion probabilities.
        @param n: how many frames processing takes place.
        @return: probabilities of emotions.
        """
        imgs = frames[::n]
        scores = None
        faces = list(map(VideoEmotions.get_main_face, imgs))
        non_empty_faces = []
        for face in faces:
            if face is not None:
                non_empty_faces.append(face)
        if (len(non_empty_faces) > 0):
            emotions, scores = VideoEmotions.fer.predict_multi_emotions(non_empty_faces, logits=False)
        return scores
