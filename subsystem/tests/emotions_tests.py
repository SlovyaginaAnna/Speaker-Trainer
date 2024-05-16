import unittest
import random

import cv2
import hypothesis
import numpy as np
from PIL import Image
from hypothesis import given
from hypothesis import strategies as st
from emotions import VideoEmotions

class TestVideoEmotions(unittest.TestCase):
    def setUp(self):
        self.emotion_instance = VideoEmotions()

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(["woman.jpeg", "group.jpg", "face.jpg"]))
    def test_get_main_face(self, image):
        path = "C:\\Users\\Operator\\Pictures\\" + image
        image = Image.open(path)
        image = np.array(image)
        main_face = self.emotion_instance.get_main_face(image)
        self.assertIsNotNone(main_face, path)

    def test_calculate_emotion_change_frequency(self):
        emotions = ['Sadness', 'Disgust', 'Fear', 'Neutral', 'Happiness', 'Anger', 'Contempt']
        random_list = random.choices(emotions, k=10)
        self.assertIsInstance(self.emotion_instance.calculate_emotion_change_frequency(random_list), float)

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_process_frames(self, inp):
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        cap = cv2.VideoCapture(path)
        frame_count = 0
        frames = []
        try:
            while cap.isOpened() and frame_count < 5:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
        em, scores = self.emotion_instance.process_frames(frames)
        self.assertIsNotNone(em)
        self.assertIsNotNone(scores)

if __name__ == '__main__':
    unittest.main()
