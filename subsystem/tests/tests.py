import math
import unittest
import math
import cv2
import hypothesis
from hypothesis import given, strategies as st

from clothes import ClothesEvaluator
from emotions import VideoEmotions
from gestures import Gestures
from main import ComputerVisionSubsystem
import numpy as np

class MyTestCase(unittest.TestCase):

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_get_frames(self, inp):
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        cap = cv2.VideoCapture(path)
        cap.release()
        test_subsystem = ComputerVisionSubsystem(path)
        self.assertTrue(len(test_subsystem.frames) > 0)

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_angle_instance(self, inp):
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        test_subsystem = ComputerVisionSubsystem(path)
        result = test_subsystem.uncorrect_angle()
        self.assertIsInstance(result, float, "Значение не является типом float")
        self.assertTrue(0 <= result <= 1, "Значение находится вне диапазона от 0 до 1")

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_glance_instance(self, inp):
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        test_subsystem = ComputerVisionSubsystem(path)
        result = test_subsystem.eye_tracking()
        self.assertIsInstance(result, float, "Значение не является типом float")
        self.assertTrue(0 <= result <= 1, "Значение находится вне диапазона от 0 до 1")

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_count_velocity(self, inp):
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        test_subsystem = ComputerVisionSubsystem(path)
        result = Gestures.count_mean_velocity(test_subsystem.frames)
        self.assertIsInstance(result, float, "Значение не является типом float")

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_get_emotions(self, inp):
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        test_subsystem = ComputerVisionSubsystem(path)
        scores = VideoEmotions.process_frame_every_n(test_subsystem.frames, 1)
        if (scores is not None):
            for score in scores:
                self.assertTrue(len(score) == 8, "Длина списка неверна")

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_get_modified_emotions(self, inp):
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        test_subsystem = ComputerVisionSubsystem(path)
        scores = test_subsystem.get_emotions_scores()
        if scores is not None:
            for score in scores:
                self.assertTrue(len(score) == 6, "Длина списка неверна")

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(["woman.jpeg", "face.jpg", "main_face.jpg"]))
    def test_detect_and_crop_with_image(self, image):
        img = cv2.imread(f'C:\\Users\\Operator\\Pictures\\{image}')
        cropped_frame = ClothesEvaluator.detect_and_crop(img)
        self.assertIsNotNone(cropped_frame)

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(["woman.jpeg", "face.jpg"]))
    def test_get_main_face(self, image):
        frame = cv2.imread(f'C:\\Users\\Operator\\Pictures\\{image}')
        main_face = VideoEmotions.get_main_face()
        self.assertIsNotNone(main_face)

if __name__ == '__main__':
    unittest.main()
