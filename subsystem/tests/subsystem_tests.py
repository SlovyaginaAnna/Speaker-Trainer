import unittest

import cv2
import hypothesis
from hypothesis import given
from hypothesis import strategies as st
from subsystem import VideoSubsystem


class SubsystemTestCase(unittest.TestCase):
    def get_frames(self, inp):
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
        return frames

    @given(st.lists(st.floats(0, 1), min_size=10, max_size=10), st.integers(1, 10), st.integers(0, 10))
    def test_get_subarray(self, arr, num1, num2):
        result = VideoSubsystem.get_subarray(arr, num1, num2)
        self.assertTrue(len(result) <= num1)

    @given(st.lists(st.integers(0, 5), min_size=1))
    def test_calculate_percentage(self, inp):
        result = VideoSubsystem.calculate_percentage(inp)
        self.assertIsInstance(result, dict)

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_process_emotions(self, inp):
        frames = self.get_frames(inp)
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        system = VideoSubsystem(path, [])
        res1, res2 = system.process_emotions(frames)
        self.assertTrue(0 <= res1 <= 1)
        self.assertTrue(0 <= res2 <= 1)

    # TO DO
    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_process_gesticulation(self, inp):
        frames = self.get_frames(inp)
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        system = VideoSubsystem(path, [])
        result = system.process_gesticulation(frames)
        self.assertIsNotNone(result)
        self.assertTrue(result in [0, 1, 2])

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_process_gaze(self, inp):
        frames = self.get_frames(inp)
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        system = VideoSubsystem(path, [])
        result = system.process_gaze(frames)
        self.assertTrue(0 <= result <= 1)

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_process_angle(self, inp):
        frames = self.get_frames(inp)
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        system = VideoSubsystem(path, [])
        result = system.process_angle(frames)
        self.assertTrue(len(result) == 4)

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_process_clothes(self, inp):
        frames = self.get_frames(inp)
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        system = VideoSubsystem(path, [])
        result = system.process_clothes(frames)
        self.assertIsNotNone(result)

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 10)))
    def test_process_video(self, inp):
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        system = VideoSubsystem(path, [])
        system.process_video(3)
        self.assertTrue(len(system.get_gaze()) > 0)
        self.assertTrue(len(system.get_angle()) > 0)
        self.assertTrue(len(system.get_angle_len()) > 0)
        self.assertTrue(len(system.get_emotions()) > 0)
        self.assertTrue(len(system.get_gestures()) > 0)
        self.assertTrue(len(system.get_inappropriate_emotion_percentage()) > 0)
        self.assertTrue(len(system.get_incorrect_angle_ind()) > 0)
        self.assertTrue(len(system.get_lightning()) > 0)
        self.assertIsNotNone(system.get_clothes_estimation())


if __name__ == '__main__':
    unittest.main()
