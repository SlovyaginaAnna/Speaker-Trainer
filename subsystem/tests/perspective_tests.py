import unittest
import hypothesis
from hypothesis import given
from hypothesis import strategies as st
import cv2
from perspective import Perspective

class PerspectiveTestCase(unittest.TestCase):
    def setUp(self):
        self.perspective_instance = Perspective()

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

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_count_brightness(self, inp):
        frames = self.get_frames(inp)
        result = self.perspective_instance.count_brightness(frames)
        self.assertIsNotNone(result)
        self.assertTrue(result in [0, 1, 2])

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_count_angle(self, inp):
        frames = self.get_frames(inp)
        result = self.perspective_instance.count_angle(frames)
        self.assertIsNotNone(result)
        self.assertTrue(0 <= result[0] <= 1)
        self.assertTrue(len(result) == 3)

if __name__ == '__main__':
    unittest.main()
