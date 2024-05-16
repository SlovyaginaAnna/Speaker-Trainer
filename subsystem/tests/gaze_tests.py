import unittest
import cv2
import hypothesis
from hypothesis import given
from hypothesis import strategies as st
from gaze import GazeDirection

class TestGazeDirection(unittest.TestCase):
    def setUp(self):
        self.gaze_instance = GazeDirection()

    @given(st.floats(0, 1), st.floats(0, 1), st.floats(0, 1), st.floats(0, 1))
    def test_process_gaze(self, x1, y1, x2, y2):
        result = self.gaze_instance.process_gaze(x1, y1, x2, y2)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_gaze_detection(self, inp):
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
        result = self.gaze_instance.gaze_detection(frames)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)

if __name__ == '__main__':
    unittest.main()
