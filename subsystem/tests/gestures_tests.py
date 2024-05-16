import unittest
import cv2
import hypothesis
from hypothesis import given
from hypothesis import strategies as st
from gestures import Gestures

class TestGestures(unittest.TestCase):
    def setUp(self):
        self.gesture_instance = Gestures()

    @given(
        st.lists(st.floats(0, 1), min_size=2, max_size=2),
        st.lists(st.floats(0, 1), min_size=2, max_size=2)
    )
    def test_get_vector_between_points(self, v1, v2):
        res = self.gesture_instance.get_vector_between_points(v1, v2)
        self.assertIsNotNone(res)
        self.assertTrue(len(res) == 2)

    @given(
        st.lists(st.floats(0.1, 1), min_size=2, max_size=2),
        st.lists(st.floats(0.1, 1), min_size=2, max_size=2)
    )
    def test_angle_between_vectors(self, v1, v2):
        res = self.gesture_instance.angle_between_vectors(v1, v2)
        self.assertIsNotNone(res)
        self.assertTrue(0 <= res <= 360)

    @given(st.floats(0, 360), st.floats(0, 360))
    def test_min_angle_difference(self, ang1, ang2):
        res = self.gesture_instance.min_angle_difference(ang1, ang2)
        self.assertIsInstance(res, float)
        self.assertTrue(res <= 180)

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 3)))
    def test_process_velocity(self, inp):
        path = f"C:\\Users\\Operator\\Desktop\\Project\\video\\{inp}.mp4"
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()
            cv2.destroyAllWindows()
        try:
            self.gesture_instance.process_velocity(frames)
        except Exception as e:
            self.fail(f"Method raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()