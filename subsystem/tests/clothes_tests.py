import unittest

import hypothesis
from hypothesis import given
from hypothesis import strategies as st
import numpy as np
from PIL import Image
import random
from clothes import Clothes
import cv2


class ClothesTestCase(unittest.TestCase):
    def setUp(self):
        self.clothes_instance = Clothes()

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(["woman.jpeg", "face.jpg", "main_face.jpg"]))
    def test_compute_image_sharpness(self, image):
        path = "C:\\Users\\Operator\\Pictures\\" + image
        image = Image.open(path)
        image = np.array(image)
        sharpness = self.clothes_instance.compute_image_sharpness(image)
        self.assertIsNotNone(sharpness)
        self.assertIsInstance(sharpness, float)

    def test_choose_sharpest_image(self):
        images = ["woman.jpeg", "face.jpg", "main_face.jpg"]
        path = "C:\\Users\\Operator\\Pictures\\"
        for i in range(len(images)):
            images[i] = path + images[i]
            images[i] = Image.open(images[i])
            images[i] = np.array(images[i])
        sharpest_image = self.clothes_instance.choose_sharpest_image(images)
        self.assertIsNotNone(sharpest_image)

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(["woman.jpeg", "face.jpg", "main_face.jpg"]))
    def test_transform_image(self, image):
        path = "C:\\Users\\Operator\\Pictures\\" + image
        image = Image.open(path)
        image = np.array(image)
        transformed_img = self.clothes_instance.transform_image(image)
        self.assertIsNotNone(transformed_img)


    def test_check_arrays(self):
        lower_bound = 0
        upper_bound = 100
        array_length = 10
        array1 = random.sample(range(lower_bound, upper_bound), array_length)
        array2 = random.sample(set(range(lower_bound, upper_bound)) - set(array1), array_length)
        random_element = random.choice(array1)
        array3 = [random_element] + random.sample(set(array1) - {random_element}, len(array1) - 1)
        self.assertTrue(self.clothes_instance.check_arrays(array1, array2))
        self.assertFalse(self.clothes_instance.check_arrays(array1, array3))

    @hypothesis.settings(deadline=None)
    @given(st.sampled_from(range(1, 21)))
    def test_assess_appearance(self, inp):
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
        self.assertIsNotNone(self.clothes_instance.assess_appearance(frames))


if __name__ == '__main__':
    unittest.main()
