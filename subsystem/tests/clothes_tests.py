import unittest
import numpy as np
from clothes import Clothes


class ClothesTestCase(unittest.TestCase):
    def setUp(self):
        self.clothes_instance = Clothes()

    def test_compute_image_sharpness(self):
        image = np.zeros((50, 50, 3), np.uint8)
        sharpness = self.clothes_instance.compute_image_sharpness(image)
        self.assertEqual(sharpness, 0)

    def test_choose_sharpest_image(self):
        image1 = np.zeros((50, 50, 3), np.uint8)
        image2 = np.ones((50, 50, 3), np.uint8) * 255
        images = [image1, image2]
        sharpest_image = self.clothes_instance.choose_sharpest_image(images)
        self.assertIsNotNone(sharpest_image)
        self.assertEqual(sharpest_image.tolist(), image2.tolist())


if __name__ == '__main__':
    unittest.main()
