import cv2
import mediapipe as mp
import torchvision.transforms as transforms
from PIL import Image
from clothes_attributes import CustomResNet, CustomModel
import torch
import torch.nn.functional as F


class Clothes:
    attributes = ['floral', 'graphic', 'striped', 'embroidered', 'solid', 'lattice',
                  'long_sleeve', 'short_sleeve', 'sleeveless', 'maxi_length',
                  'mini_length', 'crew_neckline', 'v_neckline', 'square_neckline',
                  'no_neckline', 'denim', 'tight', 'loose', 'conventional']
    not_acceptable_attributes = ['sleeveless', 'mini_length', 'denim', 'tight', 'loose']

    def __init__(self):
        """
        Initialize transforms.
        """
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def compute_image_sharpness(self, image):
        """
        Calculate sharpness of one image.
        :param image: image to process.
        :return: float value - sharpness of image.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray_image, cv2.CV_64F).var()

    def choose_sharpest_image(self, images):
        """
        Choose sharpest image for future assessing.
        :param images: frames for choosing.
        :return: sharpest image.
        """
        sharpest_image = None
        max_sharpness = 0

        for image in images:
            sharpness = self.compute_image_sharpness(image)
            if sharpness > max_sharpness:
                max_sharpness = sharpness
                sharpest_image = image

        return sharpest_image

    def transform_image(self, image):
        """
        Transform image into model input.
        :param image: image for processing.
        :return: tensor - transformed image.
        """
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        image_h, image_w, _ = image.shape
        results = pose.process(image)

        if results.pose_landmarks:
            # Identify bound box.
            x_min, y_min, x_max, y_max = image_w, image_h, 0, 0
            for landmark in results.pose_landmarks.landmark:
                x, y = int(landmark.x * image_w), int(landmark.y * image_h)
                x_min = max(0, min(x_min, x))
                y_min = max(0, min(y_min, y))
                x_max = min(image_w - 1, max(x_max, x))
                y_max = min(image_h - 1, max(y_max, y))
            image = image[y_min:y_max, x_min:x_max]

        pose.close()
        pil_image = Image.fromarray(image)
        image = self.transform(pil_image)
        return image

    def check_arrays(self, arr1, arr2):
        """
        Check presence of first array elements in second array.
        :param arr1: array for checking elements.
        :param arr2: second array for processing.
        :return: bool value if none of elements in first array is in the second.
        """
        for elem in arr1:
            if elem in arr2:
                return False
        return True

    def assess_appearance(self, frames):
        """
        Assess clothes attributes.
        :param frames: frames for choosing best frame for processing.
        :return: bool value if clothes is acceptable.
        """
        model = CustomResNet()
        custom_model = CustomModel(model)
        custom_model.model.load_state_dict(torch.load('saved_model_modified.pth'))
        image = self.choose_sharpest_image(frames)
        image = self.transform_image(image)
        image = image.unsqueeze(0)
        custom_model.eval()
        output = custom_model(image)
        pred = F.softmax(output, dim=1)
        topk_values, topk_indices = torch.topk(pred, 3, dim=1)
        captions = []
        for i in range(topk_indices.size(0)):
            for j in range(topk_indices.size(1)):
                captions.append(Clothes.attributes[topk_indices[i, j]])
        return self.check_arrays(captions, Clothes.not_acceptable_attributes)

