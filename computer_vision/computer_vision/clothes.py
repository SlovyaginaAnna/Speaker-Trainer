import torch
import torchvision.models as models
from torchvision.transforms import functional as F
from PIL import Image
import mediapipe as mp

class ClothesEvaluator:
    """
    Static class for processing images and detecting clothes on them.
    """
    model = models.resnet50(pretrained=True)
    model.eval()

    def detect_and_crop(frame):
        """
        Crop image to get just human clothes.
        @param frame: image to be cropped.
        @return: cropped frame or None if it is impossible to detect clothes.
        """
        # Load Mediapipe models to detect humans.
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        # Pose detection.
        results = pose.process(frame)

        if results.pose_landmarks is not None:
            # Find boundaries of human
            height, width = frame.shape[:2]
            top, right, bottom, left = height, 0, 0, width

            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if x < left:
                    left = x
                if x > right:
                    right = x
                if y < top:
                    top = y
                if y > bottom:
                    bottom = y

            # Crop frame
            cropped_frame = frame[top:bottom, left:right]
            return cropped_frame
        else:
            return None

    def load_and_preprocess_image(img):
        """
        Preprocess image into appropriate format.
        @param img: image to modify.
        @return: modified image.
        """
        img = Image.fromarray(img)
        img = F.resize(img, (224, 224))
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = torch.unsqueeze(img, 0)
        return img

    def classify_image(image):
        """
        Determine objects on the image.
        @param image: image to be classified.
        @return: class indices of found objects.
        """
        image = ClothesEvaluator.detect_and_crop(image)
        if image is None:
            return None
        img = ClothesEvaluator.load_and_preprocess_image(image)
        predictions = ClothesEvaluator.model(img)
        _, class_indices = torch.topk(predictions, k=5)
        #classes = requests.get('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json').json()
        #names = [classes[index] for index in class_indices[0]]
        #print(names, class_indices)
        return class_indices

    def is_outfit_appropriate(frame):
        """
        Determine if clothes is appropriate for public speaking.
        @param frame: image to be classified.
        @return: boolean value is clothes appropriate or not.
        """
        indices = ClothesEvaluator.classify_image(frame)
        if indices is None:
            return None
        if 617 in indices or 457 in indices or 578 in indices or 610 in indices:
            return True
        else:
            return False
