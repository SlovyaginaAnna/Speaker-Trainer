from cvzone.PoseModule import PoseDetector
import cv2
import numpy as np


class Perspective:
    def __init__(self):
        """
        Initialize model for detection.
        """
        self.detector = PoseDetector(staticMode=False,
                                modelComplexity=1,
                                smoothLandmarks=True,
                                enableSegmentation=False,
                                smoothSegmentation=True,
                                detectionCon=0.5,
                                trackCon=0.5)

    def point_between(self, point1, point2):
        """
        Calculate point between 2 points in 2d.
        :param point1: list with x and y of first point.
        :param point2: list with x and y of second point.
        :return: list with x and y of mid point.
        """
        return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]

    def count_brightness(self, frames):
        """
        Asses lightning on frames.
        :param frames: list with frames to process.
        :return: string value - lightning.
        """
        dark = 0
        optimal = 0
        bright = 0
        for frame in frames:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray_image)
            if mean_brightness < 100:
                dark += 1
            elif mean_brightness > 200:
                bright += 1
            else:
                optimal += 1
        dark /= len(frames)
        optimal /= len(frames)
        bright /= len(frames)
        if dark >= optimal and dark >= bright:
            return 'dark'
        elif bright >= dark and bright >= optimal:
            return 'bright'
        else:
            return 'optimal'

    def check_correct_pose(self, bounding_box, eye_coords, image_width, image_height):
        """
        Check if speaker in a right position.
        :param bounding_box: coordinates of the speakers bound box.
        :param eye_coords: eyes coordinates.
        :param image_width: width of an image.
        :param image_height: height of an image.
        :return: 1 or 0 - if position is correct.
        """
        x_center = image_width // 2
        y_third_line = image_height // 3
        x1, y1, x2, y2 = bounding_box['bbox']
        if abs(x1 + ((x2 - x1) / 2) - x_center) > 0.2 * image_width:
            return 0

        # Check eye position according rule of the third.
        eye_x, eye_y = eye_coords
        if eye_y < y_third_line - 0.15 * image_height or eye_y > y_third_line + 0.15 * image_height:
            return 0
        return 1

    def count_angle(self, frames):
        """
        Count percent of incorrect frames.
        :param frames: frames to process.
        :return: percent of incorrect frames.
        """
        incorrect_pose = 0
        bbox_length = 0
        inc_index = []
        ind = 0
        for frame in frames:
            img = self.detector.findPose(frame, draw=False)
            lm_list, bbox_info = self.detector.findPosition(img, draw=False, bboxWithHands=False)
            right_coords = [lm_list[5][0], lm_list[5][1]]
            left_coords = [lm_list[2][0], lm_list[2][1]]
            width, height, _ = frame.shape
            if self.check_correct_pose(bbox_info, self.point_between(right_coords, left_coords), width, height) == 0:
                inc_index.append(ind)
            else:
                incorrect_pose += 1
            ind += 1
            length = bbox_info['bbox'][2] - bbox_info['bbox'][0]
            if length > bbox_length:
                bbox_length = length
        return incorrect_pose / len(frames), bbox_length, inc_index
