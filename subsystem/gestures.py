import numpy as np
import mediapipe as mp


class Gestures:
    body_angles = [[16, 14, 12], [14, 12, 11], [15, 13, 11], [12, 11, 13],
                   [21, 15, 19], [19, 15, 17], [22, 16, 20], [20, 16, 18],
                   [18, 20, 16, 14], [17, 19, 15, 13], [11, 0, 12]]

    def __init__(self):
        self.body_res = {16: {'name': 'right elbow', 'res': []}, 14: {'name': 'right shoulder', 'res': []},
                         15: {'name': 'left elbow', 'res': []},
                         12: {'name': 'left shoulder', 'res': []},
                         21: {'name': 'left thumb', 'res': []},
                         19: {'name': 'left pinky', 'res': []}, 22: {'name': 'right thumb', 'res': []},
                         20: {'name': 'right pinky', 'res': []},
                         18: {'name': 'right wrist', 'res': []}, 17: {'name': 'left wrist', 'res': []},
                         11: {'name': 'head', 'res': []}}

    def get_vector_between_points(self, first_point, second_point):
        """
        Calculate vector between two points in 2d.
        :param first_point: list or array with 2 elements (x and y) - first point to calculate vector.
        :param second_point: list or array with 2 elements (x and y) - second point to calculate vector.
        :return: list wit x and y of calculated vector.
        """
        x1, y1 = first_point[0], first_point[1]
        x2, y2 = second_point[0], second_point[1]
        vector = np.array([x2, y2]) - np.array([x1, y1])
        return vector

    def angle_between_vectors(self, v1, v2):
        """
        Calculate angle in degrees between given vectors.
        :param v1: list or array with 2 elements (x and y) - first vector.
        :param v2: list or array with 2 elements (x and y) - second vector.
        :return: float value [0:360] - angle between v1 and v2.
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (norm_v1 * norm_v2)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        # Check the angle of the sign and adjust it in the range from 0 to 360 degrees.
        if np.cross(v1, v2) < 0:
            angle_deg = 360 - angle_deg

        return angle_deg

    def min_angle_difference(self, angle1, angle2):
        """
        Get min differance between two angles.
        :param angle1: float value [0:360] - first value in degrees.
        :param angle2: float value [0:360] - second value in degrees.
        :return: float value [0:360] - min angle between two angles in closed circle.
        """
        diff1 = abs(angle1 - angle2)
        diff2 = 360 - diff1
        return min(diff1, diff2)

    def point_between(self, point1, point2):
        """
        Calculate point between 2 points in 2d.
        :param point1: landmark with x and y attributes - first point.
        :param point2: landmark with x and y attributes - second point.
        :return: list with x and y of point between 2 given points.
        """
        return [(point1.x + point2.x) / 2, (point1.y + point2.y) / 2]

    def calculate_angles(self, landmarks, mean_angle):
        """
        calculate the displacement of the joints between frames.
        :param landmarks: coordinates of the main joints.
        :param mean_angle: dictionary for calculation results.
        :return: dictionary with results.
        """
        for angles in Gestures.body_angles:
            if all(landmarks[angle].visibility >= 0.5 for angle in angles):
                point_second = [landmarks[angles[-1]].x, landmarks[angles[-1]].y]
                point_mid = [landmarks[angles[-2]].x, landmarks[angles[-2]].y]
                if len(angles) > 3:
                    point_first = self.point_between(landmarks[angles[0]], landmarks[angles[1]])
                else:
                    point_first = [landmarks[angles[0]].x, landmarks[angles[0]].y]
                v1 = self.get_vector_between_points(point_first, point_mid)
                v2 = self.get_vector_between_points(point_mid, point_second)
                angle = self.angle_between_vectors(v1, v2)
                if mean_angle[angles[0]]['prev'] is not None:
                    mean_angle[angles[0]]['res'] += self.min_angle_difference(angle, mean_angle[angles[0]]['prev'])
                    mean_angle[angles[0]]['count'] += 1
                else:
                    mean_angle[angles[0]]['prev'] = angle
            else:
                mean_angle[angles[0]]['prev'] = None
        return mean_angle

    def process_velocity(self, frames):
        """
        Count angle displacement for all frames.
        :param frames: frames to process.
        :return: dictionary with results for each joint.
        """
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            mean_angle = {}
            for angle in Gestures.body_angles:
                mean_angle[angle[0]] = {}
                mean_angle[angle[0]]['prev'] = None
                mean_angle[angle[0]]['count'] = 0
                mean_angle[angle[0]]['res'] = 0
            for image in frames:
                results = pose.process(image)
                try:
                    landmarks = results.pose_landmarks.landmark
                    mean_angle = self.calculate_angles(landmarks, mean_angle)
                except Exception as ex:
                    continue
            for angle in Gestures.body_angles:
                if mean_angle[angle[0]]['count'] > 0:
                    result = mean_angle[angle[0]]['res'] / mean_angle[angle[0]]['count']
                    self.body_res[angle[0]]['res'].append(round(result, 2))
                else:
                    self.body_res[angle[0]]['res'].append(0)

    def get_result(self):
        """
        Get result angles for body parts.
        :return: dictionary with body parts as keys and angles as values.
        """
        return {value['name']: value['res'] for key, value in self.body_res.items()}
