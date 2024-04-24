import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class GazeDirection:
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 286, 258, 257, 259, 260]
    RIGHT_IRIS = [468, 470, 469, 472, 471]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 30, 29, 28, 27, 56]
    LEFT_IRIS = [473, 475, 474, 477, 476]

    def __init__(self, threshold=0.1):
        """
        Initialize prediction model and threshold.
        :param threshold: float value - acceptable displacement of iris.
        """
        model_file = open('face_landmarker_v2_with_blendshapes.task', "rb")
        model_data = model_file.read()
        model_file.close()
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.threshold = threshold

    def count_displacement(self, eye_coords, iris_coords):
        """
        Calculate the position of iris in percent relatively center.
        :param eye_coords: all coordinates of eye.
        :param iris_coords: all coordinates of iris.
        :return: percent of x and y axis - position of an iris.
        """
        max_x = (max(eye_coords, key=lambda item: item[0]))[0]
        min_x = (min(eye_coords, key=lambda item: item[0]))[0]
        max_y = (max(eye_coords, key=lambda item: item[1]))[1]
        min_y = (min(eye_coords, key=lambda item: item[1]))[1]
        width = max_x - min_x
        height = max_y - min_y
        iris_x = iris_coords[0][0]
        iris_y = iris_coords[0][1]
        percent_x = (2 * iris_x - width - 2 * min_x) / width
        percent_y = (2 * iris_y - height - 2 * min_y) / height
        return percent_x, percent_y

    def process_gaze(self, right_x, right_y, left_x, left_y):
        """
        Asses gaze.
        :param right_x: x position of right iris.
        :param right_y: y position of right iris.
        :param left_x: x position of left iris.
        :param left_y: y position of left iris.
        :return: string value - gaze direction.
        """
        x = (right_x + left_x) / 2
        y = (right_y + left_y) / 2
        if y > 0.45:
            result = "down "
        elif y < 0.2:
            result = "up "
        else:
            result = ""

        if abs(x) > self.threshold and x > 0:
            result += "right"
        elif abs(x) > self.threshold and x < 0:
            result += "left"
        else:
            result += "center"
        return result

    def landmarks_detection(self, img_width, img_height, face_landmarks, ind):
        """
        Transform coordinates into pixels of image.
        :param img_width: width of an image.
        :param img_height: height of an image.
        :param face_landmarks: not transformed landmarks.
        :param ind: indexes of required points.
        :return: transformed coordinates.
        """
        mesh_coord = [(int(face_landmarks[i].x * img_width), int(face_landmarks[i].y * img_height)) for i in ind]
        return mesh_coord

    def gaze_detection(self, frames):
        """
        Calculate direction of eyes on each frame.
        :param frames: frames for processing.
        :return: list with string results for all frames.
        """
        result_list = []
        for frame in frames:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            results = self.detector.detect(image)
            frame_width = frame.shape[0]
            frame_height = frame.shape[1]
            try:
                face_landmarks = results.face_landmarks[0]
                left_iris_coords = self.landmarks_detection(frame_width, frame_height, face_landmarks, GazeDirection.LEFT_IRIS)
                right_iris_coords = self.landmarks_detection(frame_width, frame_height, face_landmarks, GazeDirection.RIGHT_IRIS)
                left_eye_coords = self.landmarks_detection(frame_width, frame_height, face_landmarks, GazeDirection.LEFT_EYE)
                right_eye_coords = self.landmarks_detection(frame_width, frame_height, face_landmarks, GazeDirection.RIGHT_EYE)
                right_x, right_y = self.count_displacement(right_eye_coords, right_iris_coords)
                left_x, left_y = self.count_displacement(left_eye_coords, left_iris_coords)
                res = self.process_gaze(right_x, right_y, left_x, left_y)
                result_list.append(res)
            except Exception as ex:
                continue
        return result_list
