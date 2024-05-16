import inspect
import os

import cv2
from gaze import GazeDirection
from perspective import Perspective
from gestures import Gestures
from clothes import Clothes
from emotions import VideoEmotions
from tqdm import tqdm
from joblib import load
import math


class VideoSubsystem:
    acceptable_velocity = {'right elbow': [5, 50], 'left elbow': [5, 50], 'left shoulder': [2, 25],
                           'right shoulder': [2, 25], 'left thumb': [0, 30], 'left pinky': [3, 40],
                           'right thumb': [0, 30], 'right pinky': [3, 40], 'right wrist': [3, 40],
                           'left wrist': [3, 40], 'head': [0, 12]}

    def __init__(self, path, inappropriate_emotions, emotions=True, gesticulation=True, angle=True, gaze=True, clothes=True,
                 device='cpu', dist=5):
        """
        Initialize parameters.
        :param path: path of the video to analyze.
        :param inappropriate_emotions: list of inappropriate emotions to count percentage.
        :param emotions: flag if analyze emotions.
        :param gesticulation: flag if analyze gesture velocity.
        :param angle: flag if analyze percent of incorrect perspective.
        :param gaze: flag if analyze percent of incorrect gaze direction.
        :param clothes: flag if analyze clothes.
        :param device: cpu or gpu.
        :param dist: number of frames for analyzing in one second.
        """

        self.inappropriate_emotions = inappropriate_emotions
        self.device = device
        self.video_path = path
        self.emotions = emotions
        self.gesticulation = gesticulation
        self.angle = angle
        self.gaze = gaze
        self.clothes = clothes
        self.dist = dist

        current_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
        current_directory = os.path.dirname(current_file_path)
        path = current_directory + "\\data\\model_first.joblib"
        self.emotion_model = load(path)

        self.emotion_list = []
        self.emotion_inappropriate_percentage = []
        self.gesture_list = []
        self.angle_list = []
        self.gaze_list = []
        self.lightning = []
        self.angle_len = []
        self.inc_ind = []
        self.clothes_estimation = None
        cap = cv2.VideoCapture(self.video_path)
        self.fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

    def get_emotions(self):
        """
        Get emotionality rate.
        :return: list with emotionality rates.
        """
        return self.emotion_list

    def get_gestures(self):
        """
        Get rate of gestures: 0 - low, 1 - medium, 2 - high.
        :return: list with gesticulation rates.
        """
        return self.gesture_list

    def get_angle(self):
        """
        Return percent of incorrect angle.
        :return: list with percents for each fragment.
        """
        return self.angle_list

    def get_gaze(self):
        """
        Get percent of incorrect gaze - percent of frames when speaker looks not in the center.
        :return: list of percents for each fragment.
        """
        return self.gaze_list

    def get_lightning(self):
        """
        Get flags for lightning.
        :return: list of flags for each fragment.
        """
        return self.lightning

    def get_angle_len(self):
        """
        Get length of the widest bound box of the speaker.
        :return: list with values for each fragment.
        """
        return self.angle_len

    def get_clothes_estimation(self):
        """
        Return flag for correct clothes.
        :return: flag for clothes.
        """
        return self.clothes_estimation

    def get_incorrect_angle_ind(self):
        """
        Get indices if frames when perspective is incorrect.
        :return: list of indices for each fragment.
        """
        return self.inc_ind

    def get_inappropriate_emotion_percentage(self):
        """
        Get percent of incorrect emotions.
        :return: list of percents for each fragment.
        """
        return self.emotion_inappropriate_percentage
    @staticmethod
    def get_subarray(array, subset, ind):
        """
        Get subarray.
        :param array: array to get subarray from it.
        :param subset: number of elements in subarray.
        :param ind: index of array from which subarray starts.
        :return: subarray.
        """
        last_ind = min(ind + subset, len(array))
        return array[ind:last_ind]

    @staticmethod
    def calculate_percentage(percent_list):
        """
        Calculate percent of each element in the list.
        :param percent_list: list for calculating percents.
        :return: dictionary with elements of list as keys and percents as values.
        """
        total_frames = len(percent_list)
        percentage = {}
        for element in percent_list:
            if element in percentage.keys():
                percentage[element] += 1
            else:
                percentage[element] = 1
        for element in percentage.keys():
            percentage[element] = (percentage[element] / total_frames) * 100

        return percentage

    def process_emotions(self, frames):
        """
        Evaluate emotionality of video fragment.
        :param frames: list of frames for evaluation.
        :return: string value - emotionality.
        """
        total_frames = len(frames)
        emotion_class = VideoEmotions()
        emotion_results = []
        fps = int(self.fps)
        for i in range(0, total_frames, int(fps)):
            sec_frames = self.get_subarray(frames, fps, i)[::self.dist]
            emotions, scores = emotion_class.process_frames(sec_frames)
            percentages = VideoSubsystem.calculate_percentage(emotions)
            try:
                max_emotion = max(percentages, key=percentages.get)
                emotion_results.append(max_emotion)
            except Exception as ex:
                print(ex.args[0])
                emotion_results.append('emotion not determined')
        frequency = emotion_class.calculate_emotion_change_frequency(emotion_results)
        percentages = VideoSubsystem.calculate_percentage(emotion_results)
        features = [frequency]
        for element in ['Sadness', 'Disgust', 'Fear', 'Neutral', 'Happiness', 'Anger',
        'Contempt']:
            if element in percentages.keys():
                features.append(percentages[element])
            else:
                features.append(0.0)
        res = self.emotion_model.predict([features])[0]
        percent_res = 0.0
        for element in self.inappropriate_emotions:
            if element in percentages.keys():
                percent_res += percentages[element]
        return res, percent_res * 0.01

    def replace_values_with_condition(self, dictionary):
        """
        Change values for values in rating scale.
        :param dictionary: dictionary with unprocessed values.
        :return: dictionary with processed values.
        """
        for key, value in dictionary.items():
            min_val = VideoSubsystem.acceptable_velocity[key][0]
            max_val = VideoSubsystem.acceptable_velocity[key][1]
            for i in range(len(value)):
                if value[i] < min_val:
                    value[i] = '0'
                elif value[i] > max_val:
                    value[i] = '2'
                else:
                    value[i] = '1'
            dictionary[key] = value
        return dictionary

    def process_gesticulation(self, frames, duration=10):
        """
        Estimate velocity of the speaker.
        :param frames: list of frames for estimation.
        :param duration: number of seconds for estimation.
        :return: estimated velocity.
        """
        gesture = Gestures()
        total_frames = len(frames)
        fps = int(self.fps)
        for i in range(0, total_frames, fps):
            sec_frames = self.get_subarray(frames, fps, i)[::self.dist]
            gesture.process_velocity(sec_frames)
        res = gesture.get_result()
        res = self.replace_values_with_condition(res)
        result = []
        key = list(res.keys())[0]
        cycle = len(res[key])
        for ind in range(cycle):
            percent = []
            for key in res.keys():
                percent.append(res[key][ind])
            percentage = VideoSubsystem.calculate_percentage(percent)
            if '2' in percentage.keys():
                result.append(2)
            elif '0' in percentage.keys() and percentage['0'] > 70:
                result.append(0)
            else:
                result.append(1)
        all_percent = VideoSubsystem.calculate_percentage(result)
        if 2 in all_percent.keys():
            return 2
        elif 0 in all_percent.keys() and all_percent[0] > 70:
            return 0
        else:
            return 1

    def process_gaze(self, frames):
        """
        Calculate percent of incorrect gaze.
        :param frames: list of frames for processing.
        :return: float value - percent of incorrect frames.
        """
        model = GazeDirection()
        percent = model.gaze_detection(frames)
        percentages = VideoSubsystem.calculate_percentage(percent)
        # max_key = max(percentages, key=percentages.get)
        try:
            result = (100 - percentages['center']) * 0.01
        except:
            result = 1
        return result

    def process_angle(self, frames):
        """
        Calculate incorrect angles.
        :param frames: list of frames for processing.
        :return: float value - percent of incorrect frames.
        """
        perspective = Perspective()
        brightness = perspective.count_brightness(frames[::self.dist])
        percent, length, inc_ind = perspective.count_angle(frames)
        return percent, length, brightness, inc_ind

    def process_clothes(self, frames):
        """
        Defines if clothes is appropriate.
        :param frames: list of frames for processing.
        :return: bool value if clothes is appropriate.
        """
        clothes = Clothes()
        return clothes.assess_appearance(frames)


    def process_video(self, duration=10):
        """
        Read for duration seconds and process frames.
        :param duration: number of seconds to process in one cycle.
        :return: dictionary with results.
        """
        cap = cv2.VideoCapture(self.video_path)
        try:
            segment_duration = duration
            segment_frame_count = math.ceil(cap.get(cv2.CAP_PROP_FPS) * segment_duration)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in tqdm(range(0, frame_count, segment_frame_count)):
                frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                for j in range(segment_frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                if self.emotions:
                    res, percent_res = self.process_emotions(frames)
                    self.emotion_list.append(res)
                    self.emotion_inappropriate_percentage.append(percent_res)
                if self.gesticulation:
                    res = self.process_gesticulation(frames)
                    self.gesture_list.append(res)
                if self.angle:
                    res, length, brightness, inc_ind = self.process_angle(frames)
                    self.angle_list.append(res)
                    self.angle_len.append(length)
                    self.lightning.append(brightness)
                    self.inc_ind.append(inc_ind)
                if self.gaze:
                    res = self.process_gaze(frames)
                    self.gaze_list.append(res)
                if self.clothes and self.clothes_estimation is None:
                    self.clothes_estimation = self.process_clothes(frames)
        finally:
            cap.release()
            cv2.destroyAllWindows()
