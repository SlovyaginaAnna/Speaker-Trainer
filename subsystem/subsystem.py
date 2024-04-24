import cv2
from gaze import GazeDirection
from perspective import Perspective
from gestures import Gestures
from clothes import Clothes
from emotions import VideoEmotions
from tqdm import tqdm
from joblib import load


class VideoSubsystem:
    velocity_scale = {2: 'Активная жестикуляция', 1: 'Оптимальная жестикуляция', 0: 'Неактивная жестикуляция'}
    acceptable_velocity = {'right elbow': [5, 50], 'left elbow': [5, 50], 'left shoulder': [2, 25],
                           'right shoulder': [2, 25], 'left thumb': [0, 30], 'left pinky': [3, 40],
                           'right thumb': [0, 30], 'right pinky': [3, 40], 'right wrist': [3, 40],
                           'left wrist': [3, 40], 'head': [0, 12]}

    def __init__(self, path, emotions=False, gesticulation=False, angle=False, gaze=False, clothes=False,
                 device='cpu', dist=5, acceptable_angle=0.6):
        self.fps = None
        self.frame_height = None
        self.frame_width = None
        self.device = device
        self.video_path = path
        self.emotions = emotions
        self.gesticulation = gesticulation
        self.angle = angle
        self.gaze = gaze
        self.clothes = clothes
        self.draw = {}
        self.dist = dist
        self.acceptable_angle = acceptable_angle
        self.emotion_model =load('model_first.joblib')

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
        if res < 0.2:
            self.draw['emotion'] = {'value': "Эмоциональность: не эмоционально"}
        elif res > 0.6:
            self.draw['emotion'] = {'value': "Эмоциональность: слишком эмоционально"}
        else:
            self.draw['emotion'] = {'value': "Эмоциональность: оптимально"}
        return res

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
        self.draw['velocity'] = {'value': result}
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
        max_key = max(percentages, key=percentages.get)
        if max_key != 'center':
            self.draw['gaze'] = {'value': 'Вы часто отводите взгляд'}
        return 1 - percentages['center']

    def process_angle(self, frames):
        """
        Calculate incorrect angles.
        :param frames: list of frames for processing.
        :return: float value - percent of incorrect frames.
        """
        perspective = Perspective()
        brightness = perspective.count_brightness(frames[::self.dist])
        percent, length, inc_ind = perspective.count_angle(frames[::self.dist])
        self.draw['angle'] = {'length': length, 'ind': inc_ind}
        if brightness == 'dark':
            self.draw['brightness'] = {'value': 'Слишком темное освещение'}
        elif brightness == 'bright':
            self.draw['brightness'] = {'value': 'Слишком яркое освещение'}
        else: self.draw['brightness'] = {'value': 'Оптимальное освещение'}
        return percent

    def process_clothes(self, frames):
        """
        Defines if clothes is appropriate.
        :param frames: list of frames for processing.
        :return: bool value if clothes is appropriate.
        """
        clothes = Clothes()
        return clothes.assess_appearance(frames)

    def draw_angle(self, frame, length, color):
        """
        Draw lines for correct angle.
        :param frame: image for drawing.
        :param length: length of speaker's bound box.
        :param color: red if angle is incorrect, green otherwise.
        :return: new frame with angle lines.
        """
        image_orig = frame.copy()
        height, width = frame.shape[:2]
        center_x = width // 2
        line_length = length // 2
        line_thickness = 5
        line_offset_top = height // 3 + int(0.15 * height)
        line_offset_bottom = height // 3 - int(0.15 * height)

        cv2.line(frame, (center_x - line_length, height), (center_x-line_length, 0), color, line_thickness)
        cv2.line(frame, (center_x + line_length, height), (center_x + line_length, 0), color, line_thickness)
        cv2.line(frame, (center_x - line_length, line_offset_top), (center_x + line_length, line_offset_top),
                 color, line_thickness)
        cv2.line(frame, (center_x - line_length, line_offset_bottom), (center_x + line_length, line_offset_bottom),
                 color, line_thickness)
        font = cv2.FONT_HERSHEY_COMPLEX
        bottom_left_corner_text = (center_x - line_length, line_offset_top - 20)
        font_scale = 0.5
        font_color = color
        line_type = 1
        cv2.putText(frame, 'Рекомендуемый уровень глаз', bottom_left_corner_text, font, font_scale, font_color,
                    line_type)
        image_out = cv2.addWeighted(frame, 0.3, image_orig, 0.7, 0.0)
        return image_out

    def draw_frames(self, frames):
        """
        Draw results on video frames.
        :param frames: frames for processing.
        :return: processed frames.
        """
        font = cv2.FONT_HERSHEY_COMPLEX
        x = 20
        y = 50
        font_scale = 1
        thickness = 1
        green_color = (50, 205, 50)
        red_color = (220, 20, 60)
        bb_length = self.draw['angle']['length']
        inc_ind = self.draw['angle']['ind']
        for ind in range(len(frames)):
            count = 1
            for key in self.draw.keys():
                if key != 'velocity' and key != 'angle':
                    if self.draw[key]['value'] in ['Эмоциональность: оптимально', 'Оптимальная жестикуляция',
                                                   'Оптимальное освещение']:
                        color = green_color
                    else:
                        color = red_color
                    frames[ind] = cv2.putText(frames[ind], self.draw[key]['value'], (x, y * count), font, font_scale,
                                              color, thickness, cv2.LINE_AA)
                    count += 1
                elif key == 'velocity':
                    current_sec = int(ind // self.fps)
                    text = VideoSubsystem.velocity_scale[self.draw[key]['value'][current_sec]]
                    if text == 'Оптимальная жестикуляция':
                        color = green_color
                    else:
                        color = red_color
                    frames[ind] = cv2.putText(frames[ind], text,
                                              (x, y * count), font, font_scale, color, thickness, cv2.LINE_AA)
                    count += 1
            if ind in inc_ind:
                color = red_color
            else:
                color = green_color
            frames[ind] = self.draw_angle(frames[ind], bb_length, color)
        return frames

    def process_video(self, output_path, duration=10):
        """
        Read for duration seconds and process frames.
        :param output_path: new path of processed video.
        :param duration: number of seconds to process in one cycle.
        :return: dictionary with results.
        """
        result = {'emotions': [], 'gesticulation': [], 'angle': [], 'gaze': [], 'clothes': None}
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        segment_duration = duration
        segment_frame_count = int(self.fps * segment_duration)
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
                res = self.process_emotions(frames)
                result['emotions'].append(res)
            if self.gesticulation:
                res = self.process_gesticulation(frames)
                result['gesticulation'].append(res)
            if self.angle:
                res = self.process_angle(frames)
                result['angle'].append(res)
            if self.gaze:
                res = self.process_gaze(frames)
                result['gaze'].append(res)
            if self.clothes and result['clothes'] is None:
                res = self.process_clothes(frames)
                result['clothes'] = res
            frames = self.draw_frames(frames)
            for j in range(len(frames)):
                out.write(frames[j])

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.draw = {}
        return result
