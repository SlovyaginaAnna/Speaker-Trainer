import cv2
import math


class DrawResults:
    def __init__(self, path, dist=10, good_color=(0, 128, 0), bad_color=(60, 20, 220), x_start = 20, y_start = 30):
        """
        Initialize parameters for drawing.
        :param path: path to the video for redrawing it.
        :param dist: Length of each fragment with same parameters.
        :param good_color: color for right values.
        :param bad_color: color for not right values.
        :param x_start: x coordinate of the start of the text.
        :param y_start: y coordinate of the start of the text.
        """
        self.video_path = path
        self.dist = dist
        self.right_color = good_color
        self.not_right_color = bad_color
        self.x = x_start
        self.y = y_start

    def draw_frames(self, frame, text, color_flag):
        """
        Draw text on one frame.
        :param frame: Frame on wich to draw text.
        :param text: List with string values, each element in list is separate line.
        :param color_flag: flag for right or not right color.
        :return: frame with text.
        """
        font = cv2.FONT_HERSHEY_COMPLEX
        x = self.x
        y = self.y
        font_scale = 0.5
        thickness = 1
        count = 1
        for i in range(len(text)):
            if color_flag[i]:
                color = self.right_color
            else:
                color = self.not_right_color
            if text[i] is not None:
                frame = cv2.putText(frame, text[i], (x, y * count), font, font_scale, color, thickness, cv2.LINE_AA)
                count += 1
        return frame

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
        bound_coef = 0.15
        line_offset_top = height // 3 + int(bound_coef * height)
        line_offset_bottom = height // 3 - int(bound_coef * height)
        font_color = self.right_color
        if not color:
            font_color = self.not_right_color
        cv2.line(frame, (center_x - line_length, height), (center_x - line_length, 0), font_color, line_thickness)
        cv2.line(frame, (center_x + line_length, height), (center_x + line_length, 0), font_color, line_thickness)
        cv2.line(frame, (center_x - line_length, line_offset_top), (center_x + line_length, line_offset_top),
                 font_color, line_thickness)
        cv2.line(frame, (center_x - line_length, line_offset_bottom), (center_x + line_length, line_offset_bottom),
                 font_color, line_thickness)
        font = cv2.FONT_HERSHEY_COMPLEX
        bottom_left_corner_text = (center_x - line_length, line_offset_top - 20)
        font_scale = 0.5
        line_type = 1
        cv2.putText(frame, 'Рекомендуемый уровень глаз', bottom_left_corner_text, font, font_scale, font_color,
                    line_type)
        image_out = cv2.addWeighted(frame, 0.3, image_orig, 0.7, 0.0)
        return image_out

    def draw(self, output_path, text, colors, angle, angle_color):
        """
        Draw text on the frames of the video.
        :param output_path: Path of the result video.
        :param text: List with text for drawing.
        :param colors: List with flag if color is correct.
        :param angle: Values for drawing bounds for correct position.
        :param angle_color: Color of bounds for drawing position.
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        try:
            segment_duration = self.dist
            segment_frame_count = math.ceil(fps * segment_duration)
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ind = i // segment_frame_count
                text_elements = [row[ind] for row in text]
                colors_elements = [row[ind] for row in colors]
                frame = self.draw_frames(frame, text_elements, colors_elements)
                if len(angle) > 0 and angle[ind] is not None:
                    color = True
                    if i % segment_frame_count in angle_color[ind]:
                        color = False
                    frame = self.draw_angle(frame, angle[ind], color)
                out.write(frame)
                i += 1
        except Exception as e:
            print(e.args)
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
