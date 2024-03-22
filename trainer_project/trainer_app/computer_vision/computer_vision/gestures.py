import numpy as np
import mediapipe as mp

class Gestures:
    """
    Class for processing human poses.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    num_of_joints = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    connections = {14: 12, 16: 14, 13: 11, 15: 13, 24: 12, 23: 11, 26: 24, 28: 26, 25: 23, 27: 25}

    def angular_velocity_count(point2_start, point2_end, point1_start, point1_end):
        """
        Count angular velocity of two vectors.
        @param point2_start: start of second vector.
        @param point2_end:  end of second vector.
        @param point1_start: start of the first vector.
        @param point1_end: end of the first vector.
        @return: angular velocity.
        """
        vector1_start = np.array(point1_start)
        vector1_end = np.array(point1_end)
        vector2_start = np.array(point2_start)
        vector2_end = np.array(point2_end)

        # Offset vector calculation
        displacement_vector1 = vector1_end - vector1_start
        displacement_vector2 = vector2_end - vector2_start

        # Count angles between vectors.
        angle1 = np.arctan2(displacement_vector1[1], displacement_vector1[0])
        angle2 = np.arctan2(displacement_vector2[1], displacement_vector2[0])

        # Count angular velocity.
        angular_velocity = abs(angle2 - angle1)

        return np.degrees(angular_velocity)

    def mid_point(landmark_start, landmark_end):
        """
        Count point between two points.
        @param landmark_start: first landmark.
        @param landmark_end: second landmark.
        @return: point between landmarks.
        """
        x = (landmark_end.x + landmark_start.x) / 2
        y = (landmark_end.y + landmark_start.y) / 2
        return [x, y]

    def count_mean_velocity(frames, n = 1):
        """
        Count mean angular velocity among the frames of the video.
        @param frames: frames of the video.
        @param n: how many frames processing takes place.
        @return: mean angular velocity of the video.
        """
        imgs = frames[::n]
        joints = {}
        results = list(map(Gestures.pose.process, imgs))
        for result in results:
            if result is not None and result.pose_landmarks is not None and result.pose_landmarks.landmark is not None:
                for count, landmark in enumerate(result.pose_landmarks.landmark):
                    if count in Gestures.num_of_joints:
                        if count not in joints.keys():
                            joints[count] = {}
                            joints[count]['velocity'] = 0
                            joints[count]['visibility'] = 0
                        joints[count]['current'] = landmark
                if 'prev' in joints[0].keys():
                    for key in Gestures.connections.keys():
                        nearest_joint = Gestures.connections[key]
                        prev2 = [joints[key]['prev'].x, joints[key]['prev'].y]
                        current2 = [joints[key]['current'].x, joints[key]['current'].y]
                        prev1 = [joints[nearest_joint]['prev'].x, joints[nearest_joint]['prev'].y]
                        current1 = [joints[nearest_joint]['current'].x, joints[nearest_joint]['current'].y]
                        # joints[key]['velocity'] += angular_velocity_count(prev2, current2, prev1, current1)
                        joints[key]['velocity'] += Gestures.angular_velocity_count(current1, current2, prev1, prev2)
                    mid_prev = Gestures.mid_point(joints[11]['prev'], joints[12]['prev'])
                    mid_current = Gestures.mid_point(joints[11]['current'], joints[12]['current'])
                    zero_prev = [joints[0]['prev'].x, joints[0]['prev'].y]
                    zero_current = [joints[0]['current'].x, joints[0]['current'].y]
                    # joints[0]['velocity'] += angular_velocity_count(zero_prev, zero_current, mid_prev, mid_current)
                    joints[0]['velocity'] += Gestures.angular_velocity_count(mid_current, zero_current, mid_prev, zero_prev)
                for key in joints.keys():
                    joints[key]['visibility'] += joints[key]['current'].visibility
                    joints[key]['prev'] = joints[key]['current']
        angular_velocity = 0
        num = 1
        for key in Gestures.connections.keys():
            if joints[key]['visibility'] > 0.5 and joints[Gestures.connections[key]]['visibility'] > 0.5:
                angular_velocity += joints[key]['velocity']
                num += 1
        angular_velocity += joints[0]['velocity']
        return (angular_velocity / num) / (len(imgs) - 1)