from subsystem import VideoSubsystem
from draw import DrawResults

system = VideoSubsystem('C:\\Users\\Operator\\Desktop\\Project\\video\\2.mp4', ['Disgust', 'Anger'], emotions=False,
                        gesticulation=False, gaze=False, clothes=False)
system.process_video(duration=10)
print(system.get_emotions())
print(system.get_gestures())
print(system.get_angle())
print(system.get_gaze())
print(system.get_lightning())
print(system.get_angle_len())
print(system.get_clothes_estimation())
print(system.get_incorrect_angle_ind())
print(system.get_inappropriate_emotion_percentage())

draw_res = DrawResults('C:\\Users\\Operator\\Desktop\\Project\\video\\12.mp4', dist=10)
draw_res.draw('output.mp4', [['Some text', 'Some text', 'Some text']], [[True, True, False]],
              system.get_angle_len(), system.get_incorrect_angle_ind())








