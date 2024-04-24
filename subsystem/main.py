from subsystem import VideoSubsystem

system = VideoSubsystem('C:\\Users\\Operator\\Desktop\\Project\\video\\2.mp4', emotions=True, gesticulation=True, angle=True, gaze=True, clothes=True)
res = system.process_video('output.mp4')
print(res)




