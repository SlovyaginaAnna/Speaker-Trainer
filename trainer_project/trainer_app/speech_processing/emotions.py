import math
import os
from aniemore.models import HuggingFaceModel
from aniemore.recognizers.voice import VoiceRecognizer
from moviepy.audio.io.AudioFileClip import AudioFileClip
from pydub import AudioSegment
from tqdm import tqdm


class AudioEmotions:
    params = {
        "emotions_time_window": 5,  # Number of seconds per which emotion is detected
    }

    def __init__(self, path):
        """
        Initialization of emotion classification class
        @param path: path to audio file
        """
        self.path = path

    def emotions_analysis(self):
        """
        Analyzes speech per N seconds (see params) and provides emotions probabilities
        @return: list of six-element lists with emotions probabilities
        """
        model = VoiceRecognizer(model=HuggingFaceModel.Voice.Wav2Vec2)
        audioclip = AudioFileClip(self.path)

        path = os.path.abspath(os.path.dirname(__file__))
        # Paths for N-second sub clips
        subclip_path = os.path.abspath(os.path.join(path, "file_processing/processing.wav"))
        subclip_modified_path = os.path.abspath(os.path.join(path, "file_processing/processing2.wav"))

        result = []
        time = self.params["emotions_time_window"]
        for i in tqdm(range(math.floor(audioclip.duration) // time)):
            subclip = audioclip.subclip(i * time, i * time + time)
            subclip.write_audiofile(subclip_path, logger=None)

            # Sub clip preprocessing to convert stereo to mono
            sound = AudioSegment.from_wav(subclip_path)
            sound = sound.set_channels(1)
            sound.export(subclip_modified_path, format="wav")

            res = model.recognize(subclip_modified_path, return_single_label=False)
            result.append([res["neutral"], res["happiness"], res["sadness"],
                           res["enthusiasm"], res["disgust"], res["anger"]])
        return result
