import moviepy.editor as mp_editor
import string
import whisper_timestamped
import joblib
from .filler_words import *
from .background_noise import *
from .speech_rate import *
from .emotions import *


class SpeechProcessingSubsystem:
    def __init__(self, path):
        """
        Initialization of speech processing class
        @param path: path to video file
        """
        # Rewrite video to audio file
        clip = mp_editor.VideoFileClip(path)
        audio_path = path[:path.rfind('.')] + '.wav'
        clip.audio.write_audiofile(audio_path, logger=None)
        self.clip_duration = clip.duration
        self.path = audio_path
        self.transcription = None
        self.cleaned_transcription = None
        self.word_arrays = None  # all_words, all_words_without_noise, noise
        self.duration = 0

    def speech_recognition(self):
        """
        Translates audio to text, creates words lists with timestamps (with and without background noise)
        """
        path = os.path.abspath(os.path.dirname(__file__))
        ASR_model_path = os.path.abspath(os.path.join(path, "models/whisper_model.sav"))
        model = joblib.load(open(ASR_model_path, 'rb'))
        audio = whisper_timestamped.load_audio(self.path)
        self.transcription = whisper_timestamped.transcribe(model, audio, language="ru", detect_disfluencies=True,
                                                            remove_punctuation_from_words=False)
        self.check_transcription()
        # Creation of transcription without punctuation marks
        transcription = self.transcription["text"].lower()
        transcription = transcription.translate(str.maketrans('', '', string.punctuation))
        transcription = "".join([ch for ch in transcription if ch not in string.digits])
        self.cleaned_transcription = " ".join(transcription.split())
        # Duration of speech
        self.duration = self.transcription["segments"][-1]["end"] - self.transcription["segments"][0]["start"]
        self.word_arrays = self.get_words()

    def check_transcription(self):
        """
        Checks if transcription is correct (if there are word doubles at the end of transcription)
        """
        words = self.transcription["text"].split()
        segments = self.transcription["segments"]
        end_idx = len(segments)
        for i in range(len(segments)):
            if segments[i]["end"] > self.clip_duration:
                end_idx = i
                break
        if end_idx == len(segments):
            print("correct transcription")
            return 0
        else:
            extra_words = 0
            for i in range(end_idx, len(segments)):
                extra_words += len(segments[i]["text"].split())
            # Transcription correction
            self.transcription["text"] = " ".join((self.transcription["text"].split())[:len(words) - extra_words])
            self.transcription["segments"] = self.transcription["segments"][:end_idx]

    def get_words(self):
        """
        Creates lists with all words (with background noise), words without noise and only noise
        @return: three lists with dicts of words and their timestamps
        """
        all_words, all_words_without_noise, noise = [], [], []
        for sentence in self.transcription["segments"]:
            for word in sentence["words"]:
                all_words.append(word)
                if word["text"] != "[*]":
                    all_words_without_noise.append(word)
                else:
                    noise.append((word["start"], word["end"]))
        return all_words, all_words_without_noise, noise

    def get_fraction(self, timestamps):
        """
        Counts timestamps proportion of speech period
        @param timestamps: time periods of some event
        @return: timestamps proportion of speech period
        """
        duration = 0
        for time_period in timestamps:
            duration += time_period[1] - time_period[0]
        return duration / self.duration

    def get_emotionality(self):
        """
        Analyses emotionality of file
        @return: list of lists of emotions probabilities and time period per which emotions are defined
        """
        emotions = AudioEmotions(self.path)
        audio_emotions = emotions.emotions_analysis()
        period = AudioEmotions.params["emotions_time_window"]
        return audio_emotions, period

    def get_filler_words(self):
        """
        Analyses presence of filler words
        @return: dicts with all filler words and phrases and with most common ones
        """
        filler_words = FillerWordsAndPhrases(self.cleaned_transcription)
        all_filler_words_dict, worst_words = filler_words.get_filler_words_final()
        print("all filler words:", all_filler_words_dict)
        print("worst words:", worst_words)
        return all_filler_words_dict, worst_words

    def get_speech_rate(self):
        """
        Analyses speech rate of speech
        @return: intervals with slow speech rate and their percentage of file duration
        """
        speech_rate = SpeechRate(self.word_arrays[1])
        intervals = speech_rate.unite_slow_speech_rate_intervals()
        print("pauses:", intervals, self.get_fraction(intervals))
        return intervals, self.get_fraction(intervals)

    def get_background_noise(self):
        """
        Analyses background noise presence
        @return: intervals with high background noise and their percentage of file duration
        """
        background_noise = BackgroundNoise(self.word_arrays[2])
        intervals = background_noise.get_high_noise_timestamps()
        print("background_noise:", intervals, self.get_fraction(intervals))
        return intervals, self.get_fraction(intervals)

    def get_intelligibility(self):
        """
        Analyses intelligibility of speech
        @return: approximate intelligibility (counts as percentage of high background noise and fast speech rate)
        """
        speech_rate = SpeechRate(self.word_arrays[1])
        _, fast_intervals = speech_rate.find_incorrect_speech_rate_intervals()
        fast_fraction = self.get_fraction(fast_intervals)
        noisy_intervals = BackgroundNoise(self.word_arrays[2]).get_high_noise_timestamps()
        noisy_fraction = self.get_fraction(noisy_intervals)
        return fast_fraction + noisy_fraction
