from datetime import time
from .forms import *
from .models import *
from .computer_vision.computer_vision_subsystem import *
from .speech_processing.speech_processing_subsystem import *


class FileProcessingSystem:
    def __init__(self, file: FileInfo):
        """
        Initialization of file analysis class
        @param file: video file (FileInfo instance)
        """
        self.file_id = file.id
        # Creation of FileAnalysis instance to save analysis results
        file_analysis_form = FileAnalysisForm({"file_id": self.file_id})
        if file_analysis_form.is_valid():
            file_analysis_form.save()

        self.file_analysis = FileAnalysis.objects.get(file_id=self.file_id)
        self.speech_processing = SpeechProcessingSubsystem(file.file.path)
        self.computer_vision = ComputerVisionSubsystem(file.file.path)

    def save_timestamps_to_db(self, timestamps, type_choice):
        """
        Saves timestamps of low speech rate or high background noise to database
        @param timestamps: periods to be saved
        @param type_choice: 0 for background noise, 1 for speech rate
        """
        for time_period in timestamps:
            start_seconds, end_seconds = round(time_period[0]), round(time_period[1])
            # Transform seconds to time type
            start = time(hour=start_seconds // 3600, minute=start_seconds // 60, second=start_seconds % 60)
            end = time(hour=end_seconds // 3600, minute=end_seconds // 60, second=end_seconds % 60)
            form = FileTimestampsForm(
                {"file_id": self.file_id, "start": start, "end": end, "time_period_type": type_choice})
            if form.is_valid():
                form.save()
            else:
                print(form.errors)

    def get_transcription(self):
        """
        Translates and saves file transcription
        """
        self.speech_processing.speech_recognition()
        self.file_analysis.text = self.speech_processing.transcription["text"].strip()
        self.file_analysis.save()

    def get_emotionality(self):
        """
        Gets emotionality from audio and video subsystems, unites them and saves neutral emotion fraction
        """
        video_emotions = self.computer_vision.get_emotions_scores()
        audio_emotions, period = self.speech_processing.get_emotionality()
        neutral_count = 0
        for i in range(len(video_emotions)):
            video_emotion = video_emotions[i]
            if i // period < len(audio_emotions):
                audio_emotion = np.array(audio_emotions[i // period])
                emotion = (2 * video_emotion + audio_emotion) / 3
            else:
                emotion = video_emotion
            if np.argmax(emotion) == 0 or emotion[0] > 0.3:
                neutral_count += 1
        self.file_analysis.emotionality = neutral_count / len(video_emotions)
        self.file_analysis.save()

    def get_filler_words(self):
        """
        Gets filler words and phrases, saves them and their count per minute
        """
        all_filler_words, worst_filler_words = self.speech_processing.get_filler_words()
        for word in all_filler_words:
            data = {"file_id": self.file_id, "word_or_phrase": word, "occurrence": all_filler_words[word]}
            if word in worst_filler_words:
                data["most_common"] = True
            form = FillerWordsForm(data=data)
            if form.is_valid():
                form.save()
            else:
                print(form.errors)
        overall_count = sum(list(all_filler_words.values()))
        self.file_analysis.clean_speech = max(0, 10 - overall_count / (self.speech_processing.duration / 60))
        self.file_analysis.save()

    def get_speech_rate(self):
        """
        Gets and saves intervals with slow speech rate and their percentage
        """
        intervals, fraction = self.speech_processing.get_speech_rate()
        self.save_timestamps_to_db(intervals, 1)
        self.file_analysis.speech_rate = fraction
        self.file_analysis.save()

    def get_intelligibility(self):
        """
        Gets and saves intelligibility estimation
        """
        i_index = self.speech_processing.get_intelligibility()
        self.file_analysis.intelligibility = max(0, 1 - i_index)
        self.file_analysis.save()

    def get_background_noise(self):
        """
        Gets and saves intervals with high background noise and their percentage
        """
        intervals, fraction = self.speech_processing.get_background_noise()
        self.save_timestamps_to_db(intervals, 0)
        self.file_analysis.background_noise = fraction
        self.file_analysis.save()

    def get_incorrect_angle(self):
        """
        Gets and saves incorrect angle percentage
        """
        incorrect_angle = self.computer_vision.uncorrect_angle()
        self.file_analysis.angle = incorrect_angle
        self.file_analysis.save()

    def get_incorrect_glances(self):
        """
        Gets and saves incorrect glances percentage
        """
        glances = self.computer_vision.eye_tracking()
        self.file_analysis.glances = glances
        self.file_analysis.save()

    def get_gestures(self):
        """
        Gets and saves gesticulation level
        """
        gestures = self.computer_vision.get_gestures()
        self.file_analysis.gestures = gestures
        self.file_analysis.save()

    def get_clothes(self):
        """
        Gets and saves clothes suitability
        """
        clothes = self.computer_vision.get_clothes_estimation()
        self.file_analysis.clothes = int(clothes)
        self.file_analysis.save()
