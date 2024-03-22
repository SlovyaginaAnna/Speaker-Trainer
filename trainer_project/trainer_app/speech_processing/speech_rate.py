class SpeechRate:
    params = {
        "pause_time_window": 30,  # Size of time window to view pauses
        "pause_percentage": 0.35,  # Minimal noise percentage
        "speech_rate_time_window": 60,  # Size of time window to speech rate
        "speech_rate_min_word_count": 60,  # Minimal number of words in time window for normal speech rate
        "speech_rate_max_word_count": 130,  # Maximal number of words in time window for normal speech rate
        "rules": {"word": 0.5, "punct_mark": 0.75, ".": 1, "?": 5, "!": 3},  # Allowed pauses between words
    }

    def __init__(self, all_words_without_noise):
        """
        Initialization of speech rate analysis class
        @param all_words_without_noise: list of dicts with words and their start and end timestamps
        """
        self.all_words_without_noise = all_words_without_noise

    def find_pauses(self):
        """
        Finds all pauses longer than allowed
        @return: list of two-element lists with pauses timestamps
        """
        rules = self.params["rules"]
        pauses = []
        start_idx = 0
        end_idx = 1
        while end_idx < len(self.all_words_without_noise) - 1:
            silence_start = self.all_words_without_noise[start_idx]["end"]
            silence_end = self.all_words_without_noise[end_idx]["start"]
            if self.all_words_without_noise[start_idx]["text"][-1].isalpha():
                pause_type = rules["word"]
            elif self.all_words_without_noise[start_idx]["text"][-1] in rules:
                pause_type = rules[self.all_words_without_noise[start_idx]["text"][-1]]
            else:
                pause_type = rules["punct_mark"]
            if silence_end - silence_start > pause_type:
                pauses.append([silence_start, silence_end])
            start_idx = end_idx
            end_idx += 1
        return pauses

    def find_pause_intervals(self, pauses):
        """
        Searches periods with high pauses percentage with the help of floating window
        @param pauses: pauses intervals, list of two-element lists
        @return: list of two-element lists with pause intervals timestamps
        """
        intervals = []
        if len(pauses) == 0:
            return intervals
        start_idx, end_idx = 0, 0
        summary = pauses[0][1] - pauses[0][0]
        while end_idx < len(pauses):
            while end_idx < len(pauses) - 1 and pauses[end_idx][1] - pauses[start_idx][0] < \
                    self.params["pause_time_window"]:
                end_idx += 1
                summary += pauses[end_idx][1] - pauses[end_idx][0]
            if pauses[end_idx][1] - pauses[start_idx][0] < self.params["pause_time_window"]:
                break
            # Check if the percentage of pauses is larger than parameter
            if summary / (pauses[end_idx][1] - pauses[start_idx][0]) > self.params["pause_percentage"]:
                # If period intersects with previous one - they are united
                if len(intervals) > 0 and intervals[-1][-1] > pauses[start_idx][0]:
                    intervals[-1][-1] = pauses[end_idx][1]
                else:
                    intervals.append([pauses[start_idx][0], pauses[end_idx][1]])
            summary -= pauses[start_idx][1] - pauses[start_idx][0]
            start_idx += 1
        return intervals

    def find_incorrect_speech_rate_intervals(self):
        """
        Searches intervals with too fast or slow speech rate
        @return: two lists with two-element list each - periods with too fast or slow speech rate
        """
        fast_intervals = []
        slow_intervals = []
        word_count = 1
        start = self.all_words_without_noise[0]["start"]
        end = self.all_words_without_noise[0]["end"]
        start_idx = 0
        end_idx = 1
        while end_idx < len(self.all_words_without_noise):
            if end - start < self.params["speech_rate_time_window"]:
                end = self.all_words_without_noise[end_idx]["end"]
                end_idx += 1
                word_count += 1
            else:
                # if word count is too small or too large - append time interval to corresponding list
                if word_count < self.params["speech_rate_min_word_count"]:
                    if len(slow_intervals) > 0 and slow_intervals[-1][1] >= start:
                        slow_intervals[-1][1] = end
                    else:
                        slow_intervals.append([start, end])
                elif word_count > self.params["speech_rate_max_word_count"]:
                    if len(fast_intervals) > 0 and fast_intervals[-1][1] >= start:
                        fast_intervals[-1][1] = end
                    else:
                        fast_intervals.append([start, end])
                start_idx += 1
                start = self.all_words_without_noise[start_idx]["start"]
                word_count -= 1
        return slow_intervals, fast_intervals

    def unite_slow_speech_rate_intervals(self):
        """
        Unites two lists of intervals: with pauses and with slow speech rate
        @return: list of two-element lists with slow speech rate intervals timestamps
        """
        speech_rate_results, _ = self.find_incorrect_speech_rate_intervals()
        pauses = self.find_pauses()
        pause_intervals = self.find_pause_intervals(pauses)
        final_intervals = []
        speech_rate_idx, pause_idx = 0, 0
        while speech_rate_idx < len(speech_rate_results) and pause_idx < len(pause_intervals):
            sr_start, sr_end = speech_rate_results[speech_rate_idx]
            pause_start, pause_end = pause_intervals[pause_idx][0], pause_intervals[pause_idx][1]
            if sr_start <= pause_start:
                if sr_end <= pause_start:
                    speech_rate_idx += 1
                elif pause_start < sr_end <= pause_end:
                    final_intervals.append([pause_start, sr_end])
                    speech_rate_idx += 1
                else:
                    final_intervals.append([pause_start, pause_end])
                    pause_idx += 1
            elif pause_start <= sr_start <= pause_end:
                if sr_end <= pause_end:
                    final_intervals.append([sr_start, sr_end])
                    speech_rate_idx += 1
                else:
                    final_intervals.append([sr_start, pause_end])
                    pause_idx += 1
            else:
                pause_idx += 1
        return final_intervals
