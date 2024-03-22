class BackgroundNoise:
    params = {
        "noise_time_window": 30,  # Size of time window to view noise percentage
        "noise_percentage": 0.45,  # Minimal noise percentage
    }

    def __init__(self, noise):
        """
        Initialization of background noise analysis class
        @param noise: timestamps with background noise, list of two-element lists
        """
        self.noise = noise

    def get_high_noise_timestamps(self):
        """
        Searches most noisy periods with the help of floating window
        @return: Most noisy periods, list of two-element lists
        """
        high_noise_timestamps = []
        if len(self.noise) == 0:
            return high_noise_timestamps
        start_idx, end_idx = 0, 1
        noise_sum = self.noise[0][1] - self.noise[0][0]
        while end_idx < len(self.noise):
            if self.noise[end_idx][1] - self.noise[start_idx][0] < \
                    self.params["noise_time_window"]:
                noise_sum += self.noise[end_idx][1] - self.noise[end_idx][0]
                end_idx += 1
                continue
            # Check if the percentage of noise is larger than parameter
            if noise_sum / (self.noise[end_idx][1] - self.noise[start_idx][0]) > \
                    self.params["noise_percentage"]:
                # If period intersects with previous one - they are united
                if len(high_noise_timestamps) > 0 and high_noise_timestamps[-1][1] > \
                        self.noise[start_idx][0]:
                    high_noise_timestamps[-1][1] = self.noise[end_idx][1]
                else:
                    high_noise_timestamps.append(
                        [self.noise[start_idx][0], self.noise[end_idx][1]])
            noise_sum -= (self.noise[start_idx][1] - self.noise[start_idx][0])
            start_idx += 1
        return high_noise_timestamps
