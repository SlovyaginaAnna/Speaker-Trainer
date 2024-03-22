from django.db import models


class Authentication(models.Model):
    """
    Class with user info - email, password, unique id, generated token
    """
    id = models.IntegerField(primary_key=True)
    email = models.EmailField(max_length=100, unique=True)
    password = models.CharField(max_length=100)
    token = models.CharField(max_length=256, null=True, blank=True)

    def __str__(self):
        return self.email


class FileInfo(models.Model):
    """
    Class with base file info - path, user's id, time of receipt, chosen analysis parameters
    """
    id = models.IntegerField(primary_key=True)
    user_id = models.ForeignKey("Authentication", to_field="id", on_delete=models.CASCADE)
    file = models.FileField(upload_to="uploads/")
    filename = models.CharField(max_length=100)
    timestamp = models.DateTimeField()

    emotionality = models.BooleanField(default=False)
    clean_speech = models.BooleanField(default=False)
    speech_rate = models.BooleanField(default=False)
    background_noise = models.BooleanField(default=False)
    intelligibility = models.BooleanField(default=False)

    gestures = models.BooleanField(default=False)
    clothes = models.BooleanField(default=False)
    angle = models.BooleanField(default=False)
    glances = models.BooleanField(default=False)


class FileAnalysis(models.Model):
    """
    Class with results of file analysis - enums or real numbers
    """
    UNKNOWN = -1
    LOW = 0
    NORMAL = 1
    HIGH = 2
    STATUS_CHOICES = (
        (UNKNOWN, "UNKNOWN"),
        (LOW, "Low"),
        (NORMAL, "Normal"),
        (HIGH, "High"),
    )
    file_id = models.ForeignKey("FileInfo", to_field="id", on_delete=models.CASCADE)

    clean_speech = models.DecimalField(max_digits=15, decimal_places=10, blank=True, default=-1)  # слова-паразиты
    speech_rate = models.DecimalField(max_digits=12, decimal_places=10, blank=True, default=-1)  # паузы
    background_noise = models.DecimalField(max_digits=12, decimal_places=10, blank=True, default=-1)  # шум
    intelligibility = models.IntegerField(choices=STATUS_CHOICES, blank=True, default=UNKNOWN)  # разборчивость речи

    clothes = models.IntegerField(choices=STATUS_CHOICES, blank=True, default=UNKNOWN)  # одежда
    gestures = models.IntegerField(choices=STATUS_CHOICES, blank=True, default=UNKNOWN)  # жесты
    angle = models.DecimalField(max_digits=12, decimal_places=10, blank=True, default=-1)  # ракурс
    glances = models.DecimalField(max_digits=12, decimal_places=10, blank=True, default=-1)  # взгляд

    emotionality = models.DecimalField(max_digits=12, decimal_places=10, blank=True, default=-1)  # эмоциональность
    text = models.TextField(null=True, blank=True)


class FileTimestamps(models.Model):
    """
    Class of timestamps for background noise or pause intervals
    """
    NOISE = 0
    PAUSE = 1
    TYPE_CHOICES = (
        (NOISE, "NOISE"),
        (PAUSE, "PAUSE"),
    )
    file_id = models.ForeignKey("FileInfo", to_field="id", on_delete=models.CASCADE)
    start = models.TimeField()
    end = models.TimeField()
    time_period_type = models.IntegerField(choices=TYPE_CHOICES)


class FillerWords(models.Model):
    """
    Class for filler words / phrases - their occurrence and whether they are common for a file
    """
    file_id = models.ForeignKey("FileInfo", to_field="id", on_delete=models.CASCADE)
    word_or_phrase = models.CharField(max_length=100)
    occurrence = models.IntegerField(default=0)
    most_common = models.BooleanField(default=False)
