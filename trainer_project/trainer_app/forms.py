from django import forms
from .models import *


class AuthenticationForm(forms.ModelForm):
    class Meta:
        model = Authentication
        fields = "__all__"


class FileInfoForm(forms.ModelForm):
    class Meta:
        model = FileInfo
        fields = "__all__"


class FileAnalysisForm(forms.ModelForm):
    class Meta:
        model = FileAnalysis
        fields = "__all__"


class FileTimestampsForm(forms.ModelForm):
    class Meta:
        model = FileTimestamps
        fields = "__all__"


class FillerWordsForm(forms.ModelForm):
    class Meta:
        model = FillerWords
        fields = "__all__"
