package com.app.speakertrainer.data

// Data class containing analysis results.
data class ResponseResults(
    val clean_speech: String?,
    val speech_rate: String?,
    val background_noise: String?,
    val intelligibility: String?,
    val clothes: String?,
    val gestures: String?,
    val angle: String?,
    val glances: String?,
    val emotionality: String?,
    val low_speech_rate_timestamps: List<List<String>>?,
    val background_noise_timestamps: List<List<String>>?,
    val filler_words: List<String>?
)
