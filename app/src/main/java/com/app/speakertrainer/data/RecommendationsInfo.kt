package com.app.speakertrainer.data

/**
 * Data class containing recommendations after analysing.
 */
data class RecommendationsInfo(
    val clean_speech: String?,
    val speech_rate: String?,
    val background_noise: String?,
    val intelligibility: String?,
    val clothes: String?,
    val gestures: String?,
    val angle: String?,
    val glances: String?,
    val emotionality: String?
)
