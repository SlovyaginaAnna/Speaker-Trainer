package com.app.speakertrainer.data

/**
 * Data class containing analysis results.
 */
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
    val filler_words: List<String>?,
    val analyzed_segment_len: Int,
    val clean_speech_timestamps: List<Float>?,
    val speech_rate_timestamps: List<Float>?,
    val background_noise_timestamps: List<Float>?,
    val intelligibility_timestamps: List<Float>?,
    val gestures_timestamps: List<Float>?,
    val angle_timestamps: List<Float>?,
    val glances_timestamps: List<Float>?,
    val emotionality_timestamps: List<Float>?
)
