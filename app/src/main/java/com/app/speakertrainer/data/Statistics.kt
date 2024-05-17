package com.app.speakertrainer.data

/**
 * Data class for converting values from json. These values are used for drawing graphs.
 */
data class Statistics(
    val emotionality: List<Float>?, val neutral_emotionality: List<Float>?,
    val filler_words: List<String>?, val filler_words_percentage: List<Float>?
)
