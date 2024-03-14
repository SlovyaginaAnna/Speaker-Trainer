package com.app.speakertrainer.data

// Data class for converting values from json. These values are used for drawing graphs.
data class Statistics(
    val speech_rate: List<Float>?, val angle: List<Float>?,
    val filler_words: List<String>?, val filler_words_percentage: List<Float>?
)
