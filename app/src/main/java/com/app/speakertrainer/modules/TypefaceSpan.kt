package com.app.speakertrainer.modules

import android.graphics.Typeface
import android.text.TextPaint
import android.text.style.MetricAffectingSpan

// The `TypefaceSpan` class is a custom implementation of MetricAffectingSpan that applies a custom typeface to text.
class TypefaceSpan(private val typeface: Typeface) : MetricAffectingSpan() {
    override fun updateDrawState(paint: TextPaint) {
        paint.typeface = typeface
    }

    override fun updateMeasureState(paint: TextPaint) {
        paint.typeface = typeface
    }
}