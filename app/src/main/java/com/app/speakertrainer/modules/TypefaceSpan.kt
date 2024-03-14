package com.app.speakertrainer.modules

import android.graphics.Typeface
import android.text.TextPaint
import android.text.style.MetricAffectingSpan

/**
 * Class is a custom implementation of MetricAffectingSpan that applies a custom typeface to text.
 */
class TypefaceSpan(private val typeface: Typeface) : MetricAffectingSpan() {
    /**
     * Method sets typeface.
     *
     * @param paint a TextPaint to change typeface.
     */
    override fun updateDrawState(paint: TextPaint) {
        paint.typeface = typeface
    }

    /**
     * Method updates measure state.
     *
     * @param paint a TextPaint to change typeface.
     */
    override fun updateMeasureState(paint: TextPaint) {
        paint.typeface = typeface
    }
}