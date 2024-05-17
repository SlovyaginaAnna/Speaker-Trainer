package com.app.speakertrainer.modules

import android.content.Context
import android.graphics.Color
import android.text.Spannable
import android.text.SpannableString
import androidx.core.content.res.ResourcesCompat
import com.app.speakertrainer.R
import com.app.speakertrainer.data.Record
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.PieEntry

/**
 * This Class contains the `Client` object which holds network client, data, and text entries.
 */
object Client {
    val client: NetworkClient = NetworkClient()
    var token: String = ""
    var graphEntries = arrayListOf(
        Entry(0f, 10f), Entry(1f, 20f), Entry(2f, 15f),
        Entry(3f, 30f), Entry(4f, 25f)
    )
    var lineGraphEntries = arrayListOf(
        Entry(0f, 4f), Entry(1f, 5f), Entry(2f, 11f),
        Entry(3f, 3f), Entry(4f, 21f), Entry(5f, 26f)
    )
    var pieEntries = arrayListOf(PieEntry(1f, 25f), PieEntry(2f, 75f))
    var recordList = ArrayList<Record>()
    var pieText = "Здесь будет отображаться ваш прогресс" // Your progress will be here
    var lineText = "Здесь будет отображаться ваш прогресс"
    val colors = arrayListOf(Color.rgb(64, 224, 208), Color.rgb(60, 179, 113),
        Color.rgb(70, 130, 180),
        Color.rgb(169, 169, 169), Color.rgb(0, 128, 128))

    /**
     * The `resetData` function resets all the data fields to their initial values.
     */
    fun resetData() {
        token = ""
        recordList = ArrayList<Record>()
        graphEntries = arrayListOf(
            Entry(0f, 10f), Entry(1f, 20f), Entry(2f, 15f),
            Entry(3f, 30f), Entry(4f, 25f), Entry(5f, 35f)
        )
        pieEntries = arrayListOf(PieEntry(1f, 25f), PieEntry(2f, 75f))
        lineGraphEntries = arrayListOf(
            Entry(0f, 4f), Entry(1f, 5f), Entry(2f, 11f),
            Entry(3f, 3f), Entry(4f, 21f), Entry(5f, 26f)
        )
        pieText = "Здесь будет отображаться ваш прогресс"
        lineText = "Здесь будет отображаться ваш прогресс"
    }

    /**
     * The `setCustomString` function creates a SpannableString with custom typefaces for different parts.
     *
     * @param boldText is string to make bold.
     * @param regularText is string to concatenate with bold text.
     */
    fun setCustomString(boldText: String, regularText: String, context: Context): SpannableString {
        val boldTypeface = ResourcesCompat.getFont(context, R.font.comfortaa_bold)
        val regularTypeface = ResourcesCompat.getFont(context, R.font.comfortaa_light)

        val spannableString = SpannableString("$boldText: $regularText")
        spannableString.setSpan(
            TypefaceSpan(boldTypeface!!), 0, boldText.length + 1,
            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
        )
        spannableString.setSpan(
            TypefaceSpan(regularTypeface!!), boldText.length + 2,
            spannableString.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
        )

        return spannableString
    }
}