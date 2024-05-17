package com.app.speakertrainer.activities

import android.content.Intent
import android.graphics.Color
import android.media.MediaPlayer
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.app.speakertrainer.databinding.ActivityAnalysisResultsBinding
import android.net.Uri
import android.text.SpannableString
import android.view.View
import android.widget.MediaController
import android.widget.RelativeLayout
import android.widget.TextView
import android.widget.Toast
import android.widget.VideoView
import com.app.speakertrainer.modules.Client
import com.app.speakertrainer.modules.Client.recordList
import com.app.speakertrainer.R
import com.app.speakertrainer.data.ResponseResults
import com.app.speakertrainer.modules.ApiManager
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.highlight.Highlight
import com.github.mikephil.charting.listener.OnChartValueSelectedListener

/**
 * Activity for demonstrating analysis results for chosen video record.
 */
class AnalysisResults : AppCompatActivity() {
    private lateinit var binding: ActivityAnalysisResultsBinding
    private lateinit var mediaController: MediaController
    private var index: Int = -1
    private lateinit var videoUri: Uri
    private val apiManager = ApiManager(this)

    /**
     * Method called when the activity is created.
     * Initializes the binding to the layout and displays it on the screen.
     * Set actions for menu buttons.
     * Call function for setting analysis results.
     *
     * @param savedInstanceState a Bundle object containing the previous state of the activity (if saved)
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAnalysisResultsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.frameLayout.visibility = View.VISIBLE
        index = intent.getStringExtra("index")?.toInt() ?: -1
        // Set menu actions.
        binding.bNav.setOnItemSelectedListener {
            when (it.itemId) {
                R.id.back -> {
                    val intent = Intent(this, PreviewVideoActivity::class.java)
                    startActivity(intent)
                    finish()
                }
                R.id.home -> {
                    val intent = Intent(this, Home::class.java)
                    startActivity(intent)
                    finish()
                }
                R.id.forward -> {
                    val id = recordList[index].index
                    val intent = Intent(this, Recommendations::class.java)
                    intent.putExtra("adress", "one_file")
                    intent.putExtra("id", id.toString())
                    intent.putExtra("index", index.toString())
                    startActivity(intent)
                    finish()
                }
            }
            true
        }
        setResults(index)
    }

    /**
     *  Get the results of analysis of the video recording in the list under [index].
     */
    private fun setResults(index: Int) {
        if (index != -1) {
            val id = recordList[index].index
            apiManager.getAnalysisInfo(id) { responseResults ->
                setInfo(responseResults)
            }
            apiManager.getVideo(id) {responseResults ->
                videoUri = responseResults
                setVideo()
            }
        } else apiManager.toastResponse("Ошибка. Повторите позднее")
    }

    /**
     * Method set line charts settings.
     * Method draw line chart.
     */
    private fun drawLineChart(chart: LineChart, entries: ArrayList<Entry>) {
        runOnUiThread {
            val lineDataSet = LineDataSet(entries, "Analysis results") // Результаты анализа
            lineDataSet.color = Client.colors[0]
            lineDataSet.valueTextSize = 12f
            lineDataSet.setDrawValues(false)
            val lineData = LineData(lineDataSet)
            chart.data = lineData
            chart.isDragEnabled = true
            chart.setScaleEnabled(true)
            chart.setPinchZoom(true)
            chart.description.isEnabled = false
            chart.setOnChartValueSelectedListener(object : OnChartValueSelectedListener {
                override fun onValueSelected(e: Entry?, h: Highlight?) {
                    val mediaPlayer: MediaPlayer? = binding.videoView?.let { videoView ->
                        try {
                            val mediaPlayerField = VideoView::class.java.getDeclaredField("mMediaPlayer")
                            mediaPlayerField.isAccessible = true
                            mediaPlayerField.get(videoView) as MediaPlayer
                        } catch (e: NoSuchFieldException) {
                            e.printStackTrace()
                            null
                        } catch (e: IllegalAccessException) {
                            e.printStackTrace()
                            null
                        }
                    }

                    mediaPlayer?.let { mediaPlayer ->
                        val newPositionMillis = (e?.x?.toInt() ?: 1) * 1000
                        mediaPlayer.seekTo(newPositionMillis)
                    }
                }
                override fun onNothingSelected() {}
            })
        }
    }

    /**
     * Draw line charts and activate them.
     */
    private fun setVisibility(text: TextView, textToSet: SpannableString, chart: LineChart,
                              values: List<Float>?, len: Int) {
        text.visibility = View.VISIBLE
        text.text = textToSet
        if (values != null) {
            val graphEntries = ArrayList<Entry>()
            for (i in values.indices) {
                graphEntries.add(Entry(i.toFloat() * len, values[i]))
            }
            drawLineChart(chart, graphEntries)
            text.setOnClickListener {
                if (chart.visibility == View.GONE) {
                    chart.visibility = View.VISIBLE
                } else {
                    chart.visibility = View.GONE
                }
            }
        }
    }

    /**
     * Set the fields of the [info] to the corresponding text views
     * and make corresponding text views visible.
     */
    private fun setInfo(info: ResponseResults) {
        runOnUiThread {
            binding.apply {
                if (info.clean_speech != null) {
                    val textToSet = Client.setCustomString("Чистота речи",
                        info.clean_speech.toString(), this@AnalysisResults
                    )
                    textViewClean.visibility = View.VISIBLE
                    textViewClean.text = textToSet
                    val result = info.filler_words?.joinToString(separator = ", ")
                    textViewFillerWords.text = result
                    textViewClean.setOnClickListener {
                        if (textViewFillerWords.visibility == View.GONE) {
                            textViewFillerWords.visibility = View.VISIBLE
                        } else {
                            textViewFillerWords.visibility = View.GONE
                        }
                    }
                }
                if (info.speech_rate != null) {
                    val textToSet = Client.setCustomString("Доля плохого темпа речи",
                        info.speech_rate.toString(), this@AnalysisResults
                    )
                    setVisibility(textViewRate, textToSet, lineChartRate, info.speech_rate_timestamps,
                        info.analyzed_segment_len)
                }
                if (info.background_noise != null) {
                    textViewNoise.visibility = View.VISIBLE
                    val textToSet = Client.setCustomString("Доля времени с высоким шумом", //
                        info.background_noise.toString(), this@AnalysisResults
                    )
                    setVisibility(textViewNoise, textToSet, lineChartNoise, info.background_noise_timestamps,
                        info.analyzed_segment_len)
                }
                if (info.intelligibility != null) {
                    textViewIntelligibility.visibility = View.VISIBLE
                    val textToSet = Client.setCustomString("Разборчивость речи", //
                        info.intelligibility.toString(), this@AnalysisResults
                    )
                    setVisibility(textViewIntelligibility, textToSet, lineChartIntelligibility,
                        info.intelligibility_timestamps, info.analyzed_segment_len)
                }
                if (info.clothes != null) {
                    textViewClothes.visibility = View.VISIBLE
                    textViewClothes.text = Client.setCustomString("Образ", //
                        info.clothes.toString(), this@AnalysisResults
                    )
                }
                if (info.gestures != null) {
                    textViewGestures.visibility = View.VISIBLE
                    val textToSet = Client.setCustomString("Жестикуляция", //
                        info.gestures.toString(), this@AnalysisResults
                    )
                    setVisibility(textViewGestures, textToSet, lineChartGestures, info.gestures_timestamps,
                        info.analyzed_segment_len)
                }
                if (info.angle != null) {
                    textViewAngle.visibility = View.VISIBLE
                    val textToSet = Client.setCustomString("Доля времени неверного ракурса", //
                        info.angle.toString(), this@AnalysisResults
                    )
                    setVisibility(textViewAngle, textToSet, lineChartAngle, info.angle_timestamps,
                        info.analyzed_segment_len)
                }
                if (info.glances != null) {
                    textViewGlances.visibility = View.VISIBLE
                    val textToSet = Client.setCustomString("Доля времени некорректного направления взгляда", //
                        info.glances.toString(), this@AnalysisResults
                    )
                    setVisibility(textViewGlances, textToSet, lineChartGlances, info.glances_timestamps,
                        info.analyzed_segment_len)
                }
                if (info.emotionality != null) {
                    textViewEmotionality.visibility = View.VISIBLE
                    val textToSet = Client.setCustomString("Доля неподходящих эмоций", //
                        info.emotionality.toString(), this@AnalysisResults
                    )
                    setVisibility(textViewEmotionality, textToSet, lineChartEmotionality, info.emotionality_timestamps,
                        info.analyzed_segment_len)
                }
            }
        }
    }

    /**
     * Set obtained footage to the video view.
     * Set media controller to the video view.
     * Start displaying the video record.
     */
    private fun setVideo() {
        runOnUiThread {
            binding.videoView.setVideoURI(videoUri)
            mediaController = MediaController(this)
            // Attach mediaController to the bottom of the video recording.
            mediaController.setAnchorView(binding.videoView)
            binding.videoView.setMediaController(mediaController)
            binding.videoView.start()
        }
    }
}