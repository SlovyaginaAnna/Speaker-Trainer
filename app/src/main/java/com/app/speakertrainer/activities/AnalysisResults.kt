package com.app.speakertrainer.activities

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.app.speakertrainer.databinding.ActivityAnalysisResultsBinding
import android.net.Uri
import android.view.View
import android.widget.MediaController
import com.app.speakertrainer.modules.Client
import com.app.speakertrainer.modules.Client.recordList
import com.app.speakertrainer.R
import com.app.speakertrainer.data.ResponseResults
import com.app.speakertrainer.modules.ApiManager

/**
 * Activity for demonstrating analysis results for chosen video record.
 */
class AnalysisResults : AppCompatActivity() {
    private lateinit var binding: ActivityAnalysisResultsBinding
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
     * Set the fields of the [info] to the corresponding text views
     * and make corresponding text views visible.
     */
    private fun setInfo(info: ResponseResults) {
        runOnUiThread {
            binding.apply {
                if (info.clean_speech != null) {
                    textViewClean.visibility = View.VISIBLE
                    textViewClean.text = Client.setCustomString(
                        "Чистота речи",
                        info.clean_speech.toString(), this@AnalysisResults
                    )
                }
                if (info.speech_rate != null) {
                    textViewRate.visibility = View.VISIBLE
                    textViewRate.text = Client.setCustomString(
                        "Доля плохого темпа речи",
                        info.speech_rate.toString(), this@AnalysisResults
                    )
                }
                if (info.background_noise != null) {
                    textViewNoise.visibility = View.VISIBLE
                    textViewNoise.text = Client.setCustomString(
                        "Доля времени с высоким шумом",
                        info.background_noise.toString(), this@AnalysisResults
                    )
                }
                if (info.intelligibility != null) {
                    textViewIntelligibility.visibility = View.VISIBLE
                    textViewIntelligibility.text = Client.setCustomString(
                        "Разборчивость речи",
                        info.intelligibility.toString(), this@AnalysisResults
                    )
                }
                if (info.clothes != null) {
                    textViewClothes.visibility = View.VISIBLE
                    textViewClothes.text = Client.setCustomString(
                        "Образ",
                        info.clothes.toString(), this@AnalysisResults
                    )
                }
                if (info.gestures != null) {
                    textViewGestures.visibility = View.VISIBLE
                    textViewGestures.text = Client.setCustomString(
                        "Жестикуляция",
                        info.gestures.toString(), this@AnalysisResults
                    )
                }
                if (info.angle != null) {
                    textViewAngle.visibility = View.VISIBLE
                    textViewAngle.text = Client.setCustomString(
                        "Доля времени неверного ракурса",
                        info.angle.toString(), this@AnalysisResults
                    )
                }
                if (info.glances != null) {
                    textViewGlances.visibility = View.VISIBLE
                    textViewGlances.text = Client.setCustomString(
                        "Доля времени некорректного направления взгляда",
                        info.glances.toString(), this@AnalysisResults
                    )
                }
                if (info.emotionality != null) {
                    textViewEmotionality.visibility = View.VISIBLE
                    textViewEmotionality.text = Client.setCustomString(
                        "Эмоциональность",
                        info.emotionality.toString(), this@AnalysisResults
                    )
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
            val mediaController = MediaController(this)
            // Attach mediaController to the bottom of the video recording.
            mediaController.setAnchorView(binding.videoView)
            binding.videoView.setMediaController(mediaController)
            binding.videoView.start()
        }
    }
}