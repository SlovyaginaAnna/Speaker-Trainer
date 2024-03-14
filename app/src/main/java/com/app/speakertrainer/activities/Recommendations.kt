package com.app.speakertrainer.activities

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.data.RecommendationsInfo
import com.app.speakertrainer.databinding.ActivityRecomendationsBinding
import com.app.speakertrainer.modules.ApiManager
import com.app.speakertrainer.modules.Client

/**
 * Activity for representing recommendations.
 */
class Recommendations : AppCompatActivity() {
    private lateinit var binding: ActivityRecomendationsBinding
    private val apiManager = ApiManager(this)

    /**
     * Method called when the activity is created.
     * Initializes the binding to the layout and displays it on the screen.
     * Call method that sets statistics.
     *
     * @param savedInstanceState a Bundle object containing the previous state of the activity (if saved)
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityRecomendationsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.bNav.setOnItemSelectedListener {
            when (it.itemId) {
                R.id.home -> {
                    val intent = Intent(this, Home::class.java)
                    startActivity(intent)
                    finish()
                }
            }
            true
        }
        getRecommendationsInfo()
    }

    /**
     * Post request to get recommendations from server.
     */
    private fun getRecommendationsInfo() {
        apiManager.getRecommendations { info ->
            setInfo(info)
        }
    }

    /**
     * Set the fields of the [info] to the corresponding text views
     * and make corresponding text views visible.
     */
    private fun setInfo(info: RecommendationsInfo) {
        runOnUiThread {
            binding.apply {
                if (info.clean_speech != null) {
                    textViewClean.visibility = View.VISIBLE
                    textViewClean.text = Client.setCustomString(
                        "Чистота речи",
                        info.clean_speech.toString(), this@Recommendations
                    )
                }
                if (info.speech_rate != null) {
                    textViewRate.visibility = View.VISIBLE
                    textViewRate.text = Client.setCustomString(
                        "Доля плохого темпа речи",
                        info.speech_rate.toString(), this@Recommendations
                    )
                }
                if (info.background_noise != null) {
                    textViewNoise.visibility = View.VISIBLE
                    textViewNoise.text = Client.setCustomString(
                        "Доля времени с высоким шумом",
                        info.background_noise.toString(), this@Recommendations
                    )
                }
                if (info.intelligibility != null) {
                    textViewIntelligibility.visibility = View.VISIBLE
                    textViewIntelligibility.text = Client.setCustomString(
                        "Разборчивость речи",
                        info.intelligibility.toString(), this@Recommendations
                    )
                }
                if (info.clothes != null) {
                    textViewClothes.visibility = View.VISIBLE
                    textViewClothes.text = Client.setCustomString(
                        "Образ",
                        info.clothes.toString(), this@Recommendations
                    )
                }
                if (info.gestures != null) {
                    textViewGestures.visibility = View.VISIBLE
                    textViewGestures.text = Client.setCustomString(
                        "Жестикуляция",
                        info.gestures.toString(), this@Recommendations
                    )
                }
                if (info.angle != null) {
                    textViewAngle.visibility = View.VISIBLE
                    textViewAngle.text = Client.setCustomString(
                        "Доля времени неверного ракурса",
                        info.angle.toString(), this@Recommendations
                    )
                }
                if (info.glances != null) {
                    textViewGlances.visibility = View.VISIBLE
                    textViewGlances.text = Client.setCustomString(
                        "Доля времени некорректного направления взгляда",
                        info.glances.toString(), this@Recommendations
                    )
                }
                if (info.emotionality != null) {
                    textViewEmotionality.visibility = View.VISIBLE
                    textViewEmotionality.text = Client.setCustomString(
                        "Эмоциональность",
                        info.emotionality.toString(), this@Recommendations
                    )
                }
            }
        }
    }
}