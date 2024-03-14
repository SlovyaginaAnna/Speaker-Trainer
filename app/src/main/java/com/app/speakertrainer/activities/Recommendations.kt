package com.app.speakertrainer.activities

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.constance.Constance
import com.app.speakertrainer.data.RecommendationsInfo
import com.app.speakertrainer.databinding.ActivityRecomendationsBinding
import com.app.speakertrainer.modules.Client
import com.google.gson.Gson
import okhttp3.Call
import okhttp3.Callback
import okhttp3.FormBody
import okhttp3.Response
import java.io.IOException

class Recommendations : AppCompatActivity() {
    private lateinit var binding: ActivityRecomendationsBinding
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

    private fun getRecommendationsInfo() {
        val requestBody = FormBody.Builder()
            .add("token", Client.token)
            .build()
        Client.client.postRequest("recommendations/", requestBody, object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                toastResponse("Ошибка соединения")
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    if (response.header("status") == Constance.UNKNOWN_ERROR) {
                        toastResponse("Ошибка загрузки файла. Пользователь не найден")
                    } else {
                        val gson = Gson()
                        val info = gson.fromJson(
                            response.body?.string(),
                            RecommendationsInfo::class.java
                        )
                        setInfo(info)
                    }
                } else toastResponse("Ошибка загрузки информации")
            }
        })
    }

    fun toastResponse(text: String) {
        runOnUiThread {
            Toast.makeText(this@Recommendations, text, Toast.LENGTH_SHORT).show()
        }
    }

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