package com.app.speakertrainer.activities

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Toast
import com.app.speakertrainer.constance.Constance
import com.app.speakertrainer.databinding.ActivityAnalysisResultsBinding
import com.google.gson.Gson
import okhttp3.FormBody
import java.io.File
import java.io.IOException
import android.net.Uri
import android.view.View
import android.widget.MediaController
import com.app.speakertrainer.modules.Client
import com.app.speakertrainer.modules.Client.recordList
import com.app.speakertrainer.R
import com.app.speakertrainer.data.ResponseResults
import okhttp3.Call
import okhttp3.Callback
import okhttp3.Response

// Activity for displaying results after analysing video record.
class AnalysisResults : AppCompatActivity() {
    private lateinit var binding: ActivityAnalysisResultsBinding
    private var index: Int = -1
    private lateinit var videoUri: Uri

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
                    binding.frameLayout.visibility = View.GONE
                    val intent = Intent(this, PreviewVideoActivity::class.java)
                    startActivity(intent)
                    finish()
                }

                R.id.home -> {
                    val intent = Intent(this, Home::class.java)
                    startActivity(intent)
                    binding.frameLayout.visibility = View.GONE
                    finish()
                }
            }
            true
        }
        setResults(index)
    }

    // Get information from server and set it on activity.
    private fun setResults(index: Int) {
        if (index != -1) {
            val id = recordList[index].index
            getVideo(id)
            getAnalysisInfo(id)
        } else toastResponse("Ошибка. Повторите позднее")
    }


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

    // Set video and media controller.
    private fun setVideo() {
        runOnUiThread {
            binding.videoView.setVideoURI(videoUri)
            val mediaController = MediaController(this)
            mediaController.setAnchorView(binding.videoView)
            binding.videoView.setMediaController(mediaController)
            binding.videoView.start()
        }
    }

    // Function post request to server to get results/
    private fun getAnalysisInfo(id: Int) {
        val requestBody = FormBody.Builder()
            .add("token", Client.token)
            .add("file_id", id.toString())
            .build()
        Client.client.postRequest("file_statistics/", requestBody, object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                toastResponse("Ошибка соединения")
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    if (response.header("status") == Constance.UNKNOWN_ERROR) {
                        toastResponse("Ошибка загрузки файла. Пользователь не найден")
                    } else {
                        val gson = Gson()
                        var info = gson.fromJson(
                            response.body?.string(),
                            ResponseResults::class.java
                        )
                        setInfo(info)
                    }
                } else toastResponse("Ошибка загрузки информации")
            }
        })
    }

    // Function post request to server to get analysed video.
    private fun getVideo(id: Int) {
        val requestBody = FormBody.Builder()
            .add("token", Client.token)
            .add("file_id", id.toString())
            .build()

        Client.client.postRequest("modified_file/", requestBody, object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                toastResponse("Ошибка соединения")
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    if (response.header("status") == Constance.UNKNOWN_ERROR) {
                        toastResponse("Ошибка загрузки видео. Обратитесь в поддержку")
                    } else {
                        val inputStream = response.body?.byteStream()
                        // Create temporary file to set video.
                        val videoFile = File(cacheDir, "temp_video.mp4")
                        videoFile.outputStream().use { fileOut ->
                            inputStream?.copyTo(fileOut)
                        }
                        videoUri = Uri.fromFile(videoFile)
                        setVideo()
                    }
                } else toastResponse("Ошибка загрузки видео")
            }
        })
    }

    fun toastResponse(text: String) {
        runOnUiThread {
            Toast.makeText(this@AnalysisResults, text, Toast.LENGTH_SHORT).show()
        }
    }
}