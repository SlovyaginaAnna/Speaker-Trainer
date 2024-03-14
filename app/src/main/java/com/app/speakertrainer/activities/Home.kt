package com.app.speakertrainer.activities

import android.app.Activity
import android.app.AlertDialog
import android.content.ContentValues.TAG
import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.Nullable
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.constance.Constance
import com.app.speakertrainer.constance.Constance.REQUEST_VIDEO_CAPTURE
import com.app.speakertrainer.data.Statistics
import com.app.speakertrainer.databinding.ActivityHomeBinding
import com.app.speakertrainer.modules.Client
import com.app.speakertrainer.modules.Client.graphEntries
import com.app.speakertrainer.modules.Client.lineGraphEntries
import com.app.speakertrainer.modules.Client.pieEntries
import com.app.speakertrainer.modules.Client.token
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.data.PieData
import com.github.mikephil.charting.data.PieDataSet
import com.github.mikephil.charting.data.PieEntry
import com.google.gson.Gson
import com.gowtham.library.utils.LogMessage
import com.gowtham.library.utils.TrimVideo
import okhttp3.Call
import okhttp3.Callback
import okhttp3.FormBody
import okhttp3.Response
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale


class Home : AppCompatActivity() {
    private lateinit var selectBtn: Button
    private lateinit var binding: ActivityHomeBinding
    private val PICK_IMAGE = 100
    private val TRIM_VIDEO = 102
    private var videoUri: Uri? = null
    val startForResult =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result: ActivityResult ->
            if (result.resultCode == Activity.RESULT_OK &&
                result.data != null
            ) {
                val uri = Uri.parse(TrimVideo.getTrimmedVideoPath(result.data))
                Log.d(TAG, "Trimmed path:: " + videoUri + "\n")
                val intent = Intent(this@Home, CheckListActivity::class.java)
                intent.putExtra("uri", uri.toString())
                startActivity(intent)
                finish()
            } else
                LogMessage.v("videoTrimResultLauncher data is null")
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHomeBinding.inflate(layoutInflater)
        setContentView(binding.root)
        selectBtn = binding.loadBtn
        getStatistics()
    }

    private fun getStatistics() {
        if (Client.recordList.size > 0) {
            val requestBody = FormBody.Builder()
                .add("token", token)
                .build()
            Client.client.postRequest("statistics/", requestBody, object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    toastResponse("Ошибка соединения")
                    drawLineChart()
                    drawPieChart()
                }

                override fun onResponse(call: Call, response: Response) {
                    if (response.isSuccessful) {
                        if (response.header("status") != Constance.UNKNOWN_ERROR) {
                            val gson = Gson()
                            val statistics = gson.fromJson(
                                response.body?.string(),
                                Statistics::class.java
                            )
                            setGraphs(statistics)
                        } else toastResponse("Ошибка загрузки статистики. Обратитесь в поддержку")
                    } else toastResponse("Ошибка загрузки статистики")
                    drawLineChart()
                    drawPieChart()
                }
            })
        } else {
            drawLineChart()
            drawPieChart()
        }
    }

    private fun setGraphs(statistics: Statistics) {
        if (statistics.speech_rate != null) {
            graphEntries = ArrayList<Entry>()
            for (i in statistics.speech_rate.indices) {
                graphEntries.add(Entry(i.toFloat(), statistics.speech_rate[i]))
            }
            Client.lineText = "Ваш прогресс"
        }
        if (statistics.angle != null) {
            lineGraphEntries = ArrayList<Entry>()
            for (i in statistics.angle.indices) {
                lineGraphEntries.add(Entry(i.toFloat(), statistics.angle[i]))
            }
            Client.lineText = "Ваш прогресс"
        }
        if (statistics.filler_words != null && statistics.filler_words_percentage != null) {
            pieEntries = ArrayList<PieEntry>()
            for (i in statistics.filler_words.indices) {
                pieEntries.add(
                    PieEntry(
                        statistics.filler_words_percentage[i],
                        statistics.filler_words[i]
                    )
                )
            }
            Client.pieText = "Ваш прогресс"
        }
    }

    private fun drawPieChart() {
        runOnUiThread {
            val pieChart = binding.pieChart
            pieChart.setHoleColor(Color.parseColor("#13232C"))
            val entries = pieEntries
            val colors = mutableListOf<Int>()
            for (i in entries.indices) {
                val randomColor = Color.rgb((0..255).random(), (0..255).random(), (0..255).random())
                colors.add(randomColor)
            }
            val pieDataSet = PieDataSet(entries, "Слова паразиты")
            pieDataSet.colors = colors
            val pieData = PieData(pieDataSet)
            pieChart.data = pieData
            pieChart.description.text = Client.pieText
            pieChart.setEntryLabelColor(Color.WHITE)
            pieChart.description.textColor = Color.WHITE
            pieChart.legend.textColor = Color.WHITE
            pieChart.invalidate()
        }
    }

    private fun drawLineChart() {
        runOnUiThread {
            val lineChart = binding.lineChart
            val lineDataSetRate = LineDataSet(graphEntries, "Темп речи")
            lineDataSetRate.color = Color.BLUE
            lineDataSetRate.valueTextSize = 12f
            val lineDataSetAngle = LineDataSet(lineGraphEntries, "Неверный ракурс")
            lineDataSetAngle.color = Color.RED
            lineDataSetAngle.valueTextSize = 12f
            lineDataSetRate.setDrawValues(false)
            lineDataSetAngle.setDrawValues(false)
            val lineData = LineData(lineDataSetRate, lineDataSetAngle)
            lineChart.data = lineData
            lineChart.description.text = Client.lineText
            lineChart.description.textColor = Color.WHITE
            lineChart.legend.textColor = Color.WHITE
            lineChart.xAxis.textColor = Color.WHITE
            lineChart.axisLeft.textColor = Color.WHITE
            lineChart.axisRight.textColor = Color.WHITE
            lineChart.invalidate()
        }
    }

    fun onClickLoadVideo(view: View) {
        val intent = Intent()
        intent.action = Intent.ACTION_PICK
        intent.type = "video/*"
        startActivityForResult(intent, PICK_IMAGE)
    }

    fun onClickStartRecording(view: View) {
        dispatchTakeVideoIntent()
    }

    fun onClickExit(view: View) {
        val alertDialog = AlertDialog.Builder(this)
            .setTitle("Выход из аккаунта")
            .setMessage("Вы действительно хотите выйти из аккаунта?")
            .setPositiveButton("Да") { dialog, _ ->
                logoutUser()
                dialog.dismiss()
            }
            .setNegativeButton("Отмена") { dialog, _ ->
                dialog.dismiss()
            }
            .create()

        alertDialog.show()
    }

    private fun logoutUser() {
        val requestBody = FormBody.Builder()
            .add("token", token)
            .build()
        Client.client.postRequest("logout/", requestBody, object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                toastResponse("Ошибка соединения")
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    if (response.header("status") == Constance.LOGOUT_SUCCESS) {
                        val intent = Intent(this@Home, MainActivity::class.java)
                        startActivity(intent)
                        Client.resetData()
                        finish()
                    } else {
                        toastResponse("Неизвестная ошибка. Свяжитесь с тех. поддержкой")
                    }
                } else toastResponse("Неизвестная ошибка. Повторите позже")
            }
        })
    }

    fun toastResponse(text: String) {
        runOnUiThread {
            Toast.makeText(this@Home, text, Toast.LENGTH_SHORT).show()
        }
    }

    private fun createVideoFile(): File {
        val timeStamp: String = SimpleDateFormat(
            "yyyyMMdd_HHmmss",
            Locale.getDefault()
        ).format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_MOVIES)
        return File.createTempFile(
            "VIDEO_${timeStamp}_",
            ".mp4",
            storageDir
        )
    }

    private fun dispatchTakeVideoIntent() {
        val takeVideoIntent = Intent(MediaStore.ACTION_VIDEO_CAPTURE)
        startActivityForResult(takeVideoIntent, REQUEST_VIDEO_CAPTURE)
    }

    private fun startRecording() {
        val videoFile = createVideoFile()
        videoUri = Uri.fromFile(videoFile)

        val intent = Intent(MediaStore.ACTION_VIDEO_CAPTURE)
        intent.putExtra(MediaStore.EXTRA_OUTPUT, videoUri)

        if (intent.resolveActivity(packageManager) != null) {
            startActivityForResult(intent, REQUEST_VIDEO_CAPTURE)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, @Nullable data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK) {
            if (data != null) {
                val selectVideo = data.data
                videoUri = selectVideo
                trimVideo(selectVideo.toString())
            }
        }
        if (requestCode == REQUEST_VIDEO_CAPTURE && resultCode == Activity.RESULT_OK) {
            if (data != null) {
                val selectVideo = data.data
                videoUri = selectVideo
                trimVideo(selectVideo.toString())
            }
        }
    }

    private fun trimVideo(videoUri: String?) {
        TrimVideo.activity(videoUri)
            .setHideSeekBar(true)
            .start(this, startForResult)
    }


    fun onClickArchieve(view: View) {
        val intent = Intent(this, PreviewVideoActivity::class.java)
        startActivity(intent)
        finish()
    }

    fun onClickRecommendations(vie: View) {
        val intent = Intent(this, Recommendations::class.java)
        startActivity(intent)
        finish()
    }

}