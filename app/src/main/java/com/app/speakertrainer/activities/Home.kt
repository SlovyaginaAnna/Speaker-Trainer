package com.app.speakertrainer.activities

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.ContentValues.TAG
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.app.speakertrainer.data.Statistics
import com.app.speakertrainer.databinding.ActivityHomeBinding
import com.app.speakertrainer.modules.ApiManager
import com.app.speakertrainer.modules.Client
import com.app.speakertrainer.modules.Client.graphEntries
import com.app.speakertrainer.modules.Client.lineGraphEntries
import com.app.speakertrainer.modules.Client.pieEntries
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.data.PieData
import com.github.mikephil.charting.data.PieDataSet
import com.github.mikephil.charting.data.PieEntry
import com.gowtham.library.utils.LogMessage
import com.gowtham.library.utils.TrimVideo

/**
 * Activity that represents home screen.
 */
class Home : AppCompatActivity() {
    private lateinit var selectBtn: Button
    private lateinit var binding: ActivityHomeBinding
    private var videoUri: Uri? = null
    private val apiManager = ApiManager(this)
    private val REQUEST_CAMERA_PERMISSION = 0
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
    val startLoadVideo = registerForActivityResult(ActivityResultContracts.StartActivityForResult())
    { result: ActivityResult ->
        if (result.resultCode == Activity.RESULT_OK) {
            if (result.data != null) {
                val selectVideo = result.data!!.data
                videoUri = selectVideo
                trimVideo(selectVideo.toString())
            }
        }
    }
    val startVideoRecord = registerForActivityResult(ActivityResultContracts.StartActivityForResult())
    { result: ActivityResult ->
        if (result.resultCode == Activity.RESULT_OK) {
            if (result.data != null) {
                val selectVideo = result.data!!.data
                videoUri = selectVideo
                trimVideo(selectVideo.toString())
            }
        }
    }

    /**
     * Method called when the activity is created.
     * Initializes the binding to the layout and displays it on the screen.
     * Call method that sets statistics.
     *
     * @param savedInstanceState a Bundle object containing the previous state of the activity (if saved)
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHomeBinding.inflate(layoutInflater)
        setContentView(binding.root)
        selectBtn = binding.loadBtn
        getStatistics()
    }

    /**
     * Post request to get arrays with values for graphs.
     * Call methods for drawing charts.
     */
    private fun getStatistics() {
        if (Client.recordList.size > 0) {
            apiManager.getGraphStatistics { statistics->
                setGraphs(statistics)
                drawLineChart()
                drawPieChart()
            }
        } else {
            drawLineChart()
            drawPieChart()
        }
    }

    /**
     * Method set entries values with data got from server.
     *
     * @param statistics a Statistics object which contains arrays of entries for drawing charts.
     */
    private fun setGraphs(statistics: Statistics) {
        if (!statistics.emotionality.isNullOrEmpty()) {
            graphEntries = ArrayList<Entry>()
            for (i in statistics.emotionality.indices) {
                graphEntries.add(Entry(i.toFloat(), statistics.emotionality[i]))
            }
            Client.lineText = "Ваш прогресс" // Progress
        }
        if (!statistics.neutral_emotionality.isNullOrEmpty()) {
            lineGraphEntries = ArrayList<Entry>()
            for (i in statistics.neutral_emotionality.indices) {
                lineGraphEntries.add(Entry(i.toFloat(), statistics.neutral_emotionality[i]))
            }
            Client.lineText = "Ваш прогресс"
        }
        if (!statistics.filler_words.isNullOrEmpty() && !statistics.filler_words_percentage.isNullOrEmpty()) {
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

    /**
     * Method set pie charts settings.
     * Method draw pie chart.
     */
    private fun drawPieChart() {
        runOnUiThread {
            val pieChart = binding.pieChart
            pieChart.setHoleColor(Color.parseColor("#13232C"))
            var entries = pieEntries
            val colors = mutableListOf<Int>()
            // Set colors for drawing pie chart.
            for (i in entries.indices) {
                colors.add(Client.colors[i])
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

    /**
     * Method set line charts settings.
     * Method draw line chart.
     */
    private fun drawLineChart() {
        runOnUiThread {
            val lineChart = binding.lineChart
            val lineDataSetRate = LineDataSet(graphEntries, "Доля нежелательных эмоций")
            lineDataSetRate.color = Client.colors[0]
            lineDataSetRate.valueTextSize = 12f
            val lineDataSetAngle = LineDataSet(lineGraphEntries, "Эмоциональность")
            lineDataSetAngle.color = Client.colors[1]
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

    /**
     * Method opens your phone files to choose video for analysing.
     */
    fun onClickLoadVideo(view: View) {
        val intent = Intent()
        intent.action = Intent.ACTION_PICK
        intent.type = "video/*"
        startLoadVideo.launch(intent)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                val takeVideoIntent = Intent(MediaStore.ACTION_VIDEO_CAPTURE)
                startVideoRecord.launch(takeVideoIntent)
            } else {
                Toast.makeText(this, "Доступ к камере не разрешен", Toast.LENGTH_SHORT).show()
            }
        }
    }
    /**
     * Method opens video camera for recording video.
     */
    fun onClickStartRecording(view: View) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA_PERMISSION)
        } else {
            val takeVideoIntent = Intent(MediaStore.ACTION_VIDEO_CAPTURE)
            startVideoRecord.launch(takeVideoIntent)
        }

    }

    /**
     * Method creates alert dialog to ensure in users actions.
     */
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

    /**
     * Method post request to logout and reset data.
     * Return to authorization activity.
     */
    private fun logoutUser() {
        apiManager.logoutUser() {loggedOut ->
            if (loggedOut) {
                val intent = Intent(this@Home, MainActivity::class.java)
                startActivity(intent)
                finish()
            }
        }
    }

    /**
     * Method starts trim video activity.
     */
    private fun trimVideo(videoUri: String?) {
        TrimVideo.activity(videoUri)
            .setHideSeekBar(true)
            .start(this, startForResult)
    }


    /**
     * Method open archieve and finish home activity.
     */
    fun onClickArchieve(view: View) {
        val intent = Intent(this, PreviewVideoActivity::class.java)
        startActivity(intent)
        finish()
    }

    /**
     * Method open recommendations and finish home activity.
     */
    fun onClickRecommendations(view: View) {
        val intent = Intent(this, Recommendations::class.java)
        intent.putExtra("adress", "all_files")
        startActivity(intent)
        finish()
    }

}