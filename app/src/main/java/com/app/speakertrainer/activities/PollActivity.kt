package com.app.speakertrainer.activities

import android.content.Intent
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.SeekBar
import android.widget.Toast
import com.app.speakertrainer.R
import com.app.speakertrainer.data.Record
import com.app.speakertrainer.databinding.ActivityCheckListBinding
import com.app.speakertrainer.databinding.ActivityPollBinding
import com.app.speakertrainer.modules.ApiManager
import com.app.speakertrainer.modules.Client
import java.io.File

class PollActivity : AppCompatActivity() {
    lateinit var binding: ActivityPollBinding
    private lateinit var videoUri: Uri
    private val apiManager = ApiManager(this)
    private var emotions = false
    private var parasites = false
    private var pauses = false
    private var noise = false
    private var intelligibility = false
    private var gestures = false
    private var clothes = false
    private var angle = false
    private var eye = false
    private lateinit var videoName: String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPollBinding.inflate(layoutInflater)
        setContentView(binding.root)
        videoUri = Uri.parse(intent.getStringExtra("uri"))
        emotions = intent.getStringExtra("checkBoxEmotions").toBoolean()
        parasites = intent.getStringExtra("checkBoxParasites").toBoolean()
        pauses = intent.getStringExtra("checkBoxPauses").toBoolean()
        noise = intent.getStringExtra("checkBoxNoise").toBoolean()
        intelligibility = intent.getStringExtra("checkBoxIntelligibility").toBoolean()
        gestures = intent.getStringExtra("checkBoxGesture").toBoolean()
        clothes = intent.getStringExtra("checkBoxClothes").toBoolean()
        angle = intent.getStringExtra("checkBoxAngle").toBoolean()
        eye = intent.getStringExtra("checkBoxEye").toBoolean()
        videoName = intent.getStringExtra("videoName").toString()
        binding.bNav.setOnItemSelectedListener {
            when (it.itemId) {
                R.id.back -> {
                    val intent = Intent(this, CheckListActivity::class.java)
                    intent.putExtra("uri", videoUri.toString())
                    startActivity(intent)
                    finish()
                }
                R.id.home -> {
                    val intent = Intent(this, Home::class.java)
                    startActivity(intent)
                    finish()
                }
                R.id.forward -> {
                    postData()
                }
            }
            true
        }
        binding.seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                binding.textView4.text = "Длина анализируемых фрагментов: $progress" //
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {
            }

            override fun onStopTrackingTouch(seekBar: SeekBar?) {
            }
        })
        setVisibility()
    }

    /**
     * Activate checkboxes.
     */
    private fun setVisibility() {
        binding.apply {
            if (emotions) {
                textEmotion.visibility = View.VISIBLE
                checkBoxJoy.visibility = View.VISIBLE
                checkAnger.visibility = View.VISIBLE
                checkBoxDisguist.visibility = View.VISIBLE
                checkBoxNeutral.visibility = View.VISIBLE
                checkBoxSad.visibility = View.VISIBLE
                checkBoxSurprise.visibility = View.VISIBLE
            }
            if (gestures) {
                textGesticulation.visibility = View.VISIBLE
                checkBoxActive.visibility = View.VISIBLE
                checkBoxNotActive.visibility = View.VISIBLE
                checkBoxOptimal.visibility = View.VISIBLE
            }
        }
    }

    /**
     * Send request to server for analysing loaded video record.
     */
    private fun postData() {

        val file = File(videoUri.path)
        binding.apply {
            if (checkBoxActive.isChecked && checkBoxOptimal.isChecked && checkBoxNotActive.isChecked) {
                Toast.makeText(this@PollActivity, "Выберите желательную скорость жестикуляции", Toast.LENGTH_SHORT).show()
            } else {
                val booleanList = listOf(checkBoxJoy.isChecked, checkAnger.isChecked, checkBoxDisguist.isChecked,
                    checkBoxNeutral.isChecked, checkBoxSad.isChecked, checkBoxSurprise.isChecked)
                val gestureList = listOf(checkBoxNotActive.isChecked, checkBoxOptimal.isChecked, checkBoxActive.isChecked)
                apiManager.postData(file,emotions, parasites, pauses, noise, intelligibility, gestures,
                    clothes, angle, eye, videoName, seekBar.progress, switch1.isChecked, booleanList, gestureList) {
                        responseResults -> saveData(responseResults)
                }
            }
        }
    }

    /**
     * Save record instance with id [id].
     */
    private fun saveData(id: String) {
        apiManager.getImg(id) { image ->
            apiManager.getInfo(id) {info ->
                val record = Record(
                    image, info.filename,
                    info.datetime, id.toInt()
                )
                Client.recordList.add(record)
                val intent = Intent(this@PollActivity, AnalysisResults::class.java)
                intent.putExtra("index", (Client.recordList.size - 1).toString())
                startActivity(intent)
                finish()
            }
        }
    }
}