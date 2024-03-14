package com.app.speakertrainer.activities

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.data.Record
import com.app.speakertrainer.databinding.ActivityCheckListBinding
import com.app.speakertrainer.modules.ApiManager
import com.app.speakertrainer.modules.Client.recordList
import java.io.File

/**
 * Activity for choosing components for analysing.
 */
class CheckListActivity : AppCompatActivity() {
    lateinit var binding: ActivityCheckListBinding
    private lateinit var videoUri: Uri
    private val apiManager = ApiManager(this)

    /**
     * Method called when the activity is created.
     * Initializes the binding to the layout and displays it on the screen.
     * Set actions for menu buttons.
     * Call function for setting up check boxes.
     *
     * @param savedInstanceState a Bundle object containing the previous state of the activity (if saved)
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCheckListBinding.inflate(layoutInflater)
        setContentView(binding.root)
        videoUri = Uri.parse(intent.getStringExtra("uri"))
        setCheckListeners()
        binding.bNav.setOnItemSelectedListener {
            when (it.itemId) {
                R.id.back -> {
                    returnHome()
                }
                R.id.home -> {
                    returnHome()
                }
                R.id.forward -> {
                    if (!isFieldEmpty()) {
                        postData()
                    }
                }
            }
            true
        }
    }

    /**
     * Set up check boxes.
     */
    private fun setCheckListeners() {
        binding.apply {
            checkBoxAll.setOnCheckedChangeListener { _, isChecked ->
                checkBoxVisual.isChecked = isChecked
                checkBoxAllAudio.isChecked = isChecked
                checkBoxEmotions.isChecked = isChecked
            }
            checkBoxVisual.setOnCheckedChangeListener { _, isChecked ->
                checkBoxClothes.isChecked = isChecked
                checkBoxGesture.isChecked = isChecked
                checkBoxAngle.isChecked = isChecked
                checkBoxEye.isChecked = isChecked
            }
            checkBoxAllAudio.setOnCheckedChangeListener { _, isChecked ->
                checkBoxIntelligibility.isChecked = isChecked
                checkBoxPauses.isChecked = isChecked
                checkBoxParasites.isChecked = isChecked
                checkBoxNoise.isChecked = isChecked
            }
        }
    }

    /**
     * Start home activity.
     * Finish this activity.
     */
    private fun returnHome() {
        val intent = Intent(this, Home::class.java)
        startActivity(intent)
        finish()
    }

    /**
     * Send request to server for analysing loaded video record.
     */
    private fun postData() {
        val file = File(videoUri.path)
        binding.apply {
            apiManager.postData(file,checkBoxEmotions.isChecked, checkBoxParasites.isChecked,
                checkBoxPauses.isChecked, checkBoxNoise.isChecked, checkBoxIntelligibility.isChecked,
                checkBoxGesture.isChecked, checkBoxClothes.isChecked, checkBoxAngle.isChecked,
                checkBoxEye.isChecked, videoName.text.toString()) { responseResults ->
                saveData(responseResults)
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
                recordList.add(record)
                val intent = Intent(this@CheckListActivity, AnalysisResults::class.java)
                intent.putExtra("index", (recordList.size - 1).toString())
                startActivity(intent)
                finish()
            }
        }
    }

    /**
     * Check if field with video name is empty.
     *
     * @return if video name field is empty.
     */
    private fun isFieldEmpty(): Boolean {
        binding.apply {
            if (videoName.text.isNullOrEmpty()) videoName.error =
                resources.getString(R.string.mustFillField)
            return videoName.text.isNullOrEmpty()
        }
    }

}