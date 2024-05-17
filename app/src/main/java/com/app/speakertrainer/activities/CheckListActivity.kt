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
                        startPollActivity()
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
     * Start poll activity.
     */
    private fun startPollActivity() {
        binding.apply {
                val intent = Intent(this@CheckListActivity, PollActivity::class.java)
                intent.putExtra("uri", videoUri.toString())
                intent.putExtra("checkBoxEmotions", checkBoxEmotions.isChecked.toString())
                intent.putExtra("checkBoxParasites", checkBoxParasites.isChecked.toString())
                intent.putExtra("checkBoxPauses", checkBoxPauses.isChecked.toString())
                intent.putExtra("checkBoxNoise", checkBoxNoise.isChecked.toString())
                intent.putExtra("checkBoxIntelligibility", checkBoxIntelligibility.isChecked.toString())
                intent.putExtra("checkBoxGesture", checkBoxGesture.isChecked.toString())
                intent.putExtra("checkBoxClothes", checkBoxClothes.isChecked.toString())
                intent.putExtra("checkBoxAngle", checkBoxAngle.isChecked.toString())
                intent.putExtra("checkBoxEye", checkBoxEye.isChecked.toString())
                intent.putExtra("videoName", videoName.text.toString())
                startActivity(intent)
                finish()
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