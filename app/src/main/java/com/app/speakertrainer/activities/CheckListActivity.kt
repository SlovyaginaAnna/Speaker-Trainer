package com.app.speakertrainer.activities

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.constance.Constance
import com.app.speakertrainer.data.FileInfo
import com.app.speakertrainer.data.Record
import com.app.speakertrainer.databinding.ActivityCheckListBinding
import com.app.speakertrainer.modules.Client
import com.app.speakertrainer.modules.Client.recordList
import com.google.gson.Gson
import okhttp3.Call
import okhttp3.Callback
import okhttp3.FormBody
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.Response
import java.io.File
import java.io.IOException

class CheckListActivity : AppCompatActivity() {
    lateinit var binding: ActivityCheckListBinding
    private lateinit var videoUri: Uri


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

    fun toastResponse(text: String) {
        runOnUiThread {
            Toast.makeText(this@CheckListActivity, text, Toast.LENGTH_SHORT).show()
        }
    }

    private fun returnHome() {
        val intent = Intent(this, Home::class.java)
        startActivity(intent)
        finish()
    }

    private fun postData() {
        val file = File(videoUri.path)
        binding.apply {
            val requestBody: RequestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file",
                    file.name,
                    file.asRequestBody("video/*".toMediaTypeOrNull())
                )
                .addFormDataPart("token", Client.token)
                .addFormDataPart("emotionality", checkBoxEmotions.isChecked.toString())
                .addFormDataPart("clean_speech", checkBoxParasites.isChecked.toString())
                .addFormDataPart("speech_rate", checkBoxPauses.isChecked.toString())
                .addFormDataPart("background_noise", checkBoxNoise.isChecked.toString())
                .addFormDataPart("intelligibility", checkBoxIntelligibility.isChecked.toString())
                .addFormDataPart("gestures", checkBoxGesture.isChecked.toString())
                .addFormDataPart("clothing", checkBoxClothes.isChecked.toString())
                .addFormDataPart("angle", checkBoxAngle.isChecked.toString())
                .addFormDataPart("glances", checkBoxEye.isChecked.toString())
                .addFormDataPart("filename", videoName.text.toString())
                .build()
            Client.client.postRequest("upload_file/", requestBody, object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    toastResponse("Ошибка соединения" + e.message)
                }

                override fun onResponse(call: Call, response: Response) {
                    if (response.isSuccessful) {
                        when (val responseBodyString = response.body?.string()) {
                            "token_not_found_error" -> toastResponse("Неверный аккаунт")
                            "filename_error" -> toastResponse("Неверное имя файла")
                            "parsing_error" -> toastResponse("Ошибка передачи данных")
                            else -> {
                                if (response.header("status") == "File is successfully uploaded.") {
                                    if (responseBodyString != null) {
                                        saveData(responseBodyString)
                                    }
                                }
                            }
                        }
                    } else toastResponse("Ошибка передачи видео")//saveData(response)
                }
            })
        }

    }


    private fun saveData(id: String) {
        val img = getImg(id)
        val info = getInfo(id)
        if (img != null && info != null) {
            val record: Record = Record(
                img, info.filename,
                info.datetime, id.toInt()
            )
            recordList.add(record)
            val intent = Intent(this@CheckListActivity, AnalysisResults::class.java)
            intent.putExtra("index", (recordList.size - 1).toString())
            startActivity(intent)
            finish()
        }
    }

    private fun isFieldEmpty(): Boolean {
        binding.apply {
            if (videoName.text.isNullOrEmpty()) videoName.error =
                resources.getString(R.string.mustFillField)
            return videoName.text.isNullOrEmpty()
        }
    }

    private fun getImg(id: String): Bitmap? {
        var bitmap: Bitmap? = null
        val client = Client.client.getClient()
        val requestBody = FormBody.Builder()
            .add("token", Client.token)
            .add("file_id", id)
            .build()
        val request = Client.client.getRequest("archive/file_image/", requestBody)
        try {
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    if (response.header("status") == Constance.UNKNOWN_ERROR) {
                        toastResponse("Ошибка загрузки. Пользователь не найден")
                    } else {
                        val inputStream = response.body?.byteStream()
                        bitmap = BitmapFactory.decodeStream(inputStream)
                    }
                } else toastResponse("Ошибка загрузки фото")
            }
        } catch (ex: IOException) {
            toastResponse("Ошибка соединения")
        }
        return bitmap
    }

    private fun getInfo(id: String): FileInfo? {
        var info: FileInfo? = null
        val requestBody = FormBody.Builder()
            .add("token", Client.token)
            .add("file_id", id)
            .build()
        val client = Client.client.getClient()
        val request = Client.client.getRequest("archive/file_info/", requestBody)
        try {
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    if (response.header("status") == Constance.UNKNOWN_ERROR) {
                        toastResponse("Ошибка загрузки. Пользователь не найден")
                    } else {
                        val gson = Gson()
                        info = gson.fromJson(
                            response.body?.string(),
                            FileInfo::class.java
                        )
                    }
                } else toastResponse("Ошибка загрузки информации")
            }
        } catch (ex: IOException) {
            toastResponse("Ошибка соединения")
        }
        return info
    }

}