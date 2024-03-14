package com.app.speakertrainer.activities

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.constance.Constance
import com.app.speakertrainer.data.FileInfo
import com.app.speakertrainer.data.FilesNumbers
import com.app.speakertrainer.data.Record
import com.app.speakertrainer.databinding.ActivityPasswordEnterBinding
import com.app.speakertrainer.modules.Client
import com.google.gson.Gson
import okhttp3.Call
import okhttp3.Callback
import okhttp3.FormBody
import okhttp3.Response
import java.io.IOException

class PasswordEnterActivity : AppCompatActivity() {
    private lateinit var bindingClass: ActivityPasswordEnterBinding
    private var email = ""
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        bindingClass = ActivityPasswordEnterBinding.inflate(layoutInflater)
        setContentView(bindingClass.root)
        email = intent.getStringExtra("email").toString()
    }

    fun onClickAuthorizationScreen(view: View) {
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        finish()
    }

    fun onClickRestorePassword(view: View) {
        if (!isFieldEmpty()) {
            if (isPasswordCorrect()) {
                val password = bindingClass.passwordInputLayout.text.toString()
                val requestBody = FormBody.Builder()
                    .add("email", email)
                    .add("password", password)
                    .build()
                Client.client.postRequest("password_update/", requestBody, object : Callback {
                    override fun onFailure(call: Call, e: IOException) {
                        toastResponse("Ошибка соединения")
                    }

                    override fun onResponse(call: Call, response: Response) {
                        val responseBody = response.body
                        if (response.isSuccessful) {
                            if (response.header("status") == Constance.PASSWORD_CHANGED) {
                                Client.token = responseBody?.string().toString()
                                toastResponse("Успешная смена пароля")
                                loadRecordList()
                                val intent = Intent(
                                    this@PasswordEnterActivity,
                                    Home::class.java
                                )
                                startActivity(intent)
                                finish()
                            } else {
                                toastResponse("Ошибка сервера. Повторите позже")
                            }
                        } else toastResponse("Ошибка смены пароля")
                    }
                })
            }
        }
    }

    fun toastResponse(text: String) {
        runOnUiThread {
            Toast.makeText(this@PasswordEnterActivity, text, Toast.LENGTH_SHORT).show()
        }
    }

    private fun isFieldEmpty(): Boolean {
        bindingClass.apply {
            if (passwordInputLayout.text.isNullOrEmpty())
                passwordInputLayout.error = resources.getString(R.string.mustFillField)
            if (pasAgainInputLayout.text.isNullOrEmpty())
                pasAgainInputLayout.error = resources.getString(R.string.mustFillField)
            return pasAgainInputLayout.text.isNullOrEmpty() || passwordInputLayout.text.isNullOrEmpty()

        }
    }

    private fun isPasswordCorrect(): Boolean {
        bindingClass.apply {
            if (passwordInputLayout.text.toString() != pasAgainInputLayout.text.toString())
                pasAgainInputLayout.error = "Пароли не совпадают"
            return passwordInputLayout.text.toString() == pasAgainInputLayout.text.toString()
        }
    }

    private fun loadFiles(members: FilesNumbers) {
        if (members.num_of_files > 0) {
            for (id in members.file_ids) {
                val img = getImg(id)
                val info = getInfo(id)
                if (img != null && info != null) {
                    val record: Record = Record(
                        img, info.filename,
                        info.datetime, id
                    )
                    Client.recordList.add(record)
                }
            }
        }
    }

    private fun getImg(id: Int): Bitmap? {
        var bitmap: Bitmap? = null
        val client = Client.client.getClient()
        val requestBody = FormBody.Builder()
            .add("token", Client.token)
            .add("file_id", id.toString())
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

    private fun getInfo(id: Int): FileInfo? {
        var info: FileInfo? = null
        val requestBody = FormBody.Builder()
            .add("token", Client.token)
            .add("file_id", id.toString())
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

    private fun loadRecordList() {
        val requestBody = FormBody.Builder()
            .add("token", Client.token)
            .build()
        val client = Client.client.getClient()
        val request = Client.client.getRequest("archive/number_of_files/", requestBody)
        try {
            client.newCall(request).execute().use { response ->
                if (response.isSuccessful) {
                    if (response.header("status") == Constance.FILES_FOUND) {
                        try {
                            val gson = Gson()
                            val responseData = gson.fromJson(
                                response.body?.string(),
                                FilesNumbers::class.java
                            )
                            loadFiles(responseData)
                        } catch (ex: Exception) {
                            Log.d("tag", ex.message.toString())
                            toastResponse(ex.message.toString())
                        }

                    } else toastResponse("Ошибка загрузки. Пользователь не найден")
                } else toastResponse("Ошибка загрузки статистики")
            }
        } catch (ex: IOException) {
            toastResponse("Ошибка соединения")
        }
    }
}