package com.app.speakertrainer.activities

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.constance.Constance
import com.app.speakertrainer.constance.Constance.CORRECT_AUTH_STATUS
import com.app.speakertrainer.constance.Constance.EMAIL_ERROR
import com.app.speakertrainer.constance.Constance.PASSWORD_ERROR
import com.app.speakertrainer.data.FileInfo
import com.app.speakertrainer.data.FilesNumbers
import com.app.speakertrainer.data.Record
import com.app.speakertrainer.databinding.ActivityMainBinding
import com.app.speakertrainer.modules.Client
import com.google.gson.Gson
import okhttp3.Call
import okhttp3.Callback
import okhttp3.FormBody
import okhttp3.Response
import java.io.IOException

// Authorization activity.
class MainActivity : AppCompatActivity() {
    private lateinit var bindingClass: ActivityMainBinding
    private lateinit var launcher: ActivityResultLauncher<Intent>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        bindingClass = ActivityMainBinding.inflate(layoutInflater)
        setContentView(bindingClass.root)
    }

    fun onClickAuthorization(view: View) {
        if (!isFieldEmpty()) {
            val username = bindingClass.emailET.text.toString()
            val password = bindingClass.passwordET.text.toString()
            val requestBody = FormBody.Builder()
                .add("email", username)
                .add("password", password)
                .build()
            Client.client.postRequest("login/", requestBody, object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    toastResponse("Ошибка соединения" + e.message)
                }

                override fun onResponse(call: Call, response: Response) {
                    val responseBody = response.body
                    if (response.isSuccessful) {
                        if (response.header("status") == CORRECT_AUTH_STATUS) {
                            Client.token = response.body?.string().toString()
                            toastResponse("Успешная авторизация")
                            loadRecordList()
                            val intent = Intent(this@MainActivity, Home::class.java)
                            startActivity(intent)
                            finish()
                        } else {
                            when (response.header("status")) {
                                EMAIL_ERROR -> toastResponse("Неверная почта")
                                PASSWORD_ERROR -> toastResponse("Неверный пароль")
                            }
                        }
                    } else toastResponse("Ошибка авторизации")
                }
            })
        }
    }

    fun toastResponse(text: String) {
        runOnUiThread {
            Toast.makeText(this@MainActivity, text, Toast.LENGTH_SHORT).show()
        }
    }

    fun onClickRegistration(view: View) {
        val intent = Intent(this, RegistrationActivity::class.java)
        startActivity(intent)
        finish()
    }

    fun onClickChangePassword(view: View) {
        val intent = Intent(this, EmailEnterActivity::class.java)
        startActivity(intent)
        finish()
    }

    private fun isFieldEmpty(): Boolean {
        bindingClass.apply {
            if (emailET.text.isNullOrEmpty()) emailET.error =
                resources.getString(R.string.mustFillField)
            if (passwordET.text.isNullOrEmpty()) passwordET.error =
                resources.getString(R.string.mustFillField)
            return emailET.text.isNullOrEmpty() || passwordET.text.isNullOrEmpty()
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