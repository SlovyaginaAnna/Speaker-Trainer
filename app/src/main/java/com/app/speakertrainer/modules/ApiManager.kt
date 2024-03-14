package com.app.speakertrainer.modules

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import android.widget.Toast
import com.app.speakertrainer.constance.Constance
import com.app.speakertrainer.data.FileInfo
import com.app.speakertrainer.data.FilesNumbers
import com.app.speakertrainer.data.RecommendationsInfo
import com.app.speakertrainer.data.ResponseResults
import com.app.speakertrainer.data.Statistics
import com.google.gson.Gson
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.IOException

/**
 * Class containing all requests to server.
 *
 * @param context is an activity for representing requests results to user.
 */
class ApiManager(private val context: Context) {
    val gson = Gson()

    /**
     * Method gets analysis results from server.
     *
     * @param id is an id of video which was analysed.
     */
    fun getAnalysisInfo(id: Int, onResponse: (ResponseResults) -> Unit) {
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
                        val info = gson.fromJson(response.body?.string(), ResponseResults::class.java)
                        onResponse.invoke(info)
                    }
                } else {
                    toastResponse("Ошибка загрузки информации")
                }
            }
        })
    }

    /**
     * Method gets changed video from server.
     *
     * @param id is an id of video which was analysed.
     */
    fun getVideo(id: Int, onVideoReceived: (Uri) -> Unit) {
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
                        val videoFile = File(context.cacheDir, "temp_video.mp4")
                        videoFile.outputStream().use { fileOut ->
                            inputStream?.copyTo(fileOut)
                        }
                        val videoUri = Uri.fromFile(videoFile)
                        onVideoReceived.invoke(videoUri)
                    }
                } else {
                    toastResponse("Ошибка загрузки видео")
                }
            }
        })
    }

    /**
     * Method send video and options for analysing.
     *
     * @param file a video file for analysing.
     */
    fun postData(file: File, emotionality: Boolean, parasites: Boolean, pauses: Boolean, noise: Boolean,
                 intellig: Boolean, gesture: Boolean, clothes: Boolean, angle: Boolean,
                 glance: Boolean, filename: String, onSaveData: (String) -> Unit) {
        val requestBody: RequestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file",
                file.name,
                file.asRequestBody("video/*".toMediaTypeOrNull())
            )
            .addFormDataPart("token", Client.token)
            .addFormDataPart("emotionality", emotionality.toString())
            .addFormDataPart("clean_speech", parasites.toString())
            .addFormDataPart("speech_rate", pauses.toString())
            .addFormDataPart("background_noise", noise.toString())
            .addFormDataPart("intelligibility", intellig.toString())
            .addFormDataPart("gestures", gesture.toString())
            .addFormDataPart("clothing", clothes.toString())
            .addFormDataPart("angle", angle.toString())
            .addFormDataPart("glances", glance.toString())
            .addFormDataPart("filename", filename)
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
                                    onSaveData.invoke(responseBodyString)
                                }
                            }
                        }
                    }
                } else toastResponse("Ошибка передачи видео")
            }
        })
    }

    /**
     * Method gets preview of the video from server.
     *
     * @param id is an id of video which was analysed.
     */
    fun getImg(id: String, onImageReceived: (Bitmap) -> Unit){
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
                        val bitmap = BitmapFactory.decodeStream(inputStream)
                        onImageReceived.invoke(bitmap)
                    }
                } else toastResponse("Ошибка загрузки фото")
            }
        } catch (ex: IOException) {
            toastResponse("Ошибка соединения")
        }
    }

    /**
     * Method gets information about video file from server.
     *
     * @param id is an id of video which was analysed.
     */
    fun getInfo(id: String, onInfoReceived: (FileInfo) -> Unit){
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
                        val info = gson.fromJson(
                            response.body?.string(),
                            FileInfo::class.java
                        )
                        onInfoReceived.invoke(info)
                    }
                } else toastResponse("Ошибка загрузки информации")
            }
        } catch (ex: IOException) {
            toastResponse("Ошибка соединения")
        }
    }

    /**
     * Method gets numbers that were send to users email.
     *
     * @param email is an email for sending numbers.
     */
    fun getEmailCheckNumbers(email: String, onNumbersReceived: (String) -> Unit) {
        val requestBody = FormBody.Builder()
            .add("email", email)
            .build()
        Client.client.postRequest("password_recovery/", requestBody, object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                toastResponse("Ошибка соединения" + e.message)
            }
            override fun onResponse(call: Call, response: Response) {
                val responseBody = response.body
                if (response.isSuccessful) {
                    if (response.header("status") == Constance.EMAIL_REGISTERED) {
                        val numbers = responseBody?.string()
                        toastResponse("Письмо отправлено на почту")
                        if (numbers != null) {
                            onNumbersReceived.invoke(numbers)
                        } else toastResponse("Неизвестная ошибка. Попробуйте позже")
                    } else {
                        toastResponse("Неверная почта")
                    }
                } else toastResponse("Неизвестная ошибка. Попробуйте позже")
            }
        })
    }

    /**
     * Post request to logout to delete token.
     */
    fun logoutUser(onStatusReceived: (Boolean) -> Unit) {
        val requestBody = FormBody.Builder()
            .add("token", Client.token)
            .build()
        Client.client.postRequest("logout/", requestBody, object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                toastResponse("Ошибка соединения")
            }

            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    if (response.header("status") == Constance.LOGOUT_SUCCESS) {
                        Client.resetData()
                        onStatusReceived.invoke(true)
                    } else {
                        toastResponse("Неизвестная ошибка. Свяжитесь с тех. поддержкой")
                    }
                } else toastResponse("Неизвестная ошибка. Повторите позже")
            }
        })
    }

    /**
     * Method gets arrays of entries to draw charts.
     */
    fun getGraphStatistics(onStatisticsReceived: (Statistics) -> Unit) {
        val requestBody = FormBody.Builder()
            .add("token", Client.token)
            .build()
        Client.client.postRequest("statistics/", requestBody, object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                toastResponse("Ошибка соединения")
            }
            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    if (response.header("status") != Constance.UNKNOWN_ERROR) {
                        val statistics = gson.fromJson(
                            response.body?.string(),
                            Statistics::class.java
                        )
                        onStatisticsReceived.invoke(statistics)
                    } else toastResponse("Ошибка загрузки статистики. Обратитесь в поддержку")
                } else toastResponse("Ошибка загрузки статистики")
            }
        })
    }

    /**
     * Method post request to authorize user.
     *
     * @param username is users email.
     * @param password is users password.
     */
    fun auth(username: String, password: String, onTokenReceived: () -> Unit) {
        val requestBody = FormBody.Builder()
            .add("email", username)
            .add("password", password)
            .build()
        Client.client.postRequest("login/", requestBody, object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                toastResponse("Ошибка соединения" + e.message)
            }
            override fun onResponse(call: Call, response: Response) {
                if (response.isSuccessful) {
                    if (response.header("status") == Constance.CORRECT_AUTH_STATUS) {
                        Client.token = response.body?.string().toString()
                        toastResponse("Успешная авторизация")
                        onTokenReceived.invoke()
                    } else {
                        when (response.header("status")) {
                            Constance.EMAIL_ERROR -> toastResponse("Неверная почта")
                            Constance.PASSWORD_ERROR -> toastResponse("Неверный пароль")
                        }
                    }
                } else toastResponse("Ошибка авторизации")
            }
        })
    }

    /**
     * Method gets number of analysed videos.
     */
    fun getNumberOfFiles(onNumberReceived: (FilesNumbers) -> Unit) {
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
                            onNumberReceived.invoke(responseData)
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

    /**
     * Method send request to change password.
     *
     * @param email an email of user who changes password.
     * @param password a new password of user.
     */
    fun changePassword(email: String, password: String, onTokenReceived: () -> Unit) {
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
                        onTokenReceived.invoke()
                    } else {
                        toastResponse("Ошибка сервера. Повторите позже")
                    }
                } else toastResponse("Ошибка смены пароля")
            }
        })
    }

    /**
     * Post request to get recommendations for authorized user.
     */
    fun getRecommendations(onRecommendationsReceived: (RecommendationsInfo) -> Unit) {
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
                        onRecommendationsReceived.invoke(info)
                    }
                } else toastResponse("Ошибка загрузки информации")
            }
        })
    }

    /**
     * Method post request for user registration.
     *
     * @param username is an email to registrate user.
     * @param password ia users password.
     */
    fun register(username: String, password: String, onRegister:() -> Unit) {
        val requestBody = FormBody.Builder()
            .add("email", username)
            .add("password", password)
            .build()
        Client.client.postRequest("register/", requestBody, object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                toastResponse("Ошибка соединения")
            }
            override fun onResponse(call: Call, response: Response) {
                val responseBody = response.body
                if (response.isSuccessful) {
                    if (response.header("status") == Constance.REGISTRATION_SUCCESS) {
                        Client.token = responseBody?.string().toString()
                        toastResponse("Успешная регистрация")
                        onRegister.invoke()
                    } else {
                        when (response.header("status")) {
                            Constance.ACCOUNT_EXIST_STATUS ->
                                toastResponse("Аккаунт с такой почтой уже существует")

                            Constance.UNKNOWN_ERROR ->
                                toastResponse("Ошибка сервера. Обратитесь в поддержку")

                            else -> toastResponse("Неверный формат данных")
                        }
                    }
                } else toastResponse("Ошибка регистрации")
            }
        })
    }

    /**
     * Method make toast with message to user.
     */
    fun toastResponse(text: String) {
        (context as? Activity)?.runOnUiThread {
            Toast.makeText(context, text, Toast.LENGTH_SHORT).show()
        }
    }
}