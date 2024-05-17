package com.app.speakertrainer.modules

import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import java.util.concurrent.TimeUnit

/**
 * This Class defines the `NetworkClient` class responsible for making HTTP requests using OkHttp library.
 */
class NetworkClient {
    //var partUrl = "http://10.0.2.2:8000/"
    val partUrl = "http://192.168.0.157:8000/"
    private val client = OkHttpClient.Builder()
        .readTimeout(600, TimeUnit.SECONDS)
        .writeTimeout(600, TimeUnit.SECONDS)
        .build()

    /**
     * The `postRequest` function sends a POST request with the specified URL and request body.
     *
     * @param url a string url to post request.
     * @param requestBody a request bode to post request.
     */
    fun postRequest(url: String, requestBody: RequestBody, callback: Callback) {
        val request = Request.Builder()
            .url(partUrl + url)
            .post(requestBody)
            .build()
        client.newCall(request).enqueue(callback)
    }

    /**
     * The `getRequest` function creates a GET request with the specified URL and request body.
     *
     * @param url a string url to post request.
     * @param requestBody a request bode to post request.
     */
    fun getRequest(url: String, requestBody: RequestBody): Request {
        return Request.Builder()
            .url(partUrl + url)
            .post(requestBody)
            .build()
    }

    /**
     * Method gets client.
     */
    fun getClient(): OkHttpClient {
        return client
    }

}