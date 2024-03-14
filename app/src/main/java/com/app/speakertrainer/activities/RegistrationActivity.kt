package com.app.speakertrainer.activities

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.constance.Constance
import com.app.speakertrainer.databinding.ActivityRegistrationBinding
import com.app.speakertrainer.modules.Client
import okhttp3.Call
import okhttp3.Callback
import okhttp3.FormBody
import okhttp3.Response
import java.io.IOException

class RegistrationActivity : AppCompatActivity() {
    private lateinit var bindingClass: ActivityRegistrationBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        bindingClass = ActivityRegistrationBinding.inflate(layoutInflater)
        setContentView(bindingClass.root)
    }

    fun onClickAuthorization(view: View) {
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        finish()
    }


    fun onClickRegistration(view: View) {
        if (!isFieldEmpty()) {
            if (isPasswordCorrect()) {
                val username = bindingClass.emailInputLayout.text.toString()
                val password = bindingClass.passwordInputLayout.text.toString()
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
                                val intent = Intent(
                                    this@RegistrationActivity,
                                    Home::class.java
                                )
                                startActivity(intent)
                                finish()
                            } else {
                                when (response.header("status")) {
                                    Constance.ACCOUNT_EXIST_STATUS ->
                                        toastResponse("Аккаунт с такой почтой уже существует")

                                    Constance.UNKNOWN_ERROR ->
                                        toastResponse("Ошибка сервера. Обратитесь в поддержку")

                                    else -> {
                                        if (response.body?.string().toString() == "email_error")
                                            bindingClass.emailInputLayout.error = "Неверный формат"
                                        else
                                            bindingClass.passwordInputLayout.error =
                                                "Неверный формат"
                                    }
                                }
                            }
                        } else toastResponse("Ошибка регистрации")
                    }
                })
            }
        }
    }

    private fun isFieldEmpty(): Boolean {
        bindingClass.apply {
            if (emailInputLayout.text.isNullOrEmpty())
                emailInputLayout.error = resources.getString(R.string.mustFillField)
            if (passwordInputLayout.text.isNullOrEmpty())
                passwordInputLayout.error = resources.getString(R.string.mustFillField)
            if (pasAgainInputLayout.text.isNullOrEmpty())
                pasAgainInputLayout.error = resources.getString(R.string.mustFillField)
            return emailInputLayout.text.isNullOrEmpty() || passwordInputLayout.text.isNullOrEmpty()
                    || pasAgainInputLayout.text.isNullOrEmpty()

        }
    }

    fun toastResponse(text: String) {
        runOnUiThread {
            Toast.makeText(this@RegistrationActivity, text, Toast.LENGTH_SHORT).show()
        }
    }

    // Check if
    private fun isPasswordCorrect(): Boolean {
        bindingClass.apply {
            if (passwordInputLayout.text.toString() != pasAgainInputLayout.text.toString())
                pasAgainInputLayout.error = "Пароли не совпадают"
            return passwordInputLayout.text.toString() == pasAgainInputLayout.text.toString()
        }
    }
}