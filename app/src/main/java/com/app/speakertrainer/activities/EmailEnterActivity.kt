package com.app.speakertrainer.activities

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.constance.Constance
import com.app.speakertrainer.databinding.ActivityEmailEnterBinding
import com.app.speakertrainer.modules.Client
import okhttp3.Call
import okhttp3.Callback
import okhttp3.FormBody
import okhttp3.Response
import java.io.IOException

class EmailEnterActivity : AppCompatActivity() {
    private lateinit var bindingClass: ActivityEmailEnterBinding
    private var numbers: String? = null
    private var email: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        bindingClass = ActivityEmailEnterBinding.inflate(layoutInflater)
        setContentView(bindingClass.root)
    }

    fun onClickAuthorization(view: View) {
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        finish()
    }

    fun onClickSendCode(view: View) {
        if (!isFieldEmpty()) {
            email = bindingClass.emailInput.text.toString()
            val requestBody = FormBody.Builder()
                .add("email", email.toString())
                .build()
            Client.client.postRequest("password_recovery/", requestBody, object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    toastResponse("Ошибка соединения" + e.message)
                }

                override fun onResponse(call: Call, response: Response) {
                    val responseBody = response.body
                    if (response.isSuccessful) {
                        if (response.header("status") == Constance.EMAIL_REGISTERED) {
                            numbers = responseBody?.string()
                            toastResponse("Письмо отправлено на почту")
                        } else {
                            toastResponse("Неверная почта")
                        }
                    } else toastResponse("Неизвестная ошибка. Попробуйте позже")
                }
            })
        }
    }

    fun onClickNext(view: View) {
        if (!isNumEmpty() && numbers != null) {
            if (bindingClass.numInput.text.toString() == numbers) {
                val intent = Intent(this, PasswordEnterActivity::class.java)
                intent.putExtra("email", email)
                startActivity(intent)
                finish()
            } else bindingClass.numInput.error = "Неверный код"
        }
    }

    fun toastResponse(text: String) {
        runOnUiThread {
            Toast.makeText(this@EmailEnterActivity, text, Toast.LENGTH_SHORT).show()
        }
    }

    private fun isFieldEmpty(): Boolean {
        bindingClass.apply {
            if (emailInput.text.isNullOrEmpty()) emailInput.error =
                resources.getString(R.string.mustFillField)
            return emailInput.text.isNullOrEmpty()
        }
    }

    private fun isNumEmpty(): Boolean {
        bindingClass.apply {
            if (numInput.text.isNullOrEmpty()) numInput.error =
                resources.getString(R.string.mustFillField)
            return numInput.text.isNullOrEmpty()
        }
    }
}