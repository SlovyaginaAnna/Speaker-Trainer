package com.app.speakertrainer.activities

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import com.app.speakertrainer.R
import com.app.speakertrainer.databinding.ActivityRegistrationBinding
import com.app.speakertrainer.databinding.ActivityRegistrationPasswordBinding
import com.app.speakertrainer.modules.ApiManager

class RegistrationPasswordActivity : AppCompatActivity() {
    private lateinit var bindingClass: ActivityRegistrationPasswordBinding
    private var email = ""
    private val apiManager = ApiManager(this)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        bindingClass = ActivityRegistrationPasswordBinding.inflate(layoutInflater)
        email = intent.getStringExtra("email").toString()
        setContentView(bindingClass.root)
    }

    /**
     * Open authorization activity.
     */
    fun onClickAuthorization(view: View) {
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        finish()
    }

    /**
     * Post request for checking email validation.
     * Open password enter activity if email is correct.
     */
    fun onClickRegister(view: View) {
        if (!isFieldEmpty()) {
            if (isPasswordCorrect()) {
                val password = bindingClass.passwordInputLayout.text.toString()
                apiManager.changePassword(email, password, "register/save/") {
                    val intent = Intent(this@RegistrationPasswordActivity, Home::class.java)
                    startActivity(intent)
                    finish()
                }
            }
        }
    }

    /**
     * @return if any of fields is empty
     */
    private fun isFieldEmpty(): Boolean {
        bindingClass.apply {
            if (passwordInputLayout.text.isNullOrEmpty())
                passwordInputLayout.error = resources.getString(R.string.mustFillField)
            if (pasAgainInputLayout.text.isNullOrEmpty())
                pasAgainInputLayout.error = resources.getString(R.string.mustFillField)
            return pasAgainInputLayout.text.isNullOrEmpty() || passwordInputLayout.text.isNullOrEmpty()

        }
    }

    /**
     * @return if texts in fields for entering password and for checking password are the same.
     */
    private fun isPasswordCorrect(): Boolean {
        bindingClass.apply {
            if (passwordInputLayout.text.toString() != pasAgainInputLayout.text.toString())
                pasAgainInputLayout.error = "Пароли не совпадают"
            return passwordInputLayout.text.toString() == pasAgainInputLayout.text.toString()
        }
    }

}