package com.app.speakertrainer.activities

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.databinding.ActivityRegistrationBinding
import com.app.speakertrainer.modules.ApiManager

/**
 * Activity for registration user.
 */
class RegistrationActivity : AppCompatActivity() {
    private lateinit var bindingClass: ActivityRegistrationBinding
    private val apiManager = ApiManager(this)

    /**
     * Method called when the activity is created.
     * Initializes the binding to the layout and displays it on the screen.
     * Call method that sets statistics.
     *
     * @param savedInstanceState a Bundle object containing the previous state of the activity (if saved)
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        bindingClass = ActivityRegistrationBinding.inflate(layoutInflater)
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
     * Post user data to server to registrate user.
     */
    fun onClickRegistration(view: View) {
        if (!isFieldEmpty()) {
            if (isPasswordCorrect()) {
                val username = bindingClass.emailInputLayout.text.toString()
                val password = bindingClass.passwordInputLayout.text.toString()
                apiManager.register(username, password) {
                    val intent = Intent(
                        this@RegistrationActivity,
                        Home::class.java
                    )
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