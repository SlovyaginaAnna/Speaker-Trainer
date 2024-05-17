package com.app.speakertrainer.activities

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.databinding.ActivityEmailEnterBinding
import com.app.speakertrainer.modules.ApiManager

/**
 * Activity for entering email to change password.
 */
class EmailEnterActivity : AppCompatActivity() {
    private lateinit var bindingClass: ActivityEmailEnterBinding
    private var numbers: String? = null
    private var email: String? = null
    private val apiManager = ApiManager(this)

    /**
     * Method called when the activity is created.
     * Initializes the binding to the layout and displays it on the screen.
     *
     * @param savedInstanceState a Bundle object containing the previous state of the activity (if saved)
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        bindingClass = ActivityEmailEnterBinding.inflate(layoutInflater)
        setContentView(bindingClass.root)
    }

    /**
     * Method starts authorization activity.
     */
    fun onClickAuthorization(view: View) {
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        finish()
    }

    /**
     * Method post request for getting numbers to check email.
     */
    fun onClickSendCode(view: View) {
        if (!isFieldEmpty()) {
            email = bindingClass.emailInput.text.toString()
            apiManager.getEmailCheckNumbers(email!!, "password_recovery/email_confirm/") { num ->
                numbers = num
            }
        }
    }

    /**
     * Method opens activity for inputting new password if all data is correct.
     */
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

    /**
     * Method checks if email input text is not empty.
     * If field is empty method set error.
     *
     * @return is email input field is empty.
     */
    private fun isFieldEmpty(): Boolean {
        bindingClass.apply {
            if (emailInput.text.isNullOrEmpty()) emailInput.error =
                resources.getString(R.string.mustFillField)
            return emailInput.text.isNullOrEmpty()
        }
    }

    /**
     * Method checks if field for numbers is empty.
     * If field is empty method set error.
     *
     * @return is number input field is empty.
     */
    private fun isNumEmpty(): Boolean {
        bindingClass.apply {
            if (numInput.text.isNullOrEmpty()) numInput.error =
                resources.getString(R.string.mustFillField)
            return numInput.text.isNullOrEmpty()
        }
    }
}