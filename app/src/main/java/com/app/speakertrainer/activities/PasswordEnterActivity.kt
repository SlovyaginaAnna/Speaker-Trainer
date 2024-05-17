package com.app.speakertrainer.activities

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.data.FilesNumbers
import com.app.speakertrainer.data.Record
import com.app.speakertrainer.databinding.ActivityPasswordEnterBinding
import com.app.speakertrainer.modules.ApiManager
import com.app.speakertrainer.modules.Client

/**
 * Activity for entering email for futher password change.
 */
class PasswordEnterActivity : AppCompatActivity() {
    private lateinit var bindingClass: ActivityPasswordEnterBinding
    private var email = ""
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
        bindingClass = ActivityPasswordEnterBinding.inflate(layoutInflater)
        setContentView(bindingClass.root)
        email = intent.getStringExtra("email").toString()
    }

    /**
     * Open authorization activity.
     */
    fun onClickAuthorizationScreen(view: View) {
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        finish()
    }

    /**
     * Post request for checking email validation.
     * Open password enter activity if email is correct.
     */
    fun onClickRestorePassword(view: View) {
        if (!isFieldEmpty()) {
            if (isPasswordCorrect()) {
                val password = bindingClass.passwordInputLayout.text.toString()
                apiManager.changePassword(email, password, "password_recovery/save/") {
                    apiManager.getNumberOfFiles { filesInfo ->
                        loadFiles(filesInfo)
                    }
                    val intent = Intent(this@PasswordEnterActivity, Home::class.java)
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

    /**
     * Load users info from server.
     *
     * @param members a FilesNumbers object that contains users loaded records.
     */
    private fun loadFiles(members: FilesNumbers) {
        if (members.num_of_files > 0) {
            for (id in members.file_ids) {
                apiManager.getImg(id.toString()) { image ->
                    apiManager.getInfo(id.toString()) {info ->
                        val record = Record(
                            image, info.filename,
                            info.datetime, id
                        )
                        Client.recordList.add(record)
                    }
                }
            }
        }
    }
}