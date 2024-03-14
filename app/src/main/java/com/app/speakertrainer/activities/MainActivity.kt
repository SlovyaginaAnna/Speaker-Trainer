package com.app.speakertrainer.activities

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.app.speakertrainer.R
import com.app.speakertrainer.data.FilesNumbers
import com.app.speakertrainer.data.Record
import com.app.speakertrainer.databinding.ActivityMainBinding
import com.app.speakertrainer.modules.ApiManager
import com.app.speakertrainer.modules.Client

/**
 * Activity for entering data for authorization.
 */
class MainActivity : AppCompatActivity() {
    private lateinit var bindingClass: ActivityMainBinding
    private val apiManager = ApiManager(this)

    /**
     * Method called when the activity is created.
     * Initializes the binding to the layout and displays it on the screen.
     *
     * @param savedInstanceState a Bundle object containing the previous state of the activity (if saved)
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        bindingClass = ActivityMainBinding.inflate(layoutInflater)
        setContentView(bindingClass.root)
    }

    /**
     * Method post data to server and authorize user if everything is correct.
     * Get info about user from server.
     */
    fun onClickAuthorization(view: View) {
        if (!isFieldEmpty()) {
            val username = bindingClass.emailET.text.toString()
            val password = bindingClass.passwordET.text.toString()
            apiManager.auth(username, password) {
                apiManager.getNumberOfFiles { filesInfo ->
                    loadFiles(filesInfo)
                }
                val intent = Intent(this@MainActivity, Home::class.java)
                startActivity(intent)
                finish()
            }
        }
    }

    /**
     * Open registration activity.
     */
    fun onClickRegistration(view: View) {
        val intent = Intent(this, RegistrationActivity::class.java)
        startActivity(intent)
        finish()
    }

    /**
     * Open email enter activity for changing password.
     */
    fun onClickChangePassword(view: View) {
        val intent = Intent(this, EmailEnterActivity::class.java)
        startActivity(intent)
        finish()
    }

    /**
     * @return if any of fields is empty
     */
    private fun isFieldEmpty(): Boolean {
        bindingClass.apply {
            if (emailET.text.isNullOrEmpty()) emailET.error =
                resources.getString(R.string.mustFillField)
            if (passwordET.text.isNullOrEmpty()) passwordET.error =
                resources.getString(R.string.mustFillField)
            return emailET.text.isNullOrEmpty() || passwordET.text.isNullOrEmpty()
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