package com.app.speakertrainer.activities

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.app.speakertrainer.databinding.ActivityPreviewVideoBinding
import com.app.speakertrainer.modules.ApiManager
import com.app.speakertrainer.modules.Client
import com.app.speakertrainer.modules.RecordAdapter

/**
 * Activity for displaying archive.
 */
class PreviewVideoActivity : AppCompatActivity() {
    lateinit var binding: ActivityPreviewVideoBinding
    private val adapter = RecordAdapter()
    private val apiManager = ApiManager(this)

    /**
     * Method called when the activity is created.
     * Initializes the binding to the layout and displays it on the screen.
     *
     * @param savedInstanceState a Bundle object containing the previous state of the activity (if saved)
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPreviewVideoBinding.inflate(layoutInflater)
        setContentView(binding.root)
        init()
    }

    /**
     * Method fills adapter for displaying record list.
     */
    private fun init() {
        binding.apply {
            rcView.layoutManager = LinearLayoutManager(this@PreviewVideoActivity)
            rcView.adapter = adapter
            adapter.setOnItemClickListener(object : RecordAdapter.onItemClickListener {
                override fun onItemClick(position: Int) {
                    val intent = Intent(this@PreviewVideoActivity, AnalysisResults::class.java)
                    intent.putExtra("index", position.toString())
                    startActivity(intent)
                    finish()
                }
            })
            adapter.setOnItemLongClickListener(object : RecordAdapter.OnItemLongClickListener {
                override fun onItemLongClick(position: Int) {
                    val record = adapter.recordList[position]
                    val dialog = AlertDialog.Builder(this@PreviewVideoActivity)
                        .setTitle("Удаление")
                        .setMessage("Вы уверены, что хотите удалить видео: ${record.title}")
                        .setPositiveButton("OK") { dialog, _ ->
                            dialog.dismiss()
                            apiManager.delete(Client.recordList[position].index)
                            adapter.removeItem(position)
                            Client.recordList.removeAt(position)
                        }
                        .create()
                    dialog.show()
                }
            })

            for (item in Client.recordList) {
                adapter.addRecord(item)
            }
        }
    }

    /**
     * Open home screen activity.
     */
    fun onClickHome(view: View) {
        val intent = Intent(this, Home::class.java)
        startActivity(intent)
        finish()
    }
}