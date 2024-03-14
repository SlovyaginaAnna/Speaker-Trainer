package com.app.speakertrainer.activities

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import com.app.speakertrainer.databinding.ActivityPreviewVideoBinding
import com.app.speakertrainer.modules.Client
import com.app.speakertrainer.modules.RecordAdapter

class PreviewVideoActivity : AppCompatActivity() {
    lateinit var binding: ActivityPreviewVideoBinding
    private val adapter = RecordAdapter()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPreviewVideoBinding.inflate(layoutInflater)
        setContentView(binding.root)
        init()
    }

    private fun init() {
        binding.apply {
            rcView.layoutManager = LinearLayoutManager(this@PreviewVideoActivity)
            rcView.adapter = adapter
            adapter.setOnItemClickListener(object : RecordAdapter.onItemClickListener {
                override fun onItemClick(position: Int) {
                    val intent = Intent(this@PreviewVideoActivity, AnalysisResults::class.java)
                    intent.putExtra("index", position.toString())
                    startActivity(intent)
                    //finish()
                }
            })
            for (item in Client.recordList) {
                adapter.addRecord(item)
            }
        }
    }

    fun onClickHome(view: View) {
        val intent = Intent(this, Home::class.java)
        startActivity(intent)
        finish()
    }
}