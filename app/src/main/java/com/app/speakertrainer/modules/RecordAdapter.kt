package com.app.speakertrainer.modules

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.app.speakertrainer.R
import com.app.speakertrainer.data.Record
import com.app.speakertrainer.databinding.RecordItemBinding

// The `RecordAdapter` class is an adapter for displaying records in a RecyclerView.
class RecordAdapter : RecyclerView.Adapter<RecordAdapter.RecordHolder>() {

    // It includes an interface for item click events and a ViewHolder inner class to bind data to views.
    interface onItemClickListener {
        fun onItemClick(position: Int)
    }

    val recordList = ArrayList<Record>()
    private lateinit var mListener: onItemClickListener

    class RecordHolder(item: View, listener: onItemClickListener) : RecyclerView.ViewHolder(item) {
        val binding = RecordItemBinding.bind(item)
        fun bind(record: Record) = with(binding) {
            im.setImageBitmap(record.image)
            tvName.text = record.title
            tvDate.text = record.date
        }

        init {
            itemView.setOnClickListener {
                listener.onItemClick(adapterPosition)
            }
        }
    }

    // The `onCreateViewHolder` method inflates the layout for each item in the RecyclerView.
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecordHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.record_item, parent, false)
        return RecordHolder(view, mListener)
    }

    override fun getItemCount(): Int {
        return recordList.size
    }

    // The `onBindViewHolder` method binds data to the ViewHolder.
    override fun onBindViewHolder(holder: RecordHolder, position: Int) {
        holder.bind(recordList[position])
    }

    // The adapter also provides methods to add records, set item click listeners, and notify data changes.
    fun addRecord(record: Record) {
        recordList.add(record)
        notifyDataSetChanged()
    }

    fun setOnItemClickListener(listener: onItemClickListener) {
        mListener = listener
    }
}