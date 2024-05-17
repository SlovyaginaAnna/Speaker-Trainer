package com.app.speakertrainer.modules

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.app.speakertrainer.R
import com.app.speakertrainer.data.Record
import com.app.speakertrainer.databinding.RecordItemBinding

/**
 * Class is an adapter for displaying records in a RecyclerView.
 */
class RecordAdapter : RecyclerView.Adapter<RecordAdapter.RecordHolder>() {

    /**
     * It includes an interface for item click events and a ViewHolder inner class to bind data to views.
     */
    interface onItemClickListener {
        fun onItemClick(position: Int)
    }

    interface OnItemLongClickListener {
        fun onItemLongClick(position: Int)
    }

    val recordList = ArrayList<Record>()
    private lateinit var mListener: onItemClickListener
    private lateinit var mLongClickListener: OnItemLongClickListener

    /**
     * Class for setting records to corresponded binding.
     *
     * @param item a View to set elements.
     * @param listener a listener to click on item event.
     */
    class RecordHolder(item: View, listener: onItemClickListener, longClickListener: OnItemLongClickListener) : RecyclerView.ViewHolder(item) {
        val binding = RecordItemBinding.bind(item)

        /**
         * Method binds record.
         *
         * @param record a record to bind.
         */
        fun bind(record: Record) = with(binding) {
            im.setImageBitmap(record.image)
            tvName.text = record.title
            tvDate.text = record.date
        }

        /**
         * Set click listener.
         */
        init {
            itemView.setOnClickListener {
                listener.onItemClick(adapterPosition)
            }
            itemView.setOnLongClickListener {
                longClickListener.onItemLongClick(adapterPosition)
                true
            }
        }
    }

    /**
     * Method inflates the layout for each item in the RecyclerView.
     *
     * @param parent a ViewGroup to get context from.
     * @param viewType an Int value used for identifying the type of view to be created.
     * @return a RecordHolder instance associated with the inflated view.
     */
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecordHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.record_item, parent, false)
        return RecordHolder(view, mListener, mLongClickListener)
    }

    /**
     * Gets number of records in recordList.
     */
    override fun getItemCount(): Int {
        return recordList.size
    }

    /**
     * Method binds data to the ViewHolder.
     *
     * @param holder a RecordHolder to bind element.
     * @param position a Int index of recordList element to bind.
     */
    override fun onBindViewHolder(holder: RecordHolder, position: Int) {
        holder.bind(recordList[position])
    }

    /**
     * Method adds record to recordList.
     *
     * @param record a record to add to reordList.
     */
    fun addRecord(record: Record) {
        recordList.add(record)
        notifyDataSetChanged()
    }

    fun removeItem(position: Int) {
        recordList.removeAt(position)
        notifyItemRemoved(position)
        notifyDataSetChanged()
    }

    /**
     * Method set click listener to items.
     *
     * @param listener a listener to notify item click events.
     */
    fun setOnItemClickListener(listener: onItemClickListener) {
        mListener = listener
    }

    fun setOnItemLongClickListener(longClickListener: OnItemLongClickListener) {
        mLongClickListener = longClickListener
    }
}