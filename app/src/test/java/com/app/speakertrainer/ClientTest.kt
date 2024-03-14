package com.app.speakertrainer

import org.junit.Test
import org.junit.Assert.*
import android.content.Context
import android.graphics.Bitmap
import com.app.speakertrainer.modules.Client

class ClientTest {
    @Test
    fun testResetData() {
        Client.token = "testToken"
        Client.resetData()
        assertEquals("", Client.token)
        assertTrue(Client.recordList.isEmpty())
    }

}