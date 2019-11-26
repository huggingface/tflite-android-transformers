package co.huggingface.android_transformers.gpt2

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import androidx.activity.viewModels
import androidx.lifecycle.observe

class MainActivity : AppCompatActivity() {
    private val gpt2: co.huggingface.android_transformers.gpt2.ml.GPT2Client by viewModels()
    private val handlerThread by lazy { HandlerThread("GPT2Client") }
    private val handler by lazy {
        handlerThread.start()
        Handler(handlerThread.looper)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        handler.post {
            gpt2.init()
            val generation = gpt2.generate("My name is")

            runOnUiThread {
                generation.observe(this) {
                    print(it)
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        handlerThread.quit()
    }
}
