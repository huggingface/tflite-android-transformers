package co.huggingface.android_transformers.gpt2.ml

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.liveData
import androidx.lifecycle.viewModelScope
import co.huggingface.android_transformers.gpt2.tokenization.GPT2Tokenizer
import kotlinx.coroutines.Dispatchers
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

private const val SEQUENCE_LENGTH  = 64
private const val NUM_LITE_THREADS = 4;
private const val MODEL_PATH       = "model.tflite"

class GPT2Client(application: Application) : AndroidViewModel(application) {
    private val tokenizer = GPT2Tokenizer(application)
    private lateinit var tflite: Interpreter



//    fun generate(text: String, nbTokens: Int = 10) = liveData<Pair<String, Double>>(
//            viewModelScope.coroutineContext+Dispatchers.Default) {
//
//        var tokens = tokenizer.encode(text)
//        for (i in 0 until nbTokens) {
//            val maxTokens = tokens.takeLast(SEQUENCE_LENGTH)
//            val inputIds = tokens.takeLast(SEQUENCE_LENGTH) + IntArray(SEQUENCE_LENGTH - maxTokens.size).toList()
//
//            tflite.runForMultipleInputsOutputs();
//        }
//
//
//    }

    private fun loadModel() {
        val assetFileDescriptor = this.getApplication<Application>().assets.openFd(MODEL_PATH)
        assetFileDescriptor.use {
            val fileChannel = FileInputStream(assetFileDescriptor.fileDescriptor).channel
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, it.startOffset, it.declaredLength)

            val opts = Interpreter.Options();
            opts.setNumThreads(NUM_LITE_THREADS);
        }
    }
}
