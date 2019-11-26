package co.huggingface.android_transformers.gpt2.ml

import android.app.Application
import android.util.JsonReader
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.liveData
import androidx.lifecycle.viewModelScope
import co.huggingface.android_transformers.gpt2.tokenization.GPT2Tokenizer
import kotlinx.coroutines.Dispatchers
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.min
import kotlin.random.Random

private const val SEQUENCE_LENGTH  = 64
private const val VOCAB_SIZE       = 50257
private const val NUM_HEAD         = 12
private const val NUM_LITE_THREADS = 4
private const val MODEL_PATH       = "model.tflite"
private const val VOCAB_PATH       = "gpt2-vocab.json"
private const val MERGES_PATH      = "gpt2-merges.txt"

private typealias Predictions = Array<Array<FloatArray>>

enum class GPT2StrategyEnum { GREEDY, TOPK }
data class GPT2Strategy(val strategy: GPT2StrategyEnum, val value: Int = 0)

class GPT2Client(application: Application) : AndroidViewModel(application) {
    private lateinit var tokenizer: GPT2Tokenizer
    private lateinit var tflite: Interpreter

    var strategy = GPT2Strategy(GPT2StrategyEnum.TOPK, 40)

    fun init() {
        if (!::tokenizer.isInitialized) {
            val encoder  = loadEncoder()
            val decoder  = encoder.entries.associateBy({ it.value }, { it.key })
            val bpeRanks = loadBpeRanks()

            tokenizer = GPT2Tokenizer(encoder, decoder, bpeRanks)
        }

        if (!::tflite.isInitialized) {
            tflite = loadModel()
        }
    }

    fun generate(text: String, nbTokens: Int = 10) = liveData<String>(
            viewModelScope.coroutineContext+Dispatchers.Default) {

        val tokens = tokenizer.encode(text)
        repeat (nbTokens) {
            val maxTokens    = tokens.takeLast(SEQUENCE_LENGTH).toIntArray()
            val paddedTokens = maxTokens + IntArray(SEQUENCE_LENGTH - maxTokens.size)
            val inputIds     = Array(1) { paddedTokens }

            val predictions: Predictions = Array(1) { Array(SEQUENCE_LENGTH) { FloatArray(VOCAB_SIZE) } }
            val outputs = mutableMapOf<Int, Any>(0 to predictions)

            tflite.runForMultipleInputsOutputs(arrayOf(inputIds), outputs)
            val outputLogits = predictions[0][maxTokens.size-1]

            val nextToken: Int = when (strategy.strategy) {
                GPT2StrategyEnum.TOPK -> {
                    val finalTopK = min(strategy.value, outputLogits.size)
                    val filteredLogits = outputLogits
                            .mapIndexed { index, fl -> (index to fl) }
                            .sortedBy   { it.second }
                            .takeWhile  { it.second < finalTopK }

                    // Softmax computation on filtered logits
                    val maxLogitValue = outputLogits.max()!!
                    val logitsExp     = filteredLogits.map { exp(it.second - maxLogitValue) }
                    val sumExp        = logitsExp.sum()
                    val probs         = logitsExp.map { it.div(sumExp) }

                    val logitsIndexes = filteredLogits.map { it.first }
                    sample(logitsIndexes, probs)
                }
                else -> outputLogits.argmax()
            }

            tokens.add(nextToken)
            val decodedToken = tokenizer.decode(listOf(nextToken))
            emit(decodedToken)
        }
    }

    private fun loadModel(): Interpreter {
        val assetFileDescriptor = getApplication<Application>().assets.openFd(MODEL_PATH)
        return assetFileDescriptor.use {
            val fileChannel = FileInputStream(assetFileDescriptor.fileDescriptor).channel
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, it.startOffset, it.declaredLength)

            val opts = Interpreter.Options()
            opts.setNumThreads(NUM_LITE_THREADS)
            return@use Interpreter(modelBuffer, opts)
        }
    }

    private fun loadEncoder(): Map<String, Int> {
        return hashMapOf<String, Int>().apply {
            val vocabStream = getApplication<Application>().assets.open(VOCAB_PATH)
            vocabStream.use {
                val vocabReader = JsonReader(InputStreamReader(it, "UTF-8"))
                vocabReader.beginObject()
                while (vocabReader.hasNext()) {
                    val key = vocabReader.nextName()
                    val value = vocabReader.nextInt()
                    put(key, value)
                }
                vocabReader.close()
            }
        }
    }

    private fun loadBpeRanks(): Map<Pair<String, String>, Int> {
        return hashMapOf<Pair<String, String>, Int>().apply {
            val mergesStream = getApplication<Application>().assets.open(MERGES_PATH)
            mergesStream.use { stream ->
                val mergesReader = BufferedReader(InputStreamReader(stream))
                mergesReader.useLines { seq ->
                    seq.drop(1).forEachIndexed { i, s ->
                        val list = s.split(" ")
                        val keyTuple = list[0] to list[1]
                        put(keyTuple, i)
                    }
                }
            }
        }
    }
}

private fun randomIndex(probs: List<Float>): Int {
    val rnd = Random.nextFloat()
    var acc = 0f

    probs.forEachIndexed { i, fl ->
        acc += fl
        if (rnd < acc) {
            return i
        }
    }

    return probs.size - 1
}

private fun sample(indexes: List<Int>, probs: List<Float>): Int {
    val i = randomIndex(probs)
    return indexes[i]
}

private fun FloatArray.argmax(): Int {
    var bestIndex = 0
    repeat(size) {
        if (this[it] > this[bestIndex]) {
            bestIndex = it
        }
    }

    return bestIndex
}
