package co.huggingface.android_transformers.gpt2.tokenization

import android.content.Context
import android.util.JsonReader
import java.io.BufferedReader
import java.io.InputStreamReader

private const val VOCAB_PATH  = "gpt2-vocab.json"
private const val MERGES_PATH = "gpt2-merges.txt"

class GPT2Tokenizer(private val context: Context) {
    private val encoder: Map<String, Int>
    private val decoder: Map<Int, String>
    private val bpeRanks: Map<Pair<String, String>, Int>
    private val encodeRegex = Regex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    init {
        encoder = hashMapOf<String, Int>().apply {
            val vocabStream = context.assets.open(VOCAB_PATH)
            vocabStream.use {
                val vocabReader = JsonReader(InputStreamReader(it, "UTF-8"))
                vocabReader.beginObject();
                while (vocabReader.hasNext()) {
                    val key = vocabReader.nextName()
                    val value = vocabReader.nextInt()
                    put(key, value)
                }
                vocabReader.close()
            }
        }

        decoder = encoder.entries.associateBy({ it.value }, { it.key })

        bpeRanks = hashMapOf<Pair<String, String>, Int>().apply {
            val mergesStream = context.assets.open(MERGES_PATH)
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

    fun decode(tokens: List<Int>): String {
        val text = tokens.joinToString("") { decoder.getOrDefault(it, "") }
        val utfCodepoints = text.map { byteDecoder[it.toString()]!! }
        return String(utfCodepoints.toIntArray(), 0, utfCodepoints.size)
    }

    fun encode(text: String): List<Int> {
        val tokens = encodeRegex.findAll(text).map {
            it.value.codePoints()
                    .boxed()
                    .map { byteEncoder[it]!! }
                    .toArray()
                    .joinToString("")
        }

        return tokens
                .map { bpe(it) }
                .flatten()
                .map { encoder[it]!! }
                .toList()
    }

    private fun bpe(token: String): List<String> {
        if (token.length <= 1) return listOf(token)

        var word = token.map { it.toString() }
        var pairs = getPairs(word)

        while (true) {
            val (first, second) = pairs.minBy { bpeRanks.getOrDefault(it, Int.MAX_VALUE) } ?: break

            var i = 0
            val newWord = mutableListOf<String>()
            while (i < word.size) {
                val j = word.subList(i, word.size).indexOf(first)
                if (j != -1) {
                    newWord.addAll(word.subList(i, j))
                    i = j
                } else {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }

                if (word[i] == first && i < word.size-1 && word[i+1] == second) {
                    newWord.add(first+second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }

            word = newWord
            if (word.size == 1) {
                break
            } else {
                pairs = getPairs(word)
            }
        }

        return word
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        return mutableSetOf<Pair<String, String>>().apply {
            for (i in 0 until word.size-1) {
                add(word[i] to word[i+1])
            }
        }
    }
}
