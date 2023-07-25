package co.huggingface.android_transformers.gpt2

import android.os.Bundle
import android.text.Spannable
import android.text.SpannableStringBuilder
import android.widget.TextView
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.res.ResourcesCompat
import co.huggingface.android_transformers.gpt2.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private val gpt2: co.huggingface.android_transformers.gpt2.ml.GPT2Client by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val binding: ActivityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.autocompleteButton.setOnClickListener {
            gpt2.launchAutocomplete()
        }
        binding.shuffleButton.setOnClickListener {
            gpt2.refreshPrompt()
        }
        gpt2.completion.observe(this) { completion ->
            gpt2.prompt.observe(this) { prompt ->
                binding.prompt.formatCompletion(prompt, completion)
            }
        }
    }

    private fun TextView.formatCompletion(prompt: String, completion: String) {
        text = when {
            completion.isEmpty() -> prompt
            else -> {
                val str = SpannableStringBuilder(prompt + completion)
                val bgCompletionColor =
                    ResourcesCompat.getColor(resources, R.color.colorPrimary, context.theme)
                str.apply {
                    setSpan(
                        android.text.style.BackgroundColorSpan(bgCompletionColor),
                        prompt.length,
                        str.length,
                        Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                    )
                }
            }
        }
    }

}
