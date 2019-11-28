package co.huggingface.android_transformers.gpt2

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import androidx.activity.viewModels
import androidx.databinding.DataBindingUtil
import co.huggingface.android_transformers.gpt2.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private val gpt2: co.huggingface.android_transformers.gpt2.ml.GPT2Client by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val binding: ActivityMainBinding
                = DataBindingUtil.setContentView(this, R.layout.activity_main)

        // Bind layout with ViewModel
        binding.vm = gpt2

        // LiveData needs the lifecycle owner
        binding.lifecycleOwner = this
    }
}
