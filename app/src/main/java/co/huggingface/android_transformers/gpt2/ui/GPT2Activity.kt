package co.huggingface.android_transformers.gpt2.ui

import android.os.Bundle
import android.support.design.widget.Snackbar
import android.support.v7.app.AppCompatActivity
import co.huggingface.android_transformers.R
import co.huggingface.android_transformers.gpt2.ml.GPT2Client

import kotlinx.android.synthetic.main.activity_gpt2.*

class GPT2Activity : AppCompatActivity() {
    private val t: GPT2Client by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_gpt2)
        setSupportActionBar(toolbar)

        fab.setOnClickListener { view ->
            Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                    .setAction("Action", null).show()
        }

    }

}
