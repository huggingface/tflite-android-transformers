# TensorFlow Lite Transformers w/ Android demo

Convert Transformers models
imported from the [🤗 Transformers](https://github.com/huggingface/transformers) library
and use them on Android. You can also check out our
[swift-coreml-transformers](https://github.com/huggingface/swift-coreml-transformers) repo
if you're looking for Transformers on iOS.

## DistilBERT for Question Answering

The app contains a demo of the [DistilBERT](https://arxiv.org/abs/1910.01108) model
(97% of BERT’s performance on GLUE) fine-tuned for Question answering on the SQuAD dataset.
It provides 48 passages from the dataset for users to choose from.

![demo gif](media/distilbert_qa.gif "Demo running offline on a Samsung Galaxy S8")

### Coming soon: GPT-2, quantization... and much more!

---

## Build the demo app using Android Studio

### Prerequisites

*   If you don't have already, install
    [Android Studio](https://developer.android.com/studio/index.html), following
    the instructions on the website.
*   Android Studio 3.2 or later.
*   You need an Android device or Android emulator and Android development
    environment with minimum API 15.
*   The `app/libs` directory contains a custom build of
    [TensorFlow Lite with TensorFlow ops built-in](https://www.tensorflow.org/lite/guide/ops_select),
    which is used by the app. It results in a bigger binary than the "normal" build but allows
    compatibility with models such as DistilBERT.

### Building

*   Open Android Studio, and from the Welcome screen, select `Open an existing
    Android Studio project`.
*   From the Open File or Project window that appears, select the directory where you cloned this repo.
*   You may also need to install various platforms and tools according to error
    messages.
*   If it asks you to use Instant Run, click Proceed Without Instant Run.

### Running

*   You need to have an Android device plugged in with developer options enabled
    at this point. See [here](https://developer.android.com/studio/run/device)
    for more details on setting up developer devices.
*   If you already have Android emulator installed in Android Studio, select a
    virtual device with minimum API 15.
*   Click `Run` to run the demo app on your Android device.

## Build the demo using gradle (command line)

### Building and Installing

*   Use the following command to build a demo apk:

```
./gradlew build
```

*   Use the following command to install the apk onto your connected device:

```
adb install app/build/outputs/apk/debug/app-debug.apk
```

---

## Models generation

Example scripts used to convert models are available in the `models_generation` directory.
Please note that they require the nightly version of TensorFlow and might thus be unstable.

---

## Credits

The Android app is forked from the `bertqa` example in the
[tensorflow/examples](https://github.com/tensorflow/examples) repository and uses the same
tokenizer with DistilBERT.

## License

[Apache License 2.0](LICENSE)
