# Text Generation with GPT2/DistilGPT2

On-device text generation app using [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) or [DistilGPT2](https://medium.com/huggingface/distilbert-8cf3380435b5) (same distillation process than DistilBERT, 2x faster and 33% smaller than GPT-2)

![demo gif](../media/gpt2_generation.gif "Demo running offline on a Samsung Galaxy S8, accelerated")

> Available models: GPT-2 (small), GPT-2-FP16 (small), DistilGPT2

## Build the demo app using Android Studio

### Prerequisites

*   If you don't have already, install
    [Android Studio](https://developer.android.com/studio/index.html), following
    the instructions on the website.
*   Android Studio 3.2 or later.
*   You need an Android device or Android emulator and Android development
    environment with minimum API 26.

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
    virtual device with minimum API 26.
*   Be sure the `gpt2` configuration is selected
*   Click `Run` to run the demo app on your Android device.

## Build the demo using gradle (command line)

From the repository root location:

*   Use the following command to build a demo apk:

```
./gradlew :gpt2:build
```

*   Use the following command to install the apk onto your connected device:

```
adb install gpt2/build/outputs/apk/debug/gpt2-debug.apk
```

## Change the model

To choose which model to use in the app:
*   Remove/rename the current `model.tflite` file in `src/main/assets`
*   Comment/uncomment the model to download in the `download.gradle` config file:
```java
"https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-fp16-64.tflite": "model.tflite", // <- fp16 quantized version of gpt-2 (small) (default)
// "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-64.tflite": "model.tflite", // <- "original" gpt-2 (small)
// "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-64.tflite": "model.tflite", // <- distilled version of gpt-2 (small)
```
