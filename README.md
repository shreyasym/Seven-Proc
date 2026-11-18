# Seven-Segment Meter OCR

This project reads digital meter values from photos by combining classical computer-vision techniques with optional learned digit classification. The core idea is simple:

1. Find the display region in a meter photograph.
2. Rectify the display so it is front-on and easy to analyze.
3. Split the display into individual digit boxes and detect decimal points.
4. Decode each digit using seven-segment segment logic.
5. Optionally improve the result with a trained classifier model.
6. Return a numeric reading or `null` when the reading is not confidently recoverable.

The project is designed to work from a local Python environment without a cloud service. It can be used from the command line or through a browser-based upload interface.

## What this project does

The OCR pipeline is optimized for digital meter displays that use seven-segment digits (common on utility meters, industrial gauges, and some consumer hardware). It includes:

- Bezel detection and perspective rectification.
- Stroke-based fallback localization when the rectangular display border is hard to detect.
- Digit segmentation based on connected components and digit-band analysis.
- Seven-segment recognition logic that infers which of the seven segments are active.
- A hybrid mode that blends classical recognition with a learned classifier.
- Synthetic-data generation for testing robustness under italic, noisy, glare, tilted, and ghosted conditions.
- Dataset tooling for exporting, labeling, and training digit models.
- A local Flask web app for processing photos in a browser.

## High-level architecture

The project follows a pipeline architecture:

```text
image file or uploaded photo
        |
        v
load_image() / preprocess.py
        |
        v
find_display() / roi.py
        |
        v
_read_display() / pipeline.py
        |
        v
find_digits() / digits.py
        |
        v
recognise() / recognize.py
        |
        v
(optional) classifier.py
        |
        v
JSON output or browser result card
```

This is a classic CV pipeline first, with the model only used as an optional enhancer rather than a full replacement for recognition logic.

## Project structure

```text
sevenseg_ocr/
├── README.md
├── app.py
├── annotate_meter_dataset.py
├── kaggle_workflow.py
├── requirements.txt
├── train_cnn.py
├── .gitignore
├── samples/
│   ├── meter_01.png
│   ├── meter_02.png
│   ├── meter_03.png
│   ├── meter_04.png
│   └── meter_05.png
├── models/
│   ├── digit_cnn.pt
│   ├── digit_cnn_samples.pt
│   └── digit_cnn_samples_ft.pt
├── data/
│   ├── archive/
│   ├── cnn_digit_crops/
│   └── ...
├── outputs/
│   ├── meter_readings.json
│   └── ...
├── static/
│   ├── app.js
│   └── styles.css
├── templates/
│   └── index.html
├── sevenseg/
│   ├── __init__.py
│   ├── classifier.py
│   ├── cli.py
│   ├── crops.py
│   ├── digits.py
│   ├── evaluate.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── recognize.py
│   ├── roi.py
│   └── synth.py
├── tests/
│   ├── test_synthetic.py
│   ├── test_training_data.py
│   └── test_web.py
└── __pycache__/
```

## Setup and run

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install required packages:

```bash
python -m pip install -r requirements.txt
```

If you want to use the bundled PyTorch model, install PyTorch separately:

```bash
python -m pip install torch
```

Run the command-line OCR on a sample image:

```bash
python -m sevenseg.cli samples/meter_03.png
```

Run it with a learned model:

```bash
python -m sevenseg.cli samples/meter_03.png --model models/digit_cnn.pt
```

Run the flask app:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## File-by-file explanation

### Top-level files

#### README.md
This file documents the project and explains how it works, how to set it up, and how each source file fits into the pipeline. It is the main developer guide for the repository.

#### app.py
This is the Flask web application that serves the browser UI and accepts meter image uploads. It:

- Defines the local HTTP server.
- Allows up to 10 uploaded JPEG/PNG/WEBP/BMP images.
- Rejects oversized files or unsupported formats.
- Decodes uploaded images into OpenCV arrays.
- Calls `read_image()` from `sevenseg.pipeline`.
- Returns JSON with the detected reading, confidence score, and elapsed processing time.

It is the easiest way to use the project without writing Python scripts.

#### requirements.txt
This lists Python dependencies needed by the project. At minimum, it includes the OpenCV stack and supporting libraries used by preprocessing, detection, and the web app. Install all of them with `python -m pip install -r requirements.txt`.

#### train_cnn.py
This trains a classifier model on digit crops. It supports two backends:

- `torch` backend for a PyTorch CNN
- `sklearn` backend for an MLP classifier

The script loads real digit crops from folders named `0` through `9`, optionally adds synthetic crops, splits training/validation data by source image group, and saves the trained model to disk.

It is the central script for digit-model training.

#### kaggle_workflow.py
This script automates the Kaggle digital-meter dataset workflow. It provides commands for:

- downloading the dataset
- inspecting files and labels
- splitting into train/held-out sets
- exporting crops from images
- training a model
- evaluating performance on a labeled validation set

The project uses it as a dataset pipeline to scale beyond the bundled sample images.

#### annotate_meter_dataset.py
This is a browser-based image annotation tool for creating manually verified labeled digit crops from archived meter photos. It:

- loads CSV metadata from archived dataset folders
- checks that image filenames and labels are internally consistent
- serves a local HTML page in a browser
- lets the user draw one box around each digit in order
- writes PNG crops into numbered digit folders such as `0/`, `1/`, ..., `9/`

This script is essential when building a more robust classifier from real-world meter images.

### Package: sevenseg/

#### sevenseg/__init__.py
This file exposes the public API for the package. It imports:

- `read_image()`
- `read_meter()`

and exposes them in `__all__` so external code can say `from sevenseg import read_meter`.

#### sevenseg/preprocess.py
This module prepares raw images for analysis.

Key functions:

- `load_image()` reads an image from disk and resizes it to a reasonable width.
- `resize_keep_aspect()` maintains the original aspect ratio while limiting the image width.
- `suppress_glare()` removes very bright reflective highlights using HSV thresholding and inpainting.
- `enhance_gray()` converts to grayscale, suppresses glare, denoises, and applies CLAHE contrast enhancement.
- `flatten_illumination()` corrects brightness gradients that can hide or distort digits.
- `binarize_strokes()` performs adaptive thresholding so dark segments become bright foreground on a dark background.

This file is the first stage in the recognition pipeline.

#### sevenseg/roi.py
This module finds the meter display area and rectifies it into a front-on view. It contains:

- `ROI` dataclass storing the rectified image, source box corners, and a score.
- `_order_pts()` and `_warp()` for perspective transformation.
- `_contours()` and candidate scoring logic for display detection.
- `find_display_candidates()` which searches multiple contour-based candidates.
- `find_display()` uses the best candidate.
- `locate_by_strokes()` is the fallback logic that ignores the bezel and looks for the long vertical stroke pattern characteristic of seven-segment digits.

This file is responsible for the “Where is the display?” step.

#### sevenseg/digits.py
This file does the digit segmentation after the display has been rectified.

Key responsibilities:

- `DigitBox` stores each detected digit box and whether it has a decimal point after it.
- `estimate_shear()` corrects italicized or slanted digits by measuring a shear transform.
- `apply_shear()` applies that transform.
- `strip_frame()` removes the outer display framing and border noise.
- `_digit_band()` finds the row band that contains the digits, using vertical strokes rather than horizontal noise.
- `find_digits()` returns a list of left-to-right digit boxes and cleaned binary data.

This is the segmentation and digit-boundary step, and it is one of the most important parts of the project.

#### sevenseg/recognize.py
This file performs the classical seven-segment recognition logic.

Important pieces:

- `PATTERNS` maps numeric digits to the segments needed to draw them.
- `SEGS` lists the seven segment names in order.
- `ZONES` defines the sampling windows for each segment within a digit.
- `segment_fills()` calculates how much of each segment region is filled with ink.
- `classify()` compares the observed segment fill pattern to known seven-segment patterns and returns a digit plus confidence.
- `recognise()` transforms a list of digit boxes into a text string like `4869.46` and a per-digit confidence score list.

This is the “classical OCR engine” of the project.

#### sevenseg/classifier.py
This file handles learned digit classification for better OCR on noisy or ambiguous digits. It supports two model backends:

- PyTorch CNN models (`.pt` files)
- scikit-learn model pickles (`.pkl` files)

Specific features:

- `make_crop()` converts a detected digit box into a normalized 28x28 crop.
- `normalise_crop()` centers and rescales the crop while preserving the digit shape.
- `augment()` adds random transformations to simulate real-world image variation.
- `SkClassifier` wraps a scikit-learn estimator.
- `ConfidenceEnsemble` chooses the best prediction across multiple classifiers.
- `build_torch_model()` defines the CNN architecture.
- `TorchClassifier` loads a saved PyTorch model and predicts classes.
- `load_classifier()` loads either a `.pt` or `.pkl` classifier.

This is the optional machine-learning enhancement layer.

#### sevenseg/crops.py
This module is for building and working with digit-crop datasets.

Key functions:

- `crops_from_image()` extracts digit crops from an image using the detection pipeline and returns guessed labels.
- `export()` writes all crops into folders such as `out_dir/7/` and names them by the guessed digit class.
- `synthetic()` generates synthetic digit crops from randomized seven-segment meter images.
- `load_folder()` reads labeled crop folders and returns arrays of images, labels, and source-group metadata.

This script is mainly used to prepare training data for the classifier.

#### sevenseg/synth.py
This file generates synthetic seven-segment meter images for tests and robustness checks. It creates artificial meter displays with variations such as:

- italic slant
- glare
- noise
- blur
- perspective tilt
- ghost segments
- varying meter body background

The functions `render()` and `random_reading()` create synthetic meter strings such as `245.9` or `6308.0`, then render them as realistic seven-segment style images.

This is crucial for testing because it gives repeatable, controlled examples without needing real meter photos.

#### sevenseg/evaluate.py
This is a small script to run a synthetic stress evaluation over a set of generated images. It:

- creates several condition groups: clean, italic, glare, noisy, tilted, and no-ghost
- renders random readings
- runs `read_image()` on each one
- reports exact-match accuracy and timing

This is useful when measuring how robust the pipeline is under controlled stress conditions.

#### sevenseg/pipeline.py
This is the central orchestration layer of the project.

It contains:

- `_read_display()` which normalizes the display crop, finds digits, optionally corrects shear, and returns cleaned binary data and boxes.
- `_score()` which calculates a rough confidence score based on recognized segment text and confidence values.
- `_recover_digit_row()` which tries to detect a missing or weak row of digits by scanning for a digit cluster.
- `_fuse_with_classifier()` which blends segment-recognition text with classifier output.
- `_read_display_crop_candidates()` which retries recognition across display crop candidates when needed.
- `read_image()` which attempts display localization with both standard and stroke-based methods and selects the best result.
- `read_meter()` which loads an image from disk and returns a JSON-ready dict of reading and confidence.

This is the main execution pipeline the rest of the project relies on.

#### sevenseg/cli.py
This is the command-line interface for the project. It uses `argparse` to support commands like:

```bash
python -m sevenseg.cli image1.png image2.png --debug-dir outputs/debug
```

It optionally loads one or more classifiers and prints JSON for each image. It can also save a debug image showing the detected display and recognized text.

### Tests

#### tests/test_synthetic.py
This file checks the seven-segment recognition logic against synthetic digits and a few included sample images. It validates that:

- the segment lookup table is complete and unique
- clean synthetic strings are read correctly
- italic or slanted synthetic strings remain readable
- real sample meters like `meter_01.png` and `meter_03.png` are recognized correctly

#### tests/test_training_data.py
This file validates the training-data pipeline and annotation logic. It checks that:

- annotation tasks are skipped when labels do not match the archive metadata
- imported crop folders are grouped correctly by source image
- validation splits do not leak crops from the same source image
- the trained ensemble model reads the confirmed sample meters correctly

#### tests/test_web.py
This file checks the Flask app behavior. It verifies that:

- the home page loads
- uploaded images are accepted and processed
- invalid file types or unreadable image payloads return proper 400 errors

### Front-end assets

#### static/app.js
This JavaScript file powers the browser upload UI. It handles:

- file selection and drag-and-drop uploads
- validation of acceptable image types and size limits
- camera access and photo capture
- preview rendering of selected files
- sending the processed files to the backend
- formatting and displaying results cards

This is the client-side logic that makes the app usable without command-line work.

#### static/styles.css
This file defines the visual design for the app: layout, colors, cards, buttons, dark/light interface styling, responsive spacing, and file/result panels. It keeps the web interface consistent and polished.

#### templates/index.html
This is the HTML page served by Flask. It contains the upload area, camera section, result cards, and the necessary markup for the JavaScript app to attach to. It also includes the upload instructions and layout structure that the browser sees.

## Model and data assets

### samples/
This folder contains real meter photos used for testing and demonstration. The project uses these examples to validate the OCR pipeline and to compare results across different visual conditions.

### models/
This folder stores trained digit-classifier weights. The names indicate the purpose of each model:

- `digit_cnn.pt`: a base CNN model
- `digit_cnn_samples.pt`: a model trained using verified sample crops and synthetic digits
- `digit_cnn_samples_ft.pt`: a fine-tuned version initialized from the base model

These models are optional; the classical segment reader works without them.

### data/
This folder contains the working datasets used for calibration and training:

- `archive/`: original archived meter photos and labeling metadata
- `cnn_digit_crops/`: labeled digits grouped by class (`0` through `9`)
- other generated folders are created by training and annotation scripts

### outputs/
This stores outputs such as debug images, evaluation CSVs, and any JSON results created during processing.

## Typical workflows

### 1. Quick inspection with the built-in OCR

```bash
python -m sevenseg.cli samples/meter_03.png
```

### 2. Use the browser on a local machine

```bash
python app.py
```

### 3. Export digit crops from a folder of images

```bash
python -m sevenseg.crops export path/to/meter_images dataset
```

### 4. Label images manually for better training data

```bash
python annotate_meter_dataset.py --quality HQ_digital
```

### 5. Train your own digit classifier

```bash
python train_cnn.py --data data/cnn_digit_crops --verified-only --synthetic 600 --epochs 30 --out models/digit_cnn_samples.pt --seed 13
```

### 6. Evaluate synthetic robustness

```bash
python -m sevenseg.evaluate --n 60
```

## Limitations and practical use

This OCR system is good for controlled digital displays with clear seven-segment digits, but no OCR system is perfect. The main sources of error are:

- low image resolution
- glare and highlights
- motion blur or defocus
- shadows and nonuniform illumination
- damaged or partially missing segments
- unusual fonts outside the typical seven-segment style

In production, compare the result against the physical meter when possible and retrain on more real images if accuracy needs to improve.

## How training accuracy, loss, epochs, and model comparison work

The training logic is in [train_cnn.py](train_cnn.py), and the classifier backends are defined in [sevenseg/classifier.py](sevenseg/classifier.py). This section explains how the printed numbers are generated and what they mean.

### 1. What the script is doing before training starts

The script loads labeled digit crops from folders such as `0/`, `1/`, ..., `9/` and optionally adds synthetic meter-digit crops. For each crop it stores:

- `X`: the pixel array of each digit image
- `y`: the numeric label for that digit (0 through 9)
- `groups`: the source image or synthetic source that created the crop

Then it splits the data into train and validation groups with `split_by_group(groups, rng)`. This is important because it avoids a common mistake in image datasets: putting crops from the same original meter photo into both training and validation. That would make validation unrealistically easy.

```python
tr, va = split_by_group(groups, rng)
```

Here:

- `tr` = indices used for training
- `va` = indices used for validation

The validation set is a held-out subset of source images. This is more realistic than randomly splitting individual digit crops, because the same physical meter image should not appear in both sets.

### 2. What augmentation does

Inside each epoch, the script creates an augmented training set:

```python
model.train(); Xa, ya = aug_set(tr, 3); p = rng.permutation(len(Xa))
```

`aug_set()` applies random transformations to digit crops using `augment()` from [sevenseg/classifier.py](sevenseg/classifier.py), such as small affine shifts, brightness changes, blur, and noise. This helps the model handle different meter conditions such as glare, blur, or contrast differences.

This means the training data is not static: every epoch sees slightly altered versions of the same crop, which improves robustness.

### 3. What the loss is

For each mini-batch, the training loss is computed as:

```python
loss = nn.functional.cross_entropy(model(torch.from_numpy(Xa[b][:, None])), torch.from_numpy(ya[b]))
```

This is the standard multiclass classification loss for digits 0 to 9. In plain English:

- the model predicts probabilities for each digit class
- the true class is known from the label
- `cross_entropy` measures how far the model's prediction is from the true label
- a lower loss means the model is more confident and more correct

The most important detail is that this loss is the training loss for the current mini-batch, not the validation loss.

### 4. What the optimizer does

After computing the loss, the code does:

```python
opt.zero_grad(); loss.backward(); opt.step(); sched.step()
```

This is the learning step:

- `zero_grad()` clears previous gradients
- `backward()` computes how much each weight contributed to the loss
- `opt.step()` updates the weights using the AdamW optimizer
- `sched.step()` adjusts the learning rate according to the OneCycleLR schedule

This is how the model actually learns over time.

### 5. What validation accuracy is

After finishing one full training epoch, the script evaluates the model on the held-out validation set:

```python
model.eval()
with torch.no_grad():
    acc = (model(Xv).argmax(1) == yv).float().mean().item()
```

This means:

- pass all validation digit crops through the model
- take the class with the highest probability (`argmax(1)`)
- compare to the true labels (`yv`)
- compute the fraction of correct predictions

So `acc` is the validation accuracy, a number between 0 and 1. For example:

- `acc = 0.82` means 82% of validation digits were classified correctly
- `acc = 0.45` means 45% were classified correctly

This is the key metric used to judge how well the model generalizes.

### 6. What the printed epoch line means

The script prints:

```python
print(f"epoch {ep + 1:2d}  loss {loss.item():.3f}  val acc {acc:.3f}")
```

The output usually looks like this:

```text
epoch  1  loss 1.482  val acc 0.214
epoch  2  loss 1.230  val acc 0.302
epoch  3  loss 1.005  val acc 0.401
epoch  4  loss 0.972  val acc 0.468
```

Interpretation:

- `loss` is the model's training loss from the last mini-batch in that epoch
- `val acc` is the held-out validation accuracy after that epoch

This is not a validation loss. It is not an average of all losses across the epoch. It is simply the last batch's loss value and the validation accuracy after the epoch.

### 7. Why we do not use validation loss here

The project does not compute a separate validation loss and print it. It chooses the best model using validation accuracy instead:

```python
if acc >= best:
    best = acc
    torch.save(model.state_dict(), a.out)
```

This means the script keeps the model checkpoint with the highest validation accuracy, not the one with the lowest validation loss. In other words:

- the comparison metric is validation accuracy
- the saved model is the best-performing validation model seen so far

At the end it prints:

```python
print("best val acc", best, "->", a.out)
```

So the final model is not necessarily from the final epoch — it is from the epoch with the highest validation accuracy.

### 8. What `epochs` means in this project

The argument `--epochs` controls the number of training cycles. For example:

```bash
python train_cnn.py --data dataset --synthetic 600 --epochs 30 --out models/digit_cnn.pt
```

means:

- run the training process 30 times
- each epoch reuses the same training data with augmentation and updates weights
- after each epoch, evaluate the model on validation data

More epochs usually help until the model starts to plateau or overfit. Overfitting is when training keeps improving while validation accuracy stops improving or gets worse.

### 9. What counts as a "good" comparison between training runs

The project supports several comparisons:

- training from scratch vs pretrained model
- torch backend vs sklearn backend
- real-only data vs real + synthetic data
- different numbers of epochs
- different amounts of augmentation

The comparison is based on the same metric: validation accuracy.

For comparison, a run with:

- `best val acc = 0.72`

is considered better than a run with:

- `best val acc = 0.60`

because it correctly classified more held-out digits.

### 10. What the scikit-learn version does

For the sklearn backend, the script does:

```python
Xa, ya = aug_set(tr, 4)
m = MLPClassifier((128, 64), max_iter=200, random_state=a.seed).fit(Xa.reshape(len(Xa), -1), ya)
print("val acc:", m.score(X[va].reshape(len(va), -1), y[va]))
```

This is the same idea:

- train an MLP on augmented samples
- evaluate accuracy on validation images
- print validation accuracy only

There is no training loss loop printed in the same way because the sklearn model is trained in one call.

### 11. Why the project saves the best checkpoint

Because validation accuracy can fluctuate across epochs, the script does not simply keep the last model. It tracks the maximum accuracy seen so far and saves the corresponding weights:

```python
if acc >= best:
    best = acc
    torch.save(model.state_dict(), a.out)
```

That is a standard practice in model training: keep the checkpoint with the best validation performance, because later epochs may not always improve generalization.

### 12. Example interpretation of a full training log

A typical log might look like this:

```text
real crops: 460
synthetic crops: 600
validation crops: 78 from 12 held-out source images
epoch  1  loss 1.421  val acc 0.219
epoch  2  loss 1.135  val acc 0.356
epoch  3  loss 0.998  val acc 0.480
epoch  4  loss 0.924  val acc 0.511
epoch  5  loss 0.881  val acc 0.553
epoch  6  loss 0.840  val acc 0.560
best val acc 0.560 -> models/digit_cnn.pt
```

Interpretation:

- the model is steadily learning
- validation accuracy improves from ~22% to ~56%
- the best model is saved at the epoch where accuracy peaked
- the script does not report a validation loss curve, only training loss and validation accuracy

### 13. Summary

In this project:

- `loss` = training loss from the last mini-batch in the epoch
- `val acc` = validation accuracy on held-out digit crops
- `epochs` = number of training cycles
- `best val acc` = highest validation accuracy observed during training
- the selected model is the checkpoint with the highest validation accuracy, not necessarily the last epoch

This is a practical and common training setup for image-classification tasks, and it is exactly how the model in this project is compared and selected.

## Summary

This repository is a compact but complete OCR pipeline for seven-segment digital meter readings. It blends three layers of intelligence:

- classical computer vision (display detection, digit segmentation, segment decoding)
- robustness heuristics (shear correction, glare suppression, fallback localization)
- optional machine learning (digit classification for ambiguous cases)

Together, these components turn a raw meter photograph into a usable numeric reading with a minimal amount of infrastructure.
