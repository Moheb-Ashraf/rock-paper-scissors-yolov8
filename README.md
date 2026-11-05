# Rock Paper Scissors Detection with YOLOv8

Object detection model that recognizes **Rock**, **Paper** and **Scissors** hand gestures, built by fine-tuning a pretrained **YOLOv8n** (Ultralytics) on a public Roboflow dataset. The whole pipeline lives in one Google Colab notebook.


## What the notebook does

1. Downloads the [Rock Paper Scissors SXSW](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw) dataset (version 1) from Roboflow in YOLOv8 format.
2. Fine-tunes `yolov8n.pt` for 100 epochs at 640×640.
3. Plots the training curves and metrics (`results.png`).
4. Runs inference with `best.pt` on random validation images and displays the predictions.

## Results

Evaluated with the best weights on the validation set (224 images, 169 instances), trained on a Tesla T4 in about 0.36 hours:

| Class    | Precision | Recall | mAP50 | mAP50-95 |
|----------|-----------|--------|-------|----------|
| **All**  | 0.984     | 0.948  | 0.969 | 0.682    |
| Paper    | 0.974     | 0.918  | 0.928 | 0.672    |
| Rock     | 1.000     | 0.969  | 0.985 | 0.668    |
| Scissors | 0.978     | 0.956  | 0.993 | 0.707    |

Inference speed: ~5.2 ms per image (plus ~1 ms preprocess and ~3 ms postprocess).

Sample predictions and training curves are saved in the notebook outputs.

## How to run

1. Click the **Open in Colab** badge above.
2. Set the runtime to **GPU** (Runtime → Change runtime type).
3. Get a free API key from [Roboflow](https://app.roboflow.com/) (Settings → API Keys).
4. Run the cells in order. You will be prompted for the key; it is never stored in the notebook.

> Paths such as `/content/...` assume a Colab environment. To run locally, install `requirements.txt` and adjust the paths.

## Tech stack

Python · Ultralytics YOLOv8 · Roboflow · OpenCV · Matplotlib

## License

The dataset is provided by Roboflow Universe under its own license; check the dataset page before reuse.
