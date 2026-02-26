# xai-lab

A small, config-driven sandbox for **training deep learning models** (starting with **ResNet**) and producing **Explainable AI (XAI)** outputs (starting with **vanilla saliency / input gradients**, then Grad-CAM, etc.).

Initial focus: **facial emotion classification** using a CK+ (Kaggle-packaged) folder dataset.  
Next: noisier datasets (e.g., FER2013 / FER+) to build data wrangling skills and compare how XAI behaves under noise.

## What to do right now

✅ Train a pretrained ResNet on a folder-based emotion dataset  
✅ Evaluate accuracy (and optionally F1/confusion matrix later)  
✅ Generate simple explanation maps (saliency) for test images  
✅ Keep runs reproducible via YAML configs (great for Git + experiment tracking)

## Repo layout (high-level)

```text
configs/        # YAML configs (data, models, explainers, experiments)
scripts/        # CLI entrypoints: preprocess/split/train/eval/explain
src/xai_lab/    # reusable library code
data/           # raw & processed datasets (ignored by git)
artifacts/      # outputs: checkpoints, heatmaps, reports (ignored by git)
tests/          # smoke tests and quick checks
notebooks/      # Jupyter notebooks
```
> Note: Each notebook answers one question to avoid rotting.

## Setup

> Note: Scripts work out of the box because they add `src/` to `PYTHONPATH`. Package-style imports will be added later.

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### 2) Install dependencies

Pick the right PyTorch install for your machine from:
[Pytorch Get Started Locally](https://pytorch.org/get-started/locally/) and update torch-requirements.txt

```bash
pip install -r requirements.txt
pip install -r torch-requirements.txt
```

## Data

This project assumes the dataset is organized like:

```text
data/raw/
  dataset1/
    dataset1_subfolder1/
    dataset2_subfolder2/
    dataset3_subfolder3/
  dataset2/
    dataset1_subfolder1/
    dataset2_subfolder2/
    dataset3_subfolder3/
```

> Note: This repo does **not** include datasets. Keep all datasets in `data/` (gitignored).

## Quickstart (recommended order)

### Step 1 — Create manifest + splits (data wrangling)

Generate a metadata manifest and stratified train/val/test splits:

> Filename can be anything that describes the function of the file

```bash
python scripts/make_ckplus_splits.py
```

Expected outputs:

```text
data/processed/ckplus/metadata.csv
data/processed/ckplus/splits/train.csv
data/processed/ckplus/splits/val.csv
data/processed/ckplus/splits/test.csv
```

### Step 2 — Train (ResNet)

```bash
python scripts/train.py --config configs/experiments/exp001_resnet18_saliency_ckplus.yaml
```

Expected outputs (example):

```text
artifacts/reports/exp001_resnet18_saliency_ckplus/
  best.pt
  train_log.jsonl   (optional if you log)
```

### Step 3 — Evaluate
```bash
python scripts/evaluate.py --run_dir artifacts/runs/exp001_resnet18_saliency_ckplus
```
Evaluates the best checkpoint model on the test split and produces reports.

```text
artifacts/reports/exp001_resnet18_saliency_ckplus/<run_id>/
  test_summary.json
  test_confusion_matrix.csv
```

### Step 3.5 — Visualize (Confusion Matrix)

```bash
python scripts/make_report_plots.py --run_dir artifacts/runs/exp001_resnet18_saliency_ckplus --reports_dir artifacts/reports/exp001_resnet18_saliency_ckplus
```

Expected outputs (example):

```text
artifacts/reports/exp001_resnet18_saliency_ckplus/<run_id>/
  test_confusion_matrix.png
  train_loss.png
  val_accuracy.png
  val_loss.png
  val_macrof1.png
```

### Step 4 — Explain (Saliency)

```bash
python scripts/explain.py --run_dir artifacts/runs/exp001_resnet18_saliency_ckplus --split test --num_each 8
```

Expected outputs (example):

```text
artifacts/reports/exp001_resnet18_saliency_ckplus/<run_id>/xai/saliency/
  high_conf/
    sample01.png
    sample02.png
    sample03.png
  low_conf/
    sample01.png
    sample02.png
    sample03.png
```

What to do if signals from explanations are bad:
- tighten augmentations (reduce crop randomness / rotation)
- confirm preprocessing (RGB conversion + normalization)
- check for dataset leakage/artifacts (e.g., one class has different resolution)
- use balancing (sampler) and check confusion matrix


## Configs (YAML)

Experiments are defined in YAML. An experiment YAML typically references:

* a run YAML
* a data YAML
* a model YAML
* a train YAML
* an augment YAML

Example path:

```text
configs/experiments/exp001_resnet18_saliency_ckplus.yaml
```

A separate YAML file for explaination will be added later to have better flexibility for explaining the model.

Benefits:

* reproducibility (configs committed to git)
* easy experiment swapping without editing code
* later you can store the resolved config with results (e.g., in MongoDB)



## First XAI method: Saliency (Input Gradients)

**Saliency** = gradient of the chosen class score w.r.t. input pixels.

Why start here:

* simplest implementation (one backward pass)
* good for verifying the end-to-end “train → explain → save” workflow

Known limitation:

* can look noisy. Upgrade path: **SmoothGrad** → **Grad-CAM**.



## Outputs

This repo writes outputs under `artifacts/` (gitignored):

* `artifacts/reports/<experiment_name>/<run_name>/` - Contains user-facing reports and visualizations
* `artifacts/runs/<experiment_name>/<run_name>/` - Contains model checkpoints, logs, and other run artifacts


## Roadmap (suggested)

1. **exp001**: ResNet18 + Saliency on CK+ (Kaggle pack)
2. **exp002**: ResNet18 + Grad-CAM (nicer maps)
3. Add evaluation:

   * confusion matrix
   * macro-F1
4. Move to noisy dataset:

   * FER2013 (wrangling + robustness)
   * FER+ (better labels / distributions)
5. Store results to MongoDB:

   * run config + metrics + explanation paths

## Results
<img src="docs/saliency_high_conf_prediction.png" alt="Saliency High Confidence Prediction">


<img src="docs/saliency_low_conf_prediction.png" alt="Saliency Low Confidence Prediction">

_Analysis to follow after the GradCAM implementation_

## Notes on dataset licensing

CK+/FER datasets often come with usage restrictions depending on the source. If you plan to publish results, verify the dataset’s license/terms from the original provider or the hosting platform.



## References

* CK+ Dataset: Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete dataset for action unit and emotion-specified expression. IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(12), 2260-2277.
* Grad-CAM: [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
* FER2013 (Kaggle example): [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
* FER+ (Microsoft): [https://github.com/microsoft/FERPlus](https://github.com/microsoft/FERPlus)




## License

MIT - Please see `LICENSE`.

Datasets are **not** included and remain under their respective licenses/terms.

___README.md was generated by AI and edited by the Owner___