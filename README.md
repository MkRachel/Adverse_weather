# Adverse-Weather Disaster Scene Understanding Using Deep Learning and Synthetic Data

Project ID 24AB104, AI Blend for Engineers, Saintgits College of Engineering (Autonomous).
Supervisor: Jerrin Sebastian.
Team: Mekha Rachel, Pavan Abhilash, Christina Ann Pinto, Abishek S.

## What this project does

We detect road hazards and damaged buildings in images with YOLOv8, and we test whether pre-training on synthetic CARLA images helps when the detector is later fine-tuned on real images.

Classes (YOLO IDs):

| ID | Class | Source of examples |
|---|---|---|
| 0 | `blocked_road` | CARLA (staged blockages) and real images |
| 1 | `debris` | CARLA (staged props) and real images |
| 2 | `damaged_building` | Real images only (CARLA has no damaged-building assets) |

## Main finding (honest summary)

In our setup, synthetic pre-training gave **no measurable benefit** over training on real images alone.

- A model trained only on synthetic data scored 0 on the real test set.
- The synthetic-then-real model was comparable to the real-only model on validation (mAP50 0.474 vs 0.471) and slightly lower on the test set (0.367 vs 0.391).
- The real-only baseline was slightly better on test. The difference is small and may be within run-to-run noise.

Results on the real test set (146 images, 327 boxes), single run each, seed 42:

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Synthetic only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Synthetic then real fine-tuned (30 epochs) | 0.381 | 0.446 | 0.411 | 0.367 | 0.166 |
| Real-only baseline (50 epochs) | 0.515 | 0.404 | 0.453 | 0.391 | 0.197 |

Per-class mAP50 on the real test set:

| Model | blocked_road | debris | damaged_building |
|---|---|---|---|
| Synthetic only | 0.000 | 0.000 | 0.000 |
| Synthetic then real fine-tuned | 0.284 | 0.180 | 0.639 |
| Real-only baseline | 0.268 | 0.256 | 0.649 |

On the synthetic test split (118 images) the synthetic model scored precision 0.760, recall 0.648 and mAP50 0.675 (blocked_road 0.791, debris 0.558). The data pipeline works, but the model did not transfer to real images.

An equal-epoch rerun (50 epochs for the fine-tuned model) is pending. Update the table when it is done.

### Limitations

- Unequal training budgets: 30 epochs for fine-tuning against 50 for the baseline.
- One run per model, so small differences are not reliable.
- The debris class has only 28 real test images (102 boxes).
- The real set merges several public sources with inconsistent class definitions. Source and licence list: TODO.
- The synthetic classes differ in meaning from the real ones (staged sideways cars and props against real barricades, fallen trees and rubble).
- The synthetic model never saw `damaged_building`, which is about half of the real test boxes.
- The optimizer was `auto`, which chose AdamW (lr 0.001429) and ignored the `lr0` we set.

Possible reasons for the failed transfer (hypotheses, not tested): class-meaning mismatch, visual gap between CARLA and real photos, and a real training set (706 images) large enough that pre-training added little.

## Synthetic dataset (CARLA 0.9.16)

- 1,200 RGB images, 1280 x 720, 240 per weather: clear, rain, fog, night, rainy night.
- Three camera modes: ground (1.2-2.2 m), elevated (4-10 m) and UAV-style (15-40 m, angled, not top-down).
- Split: 948 train, 134 validation, 118 test.
- Automatic YOLO labels from projected 3D boxes. `blocked_road` is one combined box around the staged blockage; `debris` is one box per prop.
- Known label issues: 179 tiny debris boxes (168 from UAV views), 8 training images with debris but no blockage box, and some objects hard to see in fog and at night. Review was partial.

The dataset is not stored in this repository. Link: TODO.

## Real dataset

1,001 images (706 train, 149 validation, 146 test), cleaned by converting 16 polygon labels to boxes and moving 7 images so no near-duplicate scene spans two splits. Sources and licences: TODO.

## Datasets, Model Weights and Experiment Outputs

The project artifact folder is available here:

[Project artifacts on Google Drive](https://drive.google.com/drive/folders/13NTni0nVUvqXvd2HAJPZn20LLxsq750o)

The folder contains the synthetic and cleaned real datasets, trained model
weights, results tables, evaluation plots and demonstration material.

Key files:
- synthetic.zip: synthetic training dataset
- real.zip: cleaned real dataset
- synthetic_best.pt: synthetic-trained model
- finetuned_best.pt: synthetic-then-real model
- real_only_best.pt: real-only baseline
- results_table.csv: experimental results
- runs/: training, evaluation and prediction outputs
- synthetic_label_flags.csv: synthetic annotation audit

check_images.zip contains generated annotation previews, not model predictions.
real_original.zip is the original real-data backup, not the cleaned training version.

Real-data source attribution and licence information are being confirmed.
Access to third-party datasets is subject to their original usage conditions.



## Repository layout

```text
scripts/capture_scenes.py        # earlier pilot: weather-only capture
scripts/capture_hazards.py       # hazard scenes, camera modes, auto labels
scripts/capture_hazards_v1.py    # first version, kept for history
docs/synthetic_samples/          # sample previews with generated boxes
notebooks/                       # Colab training notebook
results/                         # results_table.csv
```

The sample previews show generated annotations, not model predictions.

## How to run the capture script

1. Start the CARLA server: `./CarlaUE4.sh -quality-level=Low`
2. In another terminal: `source venv/bin/activate`, then `pip install -r requirements.txt`
3. `python scripts/capture_hazards.py 1200`
4. To resume after an interruption: `python scripts/capture_hazards.py 1200 <start_scene_number>`

Output goes to `hazard_dataset/` (ignored by Git).

## Training setup

YOLOv8s, 640 px, batch 16, seed 42, Ultralytics 8.4.187, PyTorch 2.11.0, Google Colab Tesla T4. The synthetic model trained 50 epochs from `yolov8s.pt`; the fine-tune started from the synthetic weights.

## Status

Completed: synthetic data generation, real dataset cleaning, three models trained and evaluated, demo on real test images. Pending: equal-epoch rerun, source and licence list, final report.
