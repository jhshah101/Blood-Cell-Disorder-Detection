# Regenerating every table on the real data

Everything in `results/` was produced by the commands below on a synthetic proxy dataset. To produce the manuscript's numbers, run the same commands on Raabin-WBC and LISC. Estimated wall time on one RTX 3060: about 2 min per epoch for the ResNet-18 hybrid at 224 × 224, about 5 min per epoch for ViT-B/16; the full matrix with 8 seeds is roughly 60–80 GPU-hours and is embarrassingly parallel over seeds.

## 0. Data layout

```
data/Raabin-WBC/Train/<Basophil|Eosinophil|Lymphocyte|Monocyte|Neutrophil>/*.jpg
data/Raabin-WBC/TestA/<same five folders>/*.jpg
data/Raabin-WBC/TestB/<Neutrophil|Lymphocyte>/*.jpg
data/LISC/Main_Dataset/<baso|eosi|lymp|mono|neut|mixt>/*.bmp
data/LISC/Ground_Truth/<same folders>/*_expert.bmp
```

Folder names are case-insensitive for the external sets; the training and Test-A folders must use the five class names above (alphabetical order defines the label indices).

## 1. Split once, commit the file

```bash
python scripts/make_splits.py --config configs/default.yaml           # writes splits/raabin_train_val.json
git add splits/raabin_train_val.json
```

## 2. Train everything on matched seeds (8 seeds recommended)

```bash
SEEDS="0 1 2 3 4 5 6 7"
python scripts/run_seeds.py --seeds $SEEDS --configs configs/baselines/resnet50.yaml \
    configs/baselines/resnet18.yaml configs/baselines/densenet121.yaml configs/baselines/mobilenet_v2.yaml \
    configs/baselines/googlenet.yaml configs/baselines/efficientnet_b1.yaml configs/baselines/squeezenet1_0.yaml \
    configs/ablation/baseline_vit.yaml configs/ablation/vit_eca_ws.yaml configs/ablation/vit_eca_cf.yaml \
    configs/ablation/vit_eca_cf_focal.yaml configs/ablation/vit_eca_cf_wce.yaml configs/ablation/vit_eca_cf_alr.yaml \
    configs/ablation/vit_eca_cf_ws_alr.yaml configs/ablation/vit_eca_cf_alr_augmented.yaml \
    configs/baselines/resnet50_augmented.yaml --skip-existing
```

To run the whole ablation on the ViT-B/16 tokeniser instead, append `--set model.backbone=vit_b16` (and give the runs a distinct `output_dir`). The ALR sensitivity grid:

```bash
for b in 0.25 0.5 1.0; do for g in 0.25 0.5 1.0; do
  python scripts/run_seeds.py --seeds 0 1 2 --configs configs/ablation/vit_eca_cf_alr.yaml \
      --set experiment=sens_b${b}_g${g} --set loss.alr_beta=$b --set loss.alr_gamma=$g --skip-existing
done; done
```

## 3. Build the tables

```bash
python scripts/build_tables.py --runs runs --out results_real --seeds $SEEDS --reference baseline_resnet50 \
  --baselines baseline_resnet18 baseline_resnet50 baseline_densenet121 baseline_mobilenet_v2 baseline_googlenet baseline_efficientnet_b1 baseline_squeezenet1_0 \
  --ablation ablation_baseline_vit ablation_vit_eca_ws ablation_vit_eca_cf ablation_vit_eca_cf_focal ablation_vit_eca_cf_wce ablation_vit_eca_cf_alr ablation_vit_eca_cf_ws_alr \
  --augmented-pairs baseline_resnet50:baseline_resnet50_augmented ablation_vit_eca_cf_alr:vit_eca_cf_alr_augmented \
  --sensitivity-prefix sens_ --trajectory ablation_vit_eca_cf_alr
```

## 4. External evaluation (no fine-tuning), for every seed of the final model

```bash
for s in $SEEDS; do
  python scripts/evaluate_external.py --checkpoint runs/ablation_vit_eca_cf_alr/seed$s/best.pt --data-dir data/Raabin-WBC/TestB --preset raabin_testb
  python scripts/evaluate_external.py --checkpoint runs/ablation_vit_eca_cf_alr/seed$s/best.pt --data-dir data/LISC/Main_Dataset --preset lisc --mask-dir data/LISC/Ground_Truth
done
```

The JSON files record the mapping, excluded folders, per-class counts, the unrestricted 5-way and the restricted results with bootstrap CIs. Aggregate them over seeds into the manuscript tables with:

```bash
python scripts/aggregate_external.py --pattern "runs/ablation_vit_eca_cf_alr/seed*/external_TestB.json" --title "Raabin-WBC Test-B" --out results_real/external_testB.md
python scripts/aggregate_external.py --pattern "runs/ablation_vit_eca_cf_alr/seed*/external_Main_Dataset.json" --title "LISC" --out results_real/external_LISC.md
```

## 5. Cost under one protocol

```bash
python scripts/benchmark.py --configs configs/default.yaml configs/ablation/baseline_vit.yaml configs/baselines/*.yaml --batch-sizes 1 32 --iters 100 --warmup 20
```

State in Table 11: GPU, driver/CUDA/torch versions, batch size, precision, warm-up, iterations, and whether FLOPs or MACs are quoted.

## 6. Grad-CAM panel

```bash
python scripts/gradcam.py --checkpoint runs/ablation_vit_eca_cf_alr/seed0/best.pt \
    --predictions runs/ablation_vit_eca_cf_alr/seed0/test_predictions.csv --per-class 2 --seed 0 --only-correct --output-dir gradcam
```

Quote `gradcam/manifest.json` in the figure caption (selection rule, correctness, normalisation).

## 7. Statistics for any two models

```bash
python scripts/paired_stats.py --runs-a runs/ablation_vit_eca_cf_alr --runs-b runs/baseline_resnet50 --metric minority_f1
python scripts/paired_stats.py --predictions-a runs/ablation_vit_eca_cf_alr/seed0/test_predictions.csv --predictions-b runs/baseline_resnet50/seed0/test_predictions.csv
```
