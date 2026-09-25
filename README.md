# Hybrid ViT-ECA-CF with Adaptive Loss Reweighting for Imbalanced White Blood Cell Classification

Reference implementation for the manuscript *A Hybrid ViT-ECA-CF Framework
with Adaptive Loss Reweighting for Imbalanced White Blood Cell
Classification* (Jabeen, Shah, Afzal, Awais; under review).

The repository implements **exactly** the method the manuscript describes,
under a **leakage-safe** experimental protocol, with every reported quantity
produced by one documented function:

| Manuscript element                                  | Implementation                                              |
|-----------------------------------------------------|-------------------------------------------------------------|
| Image-wise standardisation, Eq. (1) / ImageNet stats | `wbc/models/hybrid.py::InputNormalization` (inside the model) |
| ResNet-18 structural branch, Eqs. (2)-(3)           | `wbc/models/hybrid.py::ResNet18Features`                    |
| ECA with adaptive kernel, Eqs. (4)-(6)  (k = 5)     | `wbc/models/eca.py`                                         |
| Colour features, Eqs. (7)-(11), 515-d fused tokens  | `wbc/models/color_features.py`, `HybridViTECACF`            |
| Transformer encoder + [CLS] head, Eq. (12)          | `wbc/models/transformer.py`, `HybridViTECACF`               |
| Initial class weights, Eq. (13)                     | `wbc/losses.py::initial_class_weights`                      |
| Adaptive loss reweighting, Eqs. (14)-(15), Alg. 1   | `wbc/losses.py::AdaptiveLossReweighting`                    |
| Focal loss / weighted CE / weighted sampling         | `wbc/losses.py`, `wbc/data/datasets.py`                     |
| Metrics incl. majority / minority F1, Eq. (16)      | `wbc/metrics.py`                                            |
| Repeated seeds, Wilcoxon / permutation / McNemar    | `wbc/stats.py`, `scripts/run_seeds.py`, `scripts/paired_stats.py` |
| Test-B and LISC external evaluation                 | `wbc/data/external.py`, `scripts/evaluate_external.py`      |
| Parameters, FLOPs, latency (Table 11)               | `wbc/benchmark.py`, `scripts/benchmark.py`                  |
| Grad-CAM panel (Table 9)                            | `wbc/explain.py`, `scripts/gradcam.py`                      |

## Architecture (one computational graph)

```
raw RGB [0,1] ──► InputNormalization ──► ResNet-18 (conv1 … layer4) ──► (B,512,7,7)
      │                                          │
      │                                          ▼
      │                                  ECA, k = |log2(512)/2 + 1/2|_odd = 5
      │                                          │
      │                                          ▼
      │                                  49 structural tokens × 512
      │
      └──► colour descriptors on the same 7×7 tiling
           (mean, sd, skewness, kurtosis + 8-bin histogram for R,G,B,H,S,V,L,a,b
            = 9 × 12 = 108 values) ──► Linear ──► 3 values per token
                                                 │
                              concat ──► 515-d fused tokens ──► Linear → 768
                                                 │
                    [CLS] + positional embedding ──► 12 × (12-head pre-LN block, MLP 3072, GELU)
                                                 │
                                         LayerNorm ──► Linear ──► 5 logits
```

`model.backbone: vit_b16` swaps the ResNet-18 tokeniser for the ImageNet
ViT-B/16 patch embedding + encoder (ECA on the 768 channels, colour fusion
initialised to the identity), so the same code supports both descriptions
that appeared in the manuscript. The 515-dimensional fused token is projected
to the encoder width because 515 is not divisible by the number of heads.

## Experimental protocol

1. **Split once, reuse everywhere.** `scripts/make_splits.py` carves a
   stratified validation split (15 % by default) out of the *predefined
   Raabin-WBC training set* with a fixed seed and writes every path to
   `splits/raabin_train_val.json`. Commit that file. A `path,group` CSV
   (slide / patient id) switches the split to group-aware.
2. **Select on validation, test once.** `train.selection_metric`
   (validation macro-F1 by default) chooses the checkpoint; the test folder
   is opened once, after training, by the final evaluation. The test loader
   is never iterated during training (this is asserted by
   `tests/test_pipeline.py`).
3. **Repeat over seeds with matched splits.** `scripts/run_seeds.py` trains
   every configuration on the same seeds, reports mean ± SD, the paired
   differences, the exact Wilcoxon signed-rank test (and the smallest p-value
   the design can produce: 2 / 2ⁿ, i.e. 0.0625 for n = 5), the paired t-test,
   the exact sign-flip permutation test, Cohen's d_z, a bootstrap CI, and the
   per-seed image-level McNemar test.
4. **External sets without fine-tuning.** `scripts/evaluate_external.py`
   maps external folder names onto the training classes, counts every
   excluded folder, crops LISC cells to their masks, and reports both the
   unrestricted 5-way and the restricted (present-classes-only) results.
5. **Cost under one protocol.** `scripts/benchmark.py` measures parameters,
   FLOPs / MACs and warmed-up, synchronised latency for every model on one
   machine and stores the protocol with the numbers.

## Quick start

```bash
pip install -r requirements.txt          # or: pip install -e .[server,dev]
python -m pytest                         # 46 tests, CPU, < 2 min

# data/Raabin-WBC/{Train,TestA}/<Class>/*.jpg  (ImageFolder layout)
python scripts/make_splits.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --seed 0
python scripts/evaluate.py --checkpoint runs/vit_eca_cf_alr/seed0/best.pt --data-dir data/Raabin-WBC/TestA
python scripts/evaluate_external.py --checkpoint runs/vit_eca_cf_alr/seed0/best.pt --data-dir data/Raabin-WBC/TestB --preset raabin_testb
python scripts/evaluate_external.py --checkpoint runs/vit_eca_cf_alr/seed0/best.pt --data-dir data/LISC/Main_Dataset --preset lisc --mask-dir data/LISC/Ground_Truth
python scripts/benchmark.py --configs configs/default.yaml configs/baselines/resnet50.yaml
python scripts/gradcam.py --checkpoint runs/vit_eca_cf_alr/seed0/best.pt --predictions runs/vit_eca_cf_alr/seed0/test_predictions.csv --per-class 2 --only-correct
```

Every hyper-parameter lives in a YAML file under `configs/` and can be
overridden with `--set section.key=value`; the resolved configuration is
written next to every checkpoint and results file.

Three helpers turn run directories into manuscript material:
`scripts/build_tables.py` (Tables 4, 5–8, 10, paired statistics, ALR
sensitivity grid and weight trajectories), `scripts/aggregate_external.py`
(Test-B / LISC tables averaged over seeds) and `scripts/make_synthetic_data.py`
(a procedurally generated dataset with the Raabin-WBC structure, acquisition
shift and LISC-style masks, for exercising the whole protocol without the real
data).

### Reproducing the tables

| Table                       | Command                                                                                                  |
|-----------------------------|----------------------------------------------------------------------------------------------------------|
| CNN baselines (Table 4)     | `python scripts/run_seeds.py --configs configs/baselines/*.yaml --seeds 0 1 2 3 4 5 6 7`                 |
| Ablation (Table 5 / 6 / 8)  | `python scripts/run_seeds.py --configs configs/baselines/resnet50.yaml configs/ablation/*.yaml --seeds 0 1 2 3 4 5 6 7` |
| Augmented vs not (Table 10) | `python scripts/run_seeds.py --configs configs/baselines/resnet50.yaml configs/baselines/resnet50_augmented.yaml configs/ablation/vit_eca_cf_alr.yaml configs/ablation/vit_eca_cf_alr_augmented.yaml --seeds 0 1 2 3 4 5 6 7` |
| Cost (Table 11)             | `python scripts/benchmark.py --configs configs/default.yaml configs/ablation/baseline_vit.yaml configs/baselines/*.yaml --batch-sizes 1 32` |

Eight seeds are suggested because an exact two-sided Wilcoxon test needs at
least six non-zero paired differences to reach p < 0.05 and eleven to reach
p < 0.001.

### Ablation configurations

| Config                                   | Backbone   | ECA | CF | Sampler  | Loss  |
|------------------------------------------|------------|-----|----|----------|-------|
| `ablation/baseline_vit.yaml`             | vit_b16    | –   | –  | shuffle  | CE    |
| `ablation/vit_eca_ws.yaml`               | cnn_hybrid | ✓   | –  | weighted | CE    |
| `ablation/vit_eca_cf.yaml`               | cnn_hybrid | ✓   | ✓  | shuffle  | CE    |
| `ablation/vit_eca_cf_focal.yaml`         | cnn_hybrid | ✓   | ✓  | shuffle  | focal |
| `ablation/vit_eca_cf_wce.yaml`           | cnn_hybrid | ✓   | ✓  | shuffle  | WCE   |
| `ablation/vit_eca_cf_alr.yaml`           | cnn_hybrid | ✓   | ✓  | shuffle  | ALR   |
| `ablation/vit_eca_cf_ws_alr.yaml`        | cnn_hybrid | ✓   | ✓  | weighted | ALR   |
| `ablation/vit_eca_cf_alr_augmented.yaml` | cnn_hybrid | ✓   | ✓  | shuffle  | ALR + flip/rotate |

Add `--set model.backbone=vit_b16` to run any row on the ViT-B/16 path.

## Adaptive loss reweighting, as implemented

* Initial weights (Eq. 13): `w_c = (N_max / N_c) ** alpha`, rescaled to mean 1.
* After every epoch, from the training forward passes: `L_c` = mean unweighted
  cross-entropy of class *c*, `P_c` = mean true-class probability over the
  correctly classified samples of class *c*.
* Update (Eq. 14): `w_c ← w_c (1 + β L_c)(1 + γ (1 − P_c))`, then **rescaled
  to mean 1**, then **clipped to [w_min, w_max]**, optionally smoothed.
  Without the rescaling every factor is ≥ 1 and all weights can only grow;
  with it, a class whose factor is below the weighted average factor sees its
  weight *decrease*, which is the behaviour the manuscript claims.
* The loss (Eq. 15) is the weighted cross-entropy with the current weights.
* Defaults: α = 1, β = 0.5, γ = 0.5, w ∈ [0.2, 5]. Every epoch's `L_c`, `P_c`,
  factor and weight vector are written to `history.csv` and `results.json`.

## Metric definitions

Macro-F1 = unweighted mean of per-class F1; balanced accuracy = mean per-class
recall; majority (minority) F1 = unweighted mean of per-class F1 over the
classes not listed (listed) in `data.minority_classes`; F1 gap = majority −
minority; F1 balance ratio = minority / majority. `evaluate.py` verifies the
arithmetic identities on every results file.

## Output of a run

```
runs/<experiment>/seed<k>/
  config.yaml            resolved configuration
  history.csv            per-epoch lr, losses, validation metrics, ALR weights
  best.pt, last.pt       checkpoints (weights + classes + config)
  val_predictions.csv    per-image validation predictions
  test_predictions.csv   per-image test predictions (for McNemar)
  results.json           metrics, bootstrap CIs, counts, timing, environment
```

## Serving and demo

`Software/Backend` is a FastAPI server that rebuilds the network from the
checkpoint and exposes `/health`, `/model-info` and `/predict`;
`Software/Frontend` is a Vite + React client that reads the model facts from
the server. Both carry a research-prototype disclaimer. See their READMEs.

## Legacy scripts

`WBC_Without.py` and `main.py` are kept as entry points because the manuscript
cites them; they now delegate to `scripts/train.py`. The behaviour of the
original files is documented in `configs/legacy/*.yaml`, with two exceptions
that are deliberately **not** reproduced: test-set model selection
(`WBC_Without.py`) and the untrained, unregisterable positional embedding
(`main.py`). `Software_code.ipynb` (ResNet-50 baseline with per-batch
re-estimated class weights) is superseded by `configs/baselines/resnet50.yaml`.

## Data

Raabin-WBC (Mousavi Kouzehkanan et al., *Scientific Reports* 2022,
doi:10.1038/s41598-021-04426-x) provides the training set, Test-A and the
two-class Test-B; LISC (Rezatofighi & Soltanian-Zadeh, 2011) is the second
external set. Both are public; no private data are used. Place them as
`ImageFolder` trees and point `data.train_dir` / `data.test_dir` at them.

## Citation

See `CITATION.cff`.

## License

Academic and research use.
