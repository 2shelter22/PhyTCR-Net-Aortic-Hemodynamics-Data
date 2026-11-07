# Outer-test predictions and metrics

The prediction collection contains PhyTCR-Net and D1 outputs from five outer folds and three training seeds (7, 19 and 43). Each fit predicts its 14 outer-test geometries, giving 420 model–geometry results.

## Directory structure

Prediction and score files use the following layout:

```text
results/outer_predictions/PhyTCR-Net/fold_0/seed_7/aorta_004/predictions.npz
results/outer_predictions/PhyTCR-Net/fold_0/seed_7/aorta_004/score.json
results/outer_predictions/D1/fold_0/seed_7/aorta_004/predictions.npz
```

`prediction_index.json` lists the relative prediction/score paths and model-selection identifier for every record. `methods/model_selection.json` associates each identifier with its fold, seed, selected epoch and fitting-subset normalization statistics.

`predictions.npz` contains `pressure_mmHg` with shape `(64,N)` and `flow_cm3_s` with shape `(64,E)`. Flow in cm³/s is numerically mL/s. Geometry-specific CFD labels and masks are under `data/cases/<case_id>/labels.npz`.

## Per-geometry scores

The `P` and `Q` sections in `score.json` give RMSE, standardized RMSE, MAE, mean-error RMSE, demeaned-waveform RMSE, waveform squared-error ratio, signed mean error and the number of valid sections. Pressure uses mmHg and flow uses mL/s. `S` is one half of the sum of the squared standardized P/Q RMSEs. Additional entries give adjacent pressure-difference error, circular peak-time error and local flow-consistency RMS where local targets are available.

## Main comparison

Run from the dataset root:

```text
python -B scripts/replay_scores.py
```

The program recalculates all per-geometry scores, averages the three seed-specific RMSEs for each geometry, averages geometries within each provenance group, and gives equal weight to the 63 groups. The paired difference is D1 minus PhyTCR-Net. Confidence intervals use 20,000 within-fold paired provenance-group bootstrap samples; unstratified group intervals are included separately as sensitivity results.

The numerical summary is stored in `primary_metrics.json`. Metric formulas, units and resampling settings are in `methods/evaluation_protocol.json`.

## Local flow-consistency fields

`local_flow_consistency.sections` counts available local sections. When this count is nonzero, `local_flow_consistency.rms_mL_s` gives the RMS difference between predicted flow and `local_target_raw` for each `local_edges` entry, in mL/s. If no local sections are available, only `sections: 0` is present. These local metrics describe the available sampled sections, not whole-graph mass-conservation projection. Historical comparison levels remain in the original local-data arrays as optional metadata; they are not used to define the public scores or the primary P/Q comparison.
