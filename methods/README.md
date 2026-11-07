# Data preparation and model evaluation

## Geometry, boundary conditions and sampling

Each geometry is represented by a rooted directed vascular tree. Node positions, radii, edge lengths and terminal nodes define the physical network. The inlet flow is a prescribed waveform and the outlet rows contain proximal resistance, compliance and distal resistance in matching outlet-node order. `configuration.json` supplies density, viscosity, initial pressure, output window, distal-pressure curves and cap geometry for the RI–RCR model.

CFD pressure is a cross-sectional mean at node planes; directed-edge flow is a cross-sectional volumetric flux. Boolean masks select the positions used in supervised loss and evaluation. The 64 CFD sample times are listed in `phases.npz`. RI–RCR arrays contain 65 frames, with the additional frame one cardiac period after the first. The first 64 RI–RCR frames align with the CFD samples.

The RI–RCR state is initialized at simulation time zero: free terminal-flow coordinates are zero and capacitor pressures use `initial_pressure_mmHg`. The network is continuously advanced to the specified output window. Distal pressure is specified separately by `Pd_curves_dyn_cm2`. The `initial_states` method also provides alternative initializations for numerical studies; the dataset responses use the prescribed-pressure initialization described here.

The observation-compatible flow is the RI–RCR transport flow minus `boundary_offset_cm3_s`. The offset represents inlet-related geometric sampling effects and is stored alongside the sampling geometry. `local_constraints.npz` contains additional input-derived local flow targets for 15 geometries and 20 sections.

## Neural input construction

The static node, edge and graph-wide encoders process 14, 18 and 9 channels. Temporal decoders receive 10 node and 12 edge channels at each of 64 times. Common geometric/forcing channels and RI–RCR channels are assembled in the order specified by `data/FEATURE_SPEC.md`.

`scripts/public_data.py` provides `fit_statistics(training)`, `predictive_arrays(case_id, stats)`, `target_arrays(case_id, stats)` and `physical_inputs(case_id)`. Statistics are computed from the fitting cases listed for the selected fold. The standardized baselines, pressure differences, flow offsets and targets are rebuilt with these statistics. Predictive arrays are loaded separately from labels and masks.

## Model definitions and fitting settings

`code/models.py` defines the neural networks and the supervised/local-flow loss. `methods/model_configs.json` gives the exact PhyTCR-Net and D1 configurations. PhyTCR-Net uses eight independently parameterized, graph-wide conditioned blocks (width 160) and three mean–wave temporal decoders. D1 is the direct neural prediction comparator, using its specified adjacency-based graph blocks and direct P/Q outputs. Both models receive their configured common and physical input channels.

Each fold/seed fit starts from a fresh initialization. `methods/training_protocol.json` gives AdamW parameters, the warmup/cosine schedule, case accumulation, gradient clipping and stopping criteria. Inner-validation joint standardized mean-squared error selects one shared P/Q checkpoint. `methods/model_selection.json` lists the selected epoch, normalization statistics and prediction selection identifier for each fit. The outer-test predictions are organized by model, fold, seed and geometry.

## Evaluation and results

Run `python -B scripts/replay_scores.py` from the root directory to recompute the 420 saved prediction scores and the primary comparison. Per-geometry RMSE is calculated over 64 times and valid sections. For group-level summaries, the three seed-specific metrics are averaged within a geometry, then geometries within a provenance group are averaged, and the 63 group values receive equal weight.

`methods/evaluation_protocol.json` gives the definitions of pressure/flow RMSE, mean and waveform errors, peak-time error, adjacent pressure-difference error, local flow consistency and paired group bootstrap. `results/primary_metrics.json` contains the model means and primary/sensitivity confidence intervals. The bootstrap resamples provenance groups within each fixed outer fold and characterizes these saved predictions under the specified grouping.

## Running the numerical and neural modules

Basic reading and metric replay require Python and NumPy. RI–RCR response replay additionally uses SciPy:

```text
python -B scripts/replay_rom.py --cases aorta_017 aorta_004 aorta_026
```

The neural module requires PyTorch. It can be instantiated from the root directory:

```python
import sys
sys.path.insert(0, "scripts")
import torch
from public_data import build_model, fit_statistics, predictive_arrays, read_json, ROOT

fold = read_json(ROOT / "splits/five_fold.json")["folds"][0]
stats = fit_statistics(fold["training"])
x = {k: torch.from_numpy(v) for k, v in predictive_arrays("aorta_017", stats).items()}
model = build_model("PhyTCR-Net").eval()  # newly initialized model
with torch.inference_mode():
    output = model(x)
```

This snippet demonstrates the model interface. The saved study outputs are the arrays under `results/outer_predictions/`; their associated model-selection identifiers are listed in the result index. Software versions and component requirements are provided in `methods/environment.json`.
