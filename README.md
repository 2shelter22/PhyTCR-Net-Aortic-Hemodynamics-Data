# PhyTCR-Net: Aortic Pressure and Flow Waveform Dataset

## Download the complete dataset

- [Dataset v1.0.1 ZIP](https://github.com/2shelter22/PhyTCR-Net-Aortic-Hemodynamics-Data/releases/download/v1.0.1/PhyTCR_Net_Aortic_Hemodynamics_Data_v1.0.1.zip)
- [Release page](https://github.com/2shelter22/PhyTCR-Net-Aortic-Hemodynamics-Data/releases/tag/v1.0.1)
- ZIP size: **62,075,840 bytes**; **1,573 files** after extraction.
- ZIP SHA256: `ebddc35055bf5dead5ad425ccfc7720a54b0a523279ee45a245697f78265a581`.

Download and extract the release ZIP before running the quick-start commands below. This Git repository contains the lightweight documentation, splits, provenance and reproducibility interfaces; the complete numerical data and predictions are in the release ZIP. Run verification from the **extracted dataset root**, not this lightweight Git checkout. The ZIP contains its own complete `SHA256SUMS.json`.

### Overview

This dataset provides cross-sectional pressure and volumetric flow-rate waveforms, directed vascular-tree representations, boundary-condition data, and evaluation resources for 70 aortic geometries. The geometries are organized into 63 provenance-based groups used for source-grouped five-fold cross-validation. The data support research on hemodynamic reduced-order modeling, topology-aware learning, and full-cycle waveform prediction.

The CFD targets contain 64 temporal samples per cardiac cycle. The accompanying RI–RCR computations provide physics-based pressure and flow baselines. Saved outer-test predictions are provided for PhyTCR-Net and its direct neural prediction comparator, together with the splits and scoring routines used for the main comparison.

### Dataset contents

| Path | Description |
|---|---|
| `data/cases/` | Per-geometry vascular graph, inlet/outlet boundary conditions, CFD section labels, sampling geometry, time samples, reduced-order outputs, and input features. |
| `data/cases.json` | Geometry identifiers, provenance-group identifiers, and basic dataset statistics. |
| `splits/five_fold.json` | Training, validation, and outer-test cases for each fold, plus training seeds. |
| `results/` | Saved outer-test pressure/flow predictions and evaluation summaries for the main model comparison. |
| `code/` and `scripts/` | Reduced-order and neural-model definitions, dataset readers, and evaluation utilities. |
| `DATA_DICTIONARY.md` | Array names, shapes, units, masks, and feature definitions. |
| `SOURCES.md` and `licenses/` | Upstream data provenance, citation, and reuse information. |

### Quick start

The basic data reader and metric replay use Python and NumPy. From the dataset root, run:

```bash
python -B scripts/verify_data.py
python -B scripts/replay_scores.py
```

A single-case example is available in `examples/load_case.py`:

```bash
python -B examples/load_case.py --case aorta_017
```

For the RI–RCR numerical replay, install SciPy in the selected Python environment and run:

```bash
python -B scripts/replay_rom.py
```

### Data conventions

Pressure targets in `labels.npz/pressure_node` are stored in dyn/cm². Values in mmHg are obtained by dividing by 1333.22. Flow targets in `labels.npz/flow_edge_cm3_s` are stored in cm³/s, numerically equal to mL/s. Boolean node and edge masks identify the sections included in supervised evaluation.

Each CFD label series contains 64 temporal samples. The accompanying `phases.npz` specifies the sample times. Vascular graphs use zero-based node indices and directed edges from inlet to outlets. Additional arrays and their exact dimensions are described in `DATA_DICTIONARY.md`.

### Evaluation

The five outer folds use provenance-based group assignments. Models were evaluated using three training seeds. The provided prediction archive contains 420 outer-test model–case results across two model variants. Evaluation uses valid sections and reports RMSE in mmHg for pressure and mL/s for flow. The source-group aggregation and model configurations are documented in the accompanying evaluation files.

### Provenance and citation

Source geometries originate from the Vascular Model Repository (VMR). See `SOURCES.md` for the source paper, case provenance, and original notices. When reusing this dataset, cite both the VMR source and the associated PhyTCR-Net study or the dataset record, as applicable. The dataset repository record supplies the preferred dataset citation when published.

### License and attribution

Original study-generated data and documentation: **CC BY-NC 4.0**. Original software: **PhyTCR Noncommercial Attribution Software License 1.0**. Noncommercial use only; attribution is required. Preserve source credits and identify modifications. VMR source material retains its original terms.

See [LICENSES.md](LICENSES.md) for component terms and [ATTRIBUTION.md](ATTRIBUTION.md) for credits. Official per-geometry archive identifiers and links are listed in [data/provenance_index.json](data/provenance_index.json).
