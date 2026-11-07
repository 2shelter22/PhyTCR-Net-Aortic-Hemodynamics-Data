# Data sources and attribution

The vascular geometries used in this dataset originate from the Vascular Model Repository (VMR): https://www.vascularmodel.com/ . Case identifiers and provenance-group assignments are listed in `data/cases.json` and `splits/five_fold.json`.

The dataset provides derived vascular-tree representations, prescribed inlet and outlet boundary information, cross-sectional CFD pressure and flow-rate targets, RI–RCR baseline responses, and neural-model outer-test predictions. The data dictionary identifies these components and their physical units.

**Upstream reference:** Wilson N M, Ortiz A K and Johnson A B. The Vascular Model Repository: A Public Resource of Medical Imaging Data and Blood Flow Simulation Results. *Journal of Medical Devices* 7(4), 040923 (2013). https://doi.org/10.1115/1.4025983.

Original VMR source notices are reproduced in `licenses/VMR/`. Dataset users should attribute the upstream repository and the associated research results as appropriate.

## Per-geometry archive index

[`data/provenance_index.json`](data/provenance_index.json) links all 70 dataset case identifiers to 63 VMR archive identifiers and Stanford Digital Repository records, and records the source-model SHA256 used for provenance matching. `source_archive_id` and `source_archive_name` contain the official VMR project name (for example, `0002_H_AO_SVD`); `source_url` resolves to its Stanford archive page. These names and records are indexed in the [official VMR catalog](https://github.com/SimVascular/vascularmodel/blob/main/dataset/dataset-svprojects.csv).

`source_group` (for example, `VMR_IMAGE_0017`) is an internal grouping identifier for the fixed cross-validation splits, not an official VMR archive number. Cases sharing a source group share the same source archive identifier. `source_model_sha256` identifies the original source model, which may differ byte-for-byte from its subsequently processed geometry.

Each linked archive record identifies the original model file by path, size and checksums. The links describe technical model provenance, not additional clinical information. Original VMR rights notices and the per-case notice bindings remain in `licenses/VMR/`; component-specific terms are described in [LICENSES.md](LICENSES.md).
