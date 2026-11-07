# Data dictionary

The data contain 70 aortic geometries, 63 provenance-based groups and five grouped outer folds. N is the node count, E=N−1 the directed-edge count, M the outlet count, T_inlet the inlet-waveform sample count, N_internal the interior-node count, and K_local the number of additional local-flow sections. Indices are zero-based. Empty local arrays use K_local=0.

CFD labels use 64 time samples. RI–RCR responses use 65 samples; frame 64 is the additional endpoint one period after frame 0. `phases.npz` contains the CFD sample times, and `physics.npz/times_seconds` gives the physical sampling sequence. Absolute window timing is geometry-specific.

Pressure targets are stored in dyn/cm² and converted to mmHg by division by 1333.22. Flow in cm³/s is numerically mL/s. True values in the node/edge label masks select sections for loss and metric calculations; sections with False masks are excluded from these operations. Arrays retain entries at all graph positions.

The tables below cover all 55 NPZ fields. Machine-readable types and per-geometry shape variations are in `data/data_dictionary.json`. Neural features concatenate common and RI–RCR-derived channels: node static 12+2=14, edge static 13+5=18, graph-wide static 9, node temporal 8+2=10, edge temporal 9+3=12. Ordered channel definitions are in `data/FEATURE_SPEC.md`.


## boundary.npz

| Field | Shape | Units | Role | Definition |
|---|---|---|---|---|
| `times_seconds` | `(T_inlet,)` | s | input | Sample times of the prescribed inlet-flow waveform within its period. |
| `inlet_flow_cm3_s` | `(T_inlet,)` | cm^3/s | input | Prescribed inlet volumetric flow rate; positive into the vascular tree. |
| `outlet_nodes` | `(M,)` | zero-based node index | input | Terminal graph nodes in the same order as the RCR parameter rows. |
| `rcr` | `(M,3)` | Rp,Rd: dyn*s/cm^5; C: cm^5/dyn | input | Columns are proximal resistance Rp, compliance C, and distal resistance Rd. |
| `period_seconds` | `()` | s | input | Cardiac period used to sample inlet forcing and define the 64-phase sequence. |
| `phase_offset_seconds` | `()` | s | input | Time offset applied when sampling the periodic inlet waveform. |
| `viscosity_cgs` | `()` | g/(cm*s) | input | Dynamic viscosity in the boundary representation. |

## common_features.npz

| Field | Shape | Units | Role | Definition |
|---|---|---|---|---|
| `node_common` | `(N,12)` | mixed scaled geometric/topological channels | input | Node static feature channels 0–11 in data/FEATURE_SPEC.md. |
| `edge_common` | `(E,13)` | mixed scaled geometric/topological channels | input | Edge static feature channels 0–12 in data/FEATURE_SPEC.md. |
| `global_common` | `(9,)` | mixed scaled physical/geometric channels | input | Graph-wide static feature channels 0–8 in data/FEATURE_SPEC.md. |
| `node_common_time` | `(N,64,8)` | scaled flow, time and dimensionless phase channels | input | Eight prescribed-forcing channels, repeated at each node; data/FEATURE_SPEC.md gives their order. |

## graph.npz

| Field | Shape | Units | Role | Definition |
|---|---|---|---|---|
| `coordinates_cm` | `(N,3)` | cm | input | Three-dimensional coordinates of vascular-tree nodes. |
| `radius_cm` | `(N,)` | cm | input | Vessel radius at each graph node. |
| `edge_index` | `(2,E)` | zero-based node index | input | Row 0 contains source nodes and row 1 target nodes; edges point from inlet toward outlets. |
| `edge_length_cm` | `(E,)` | cm | input | Physical segment length for each directed edge. |
| `inlet_mask` | `(N,)` | Boolean | input | True at the inlet/root graph node. |
| `outlet_mask` | `(N,)` | Boolean | input | True at terminal outlet graph nodes. |
| `case_feature_names` | `(9,)` | text | metadata | Names describing graph-level boundary summaries: outlet count, resistance and compliance statistics. The neural tensor channel order is specified separately in data/FEATURE_SPEC.md. |

## labels.npz

| Field | Shape | Units | Role | Definition |
|---|---|---|---|---|
| `steps` | `(64,)` | solver step index | supervision | CFD output step for each of the 64 section-label frames. |
| `pressure_node` | `(64,N)` | dyn/cm^2 | supervision | CFD cross-sectional mean pressure at node sampling planes. Divide by 1333.22 for mmHg. |
| `flow_edge_cm3_s` | `(64,E)` | cm^3/s (=mL/s) | supervision | CFD cross-sectional volumetric flow along each directed edge normal. |
| `node_area_cm2` | `(N,)` | cm^2 | section metadata | Area of the pressure-sampling section at each node. |
| `edge_area_cm2` | `(E,)` | cm^2 | section metadata | Area of the flow-sampling section for each edge. |
| `node_label_mask` | `(N,)` | Boolean | supervision/evaluation | True identifies node-pressure sections included in supervised loss and scoring. |
| `edge_label_mask` | `(E,)` | Boolean | supervision/evaluation | True identifies edge-flow sections included in supervised loss and scoring. |

## local_constraints.npz

| Field | Shape | Units | Role | Definition |
|---|---|---|---|---|
| `local_edges` | `(K_local,)` | zero-based edge index | local supervision/evaluation | Edges with an input-derived local flow-consistency target. |
| `local_target_raw` | `(64,K_local)` | cm^3/s | local supervision/evaluation | Local target formed from prescribed inlet flow and the geometric observation offset. |
| `local_rms_limits` | `(K_local,)` | cm^3/s | optional historical metadata | Stored comparison levels; not predictive inputs, local-loss targets, loss weights or public metric acceptance criteria. |

## observation_map.npz

| Field | Shape | Units | Role | Definition |
|---|---|---|---|---|
| `boundary_offset_cm3_s` | `(64,E)` | cm^3/s | input/observation mapping | Input-derived correction to transport flow; observation-compatible flow is transport flow minus this array. |
| `inlet_chain` | `(E,)` | Boolean | input/observation mapping | Edges on the inlet-connected unbranched chain used by the observation mapping. |
| `inlet_contact_length_cm` | `(E,)` | cm | input/observation mapping | Geometric length of inlet-plane contact associated with each edge observation. |
| `inlet_contact_checked` | `(E,)` | Boolean | input/observation mapping | Identifies edges where inlet-plane contact was evaluated. |
| `inlet_offset_active` | `(E,)` | Boolean | input/observation mapping | Identifies edges whose observation mapping applies an inlet-related offset. |
| `plane_global_crossing_count` | `(E,)` | count | input/observation mapping | Number of surface crossings used to describe each sampling plane. |
| `BCT_replay_max_cm3_s` | `()` | cm^3/s | boundary mapping metadata | Maximum inlet-flux reconstruction difference in the prescribed BCT mapping. |

## phases.npz

| Field | Shape | Units | Role | Definition |
|---|---|---|---|---|
| `times_seconds` | `(64,)` | s | sampling | Absolute CFD output times, measured from the simulation time origin. |
| `relative_times_seconds` | `(64,)` | s | sampling | Output times relative to the first frame in the selected window. |
| `steps` | `(64,)` | solver step index | sampling | Same ordered CFD output steps as labels.npz/steps. |
| `phase_fraction` | `(64,)` | dimensionless | sampling | Sample ordinal divided by 64; absolute timing is given by times_seconds. |

## physics.npz

| Field | Shape | Units | Role | Definition |
|---|---|---|---|---|
| `times_seconds` | `(65,)` | s | physical baseline sampling | RI–RCR sampling times including the additional frame one period after the first. |
| `states` | `(65,2*M-1)` | first M-1: cm^3/s; last M: mmHg | physical baseline | Free terminal-flow coordinates followed by outlet capacitor-pressure states. |
| `pressure_mmHg` | `(65,N)` | mmHg | physical baseline/input | RI–RCR node pressure outputs on the 65-frame physical sampling sequence. |
| `flow_cm3_s` | `(65,E)` | cm^3/s | physical baseline/input | RI–RCR transport flow before subtraction of the observation offset. |

## section_geometry.npz

| Field | Shape | Units | Role | Definition |
|---|---|---|---|---|
| `physical_edge_index` | `(2,E)` | zero-based node index | geometry | Directed physical edges, ordered as graph.npz/edge_index. |
| `incidence` | `(N,E)` | dimensionless integer (-1,0,+1) | geometry/physical model | Node-edge incidence with -1 at an edge source and +1 at its target. |
| `interior_nodes` | `(N_internal,)` | zero-based node index | geometry/physical model | Nodes having both incoming and outgoing physical edges. |
| `terminal_sum_map` | `(E,M)` | dimensionless integer (0,1) | geometry/physical model | Maps terminal outlet flows to transport flow on each physical edge. |
| `message_edge_index` | `(2,2*E)` | zero-based node index | graph representation | Physical edges followed by their reversed directions for bidirectional graph operations. |
| `message_physical_edge` | `(2*E,)` | zero-based edge index | graph representation | Physical-edge index associated with each bidirectional edge entry. |
| `message_direction` | `(2*E,)` | dimensionless integer (+1,-1) | graph representation | +1 for physical direction and -1 for the reversed edge entry. |
| `pressure_center_cm` | `(N,3)` | cm | sampling geometry | Pressure-sampling plane centers at the graph nodes. |
| `pressure_normal_unit` | `(N,3)` | dimensionless | sampling geometry | Pressure-plane unit normals where pressure_plane_defined is True; zero otherwise. |
| `pressure_plane_defined` | `(N,)` | Boolean | sampling geometry | Identifies nodes with a constructed pressure sampling-plane normal. |
| `flow_center_cm` | `(E,3)` | cm | sampling geometry | Flow-sampling plane centers at edge midpoints. |
| `flow_normal_unit` | `(E,3)` | dimensionless | sampling geometry | Flow-plane unit normals directed from source node toward target node. |

## Configuration and indices

`data/cases.json` lists `case_id`, `source_group`, node/edge/sample counts, label paths, file checksums, valid section counts and local-flow section counts. `data/provenance_index.json` associates each geometry with its VMR group and source model checksum. `splits/five_fold.json` lists training, inner-validation and outer-test identifiers and training seeds; nested fitting subsets are provided within each fold.

Each `configuration.json` contains density (g/cm³), viscosity (g/(cm·s)), CFD time step (s), period (s), prescribed initial pressure (mmHg), initial velocity (cm/s), distal-pressure curves (time in s, pressure in dyn/cm²), output-window start/end (s), `wall_model`, `inflow_source` and BCT Fourier-mode count. The RI–RCR integration starts at t=0 and proceeds continuously to the output window.

`ports` follows `boundary.npz/outlet_nodes`. Each port supplies its record/node/surface indices, cap area (cm²), centroid (cm), outward unit normal, graph-node-to-cap displacement (cm), and Rp/C/Rd in the CGS units given above. `inlet_geometry` supplies inlet area, centroid, outward normal and directional coherence quantities. These are boundary/geometry inputs.

`common_features.npz` supplies input channels. `labels.npz` supplies supervised observations and section areas/masks. `local_constraints.npz` supplies additional local targets in 15 geometries (20 sections). `observation_map.npz` maps physical transport flow into the section-observation convention. The model/reader uses the first 64 physical frames, while all 65 physical frames are retained for numerical replay.

`results/prediction_index.json` gives public model/fold/seed/geometry paths. Prediction arrays are `pressure_mmHg` (64,N) and `flow_cm3_s` (64,E). Per-geometry scores and primary group-level RMSE/confidence intervals are described in `results/README.md` and `methods/evaluation_protocol.json`.


### Standard configuration metadata

| Field | Value | Units | Definition |
|---|---|---|---|
| `wall_model` | `rigid` | Not applicable (categorical) | Rigid-wall CFD assumption. |
| `inflow_source` | `surface-integrated prescribed inlet velocity` | Not applicable (description); derived flow is in cm³/s | The inlet flow waveform is obtained by surface integration of the prescribed inlet velocity boundary condition. |

`inlet_geometry`, ordered `ports`, `Pd_curves_dyn_cm2` and every other configuration value retain their original definitions and numerical values. `local_rms_limits` is retained unchanged only as optional historical comparison metadata; the local loss uses `local_edges` and `local_target_raw`, not these comparison levels.

### Source and score indices

`data/provenance_index.json` includes `source_archive_id` and `source_archive_name` (the official VMR project name), `source_url` (the Stanford archive page), `source_model_sha256` (original source-model checksum), and `source_group` (internal cross-validation grouping identifier). See [SOURCES.md](SOURCES.md).

Per-geometry `local_flow_consistency` scores contain `sections` (number of local sections) and, when sections are present, `rms_mL_s` (one RMS difference from the local target per section). Other score fields, formulas and units remain as described in [results/README.md](results/README.md) and [methods/evaluation_protocol.json](methods/evaluation_protocol.json).
