# Input feature and observation specification

The schema records **common** and **RI–RCR-derived** components separately. F/D1 consume their concatenation: node static 12+2=14, edge static 13+5=18, global static 9, node temporal 8+2=10, edge temporal 9+3=12. Every temporal feature has 64 phases. Node/edge counts vary by case; indices are zero-based. Spatial coordinates and radii use cm.

## Common static features (zero-based column order)

Node, 12 columns:

1. log(radius)
2. log1p(root-path distance)
3. log1p(depth)
4. indegree
5. outdegree
6. inlet flag
7. outlet flag
8. log1p(number of terminal descendants)
9. log1p(descendant conductance × 1e4)
10. log(Rp) at terminal nodes, zero at other nodes
11. log(C) at terminal nodes, zero at other nodes
12. log(Rd) at terminal nodes, zero at other nodes

Edge, 13 columns:

1. log(length)
2. log(source-node radius)
3. log(target-node radius)
4. log(target/source radius)
5. cosine of upstream segment angle (zero where absent)
6. source outdegree
7. target outdegree
8. log1p(source root-path distance)
9. inlet-chain flag
10. inlet-contact-checked flag
11. observation-offset-active flag
12. inlet contact length (cm)
13. log1p(plane global crossing count)

Global, 9 columns:

1. cardiac period (s)
2. density (g/cm³)
3. dynamic viscosity (g/(cm·s))
4. prescribed initial pressure (mmHg) / 100
5. output-window start time (s) / 10
6. mean(Rd × C) / 4
7. log1p(node count)
8. log1p(outlet count)
9. log1p(total terminal conductance × 1e4)

## Common temporal features

Node, 8 columns: prescribed inlet flow / 100; sin(2πphase); cos(2πphase); sin(4πphase); cos(4πphase); phase index/64; absolute time / 10; exp(-absolute time/mean(Rd×C)).

All nodes share this input forcing. Edge temporal features use the same eight columns and an additional observation-offset / fitting-flow-standard-deviation channel. The ninth edge channel is assembled at load time using the fitting-flow scale. The entire edge temporal tensor is not duplicated in the public common-feature file.

## RI–RCR features, recomputed for each fitting subset

Let p be the standardized physical pressure, qt the standardized transport flow, qo the standardized observation-compatible flow, and drop the standardized edge pressure difference. Physical flow and observation-compatible flow are different: Qobs = Qtransport − input-derived boundary_offset.

- Node static: temporal mean(p), temporal std(p).
- Edge static: mean(qt), std(qt), mean(qo), std(qo), mean(drop).
- Node temporal: p, p − temporal mean(p).
- Edge temporal: qt, qo, drop.
- Baselines: standardized physical pressure and standardized observation-compatible flow, in time-first layout.

All normalization moments are recomputed from the actual fitting cases only. Local constraints/masks/section areas are target-side information, not features. The dataset reader constructs standardized targets, physical baselines and offset channels using these per-fit statistics.

## Observation definitions

`section_geometry.npz` retains pressure planes at graph nodes and flow planes at directed-edge midpoints with the directed sampling-normal convention, incidence and terminal maps. `observation_map.npz` retains the input-derived BCT offset and geometric checks. The supplied offset is computed from prescribed BCT and geometric sampling information. It aligns the transport-flow baseline with the section observation definition.

The graph's `case_feature_names` array describes nine graph-level boundary summaries. The actual neural tensor channel order is defined above and assembled by `scripts/public_data.py`.
