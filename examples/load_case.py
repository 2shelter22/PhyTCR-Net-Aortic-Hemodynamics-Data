"""Read one geometry's pressure/flow waveforms with their original masks."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from public_data import case_dir,npz


def load(case_id):
    folder=case_dir(case_id)
    graph=npz(folder/'graph.npz')
    times=npz(folder/'phases.npz')['times_seconds']
    with np.load(folder/'labels.npz',allow_pickle=False) as data:
        pressure_mmHg=data['pressure_node']/1333.22
        flow_mL_s=data['flow_edge_cm3_s'].copy()
        pressure_mask=data['node_label_mask'].astype(bool)
        flow_mask=data['edge_label_mask'].astype(bool)
    node=int(np.flatnonzero(pressure_mask)[0])
    edge=int(np.flatnonzero(flow_mask)[0])
    return dict(case_id=case_id,nodes=len(graph['radius_cm']),edges=graph['edge_index'].shape[1],
        pressure_shape=list(pressure_mmHg.shape),flow_shape=list(flow_mL_s.shape),
        pressure_unit='mmHg',flow_unit='mL/s',time_unit='s',
        valid_pressure_sections=int(pressure_mask.sum()),valid_flow_sections=int(flow_mask.sum()),
        pressure_node_index=node,flow_edge_index=edge,
        flow_edge_source_target=graph['edge_index'][:,edge].tolist(),
        times_seconds=times.tolist(),pressure_waveform_mmHg=pressure_mmHg[:,node].tolist(),
        flow_waveform_mL_s=flow_mL_s[:,edge].tolist())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',default='aorta_017')
    args=parser.parse_args()
    print(json.dumps(load(args.case),indent=2,allow_nan=False))
