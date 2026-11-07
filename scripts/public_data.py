"""Data access, fitting-subset normalization and neural input construction."""
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MMHG_CGS = 1333.22


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


@lru_cache(maxsize=1)
def case_ids():
    return frozenset(r['case_id'] for r in read_json(ROOT/'data/cases.json')['cases'])


def case_dir(case_id):
    if case_id not in case_ids():
        raise KeyError('Unknown geometry identifier: '+str(case_id))
    return ROOT/'data/cases'/case_id


def labels(case_id):
    """Pressure in mmHg, flow in mL/s, masks and local-flow target arrays."""
    folder=case_dir(case_id)
    y=npz(folder/'labels.npz')
    return dict(P_raw=y['pressure_node']/MMHG_CGS,Q_raw=y['flow_edge_cm3_s'],
        P_mask=y['node_label_mask'].astype(bool),Q_mask=y['edge_label_mask'].astype(bool),
        **npz(folder/'local_constraints.npz'))


def fit_statistics(training):
    """Case-equal first/second moments over the supplied fitting cases only."""
    if not training or len(set(training))!=len(training):
        raise ValueError('Provide distinct fitting-case identifiers')
    moments={'p':[],'flow':[]}
    for cid in training:
        y=labels(cid)
        for name,prefix in (('p','P'),('flow','Q')):
            values=y[prefix+'_raw'][:,y[prefix+'_mask']]
            moments[name].append((float(values.mean()),float((values**2).mean())))
    stats={}
    for name,rows in moments.items():
        mean=float(np.mean([r[0] for r in rows]))
        second=float(np.mean([r[1] for r in rows]))
        stats[name+'_mean']=mean
        stats[name+'_std']=max(float(np.sqrt(max(second-mean*mean,0))),1.)
    return stats


def predictive_arrays(case_id,stats):
    """Construct inputs from geometry, prescribed forcing and RI–RCR outputs."""
    folder=case_dir(case_id)
    common=npz(folder/'common_features.npz')
    graph=npz(folder/'graph.npz')
    physical=npz(folder/'physics.npz')
    geometry=npz(folder/'observation_map.npz')
    physical_P=physical['pressure_mmHg'][:64]
    physical_Q=physical['flow_cm3_s'][:64]
    offset=geometry['boundary_offset_cm3_s']
    s,t=graph['edge_index']
    p=(physical_P-stats['p_mean'])/stats['p_std']
    qt=(physical_Q-stats['flow_mean'])/stats['flow_std']
    qo=(physical_Q-offset-stats['flow_mean'])/stats['flow_std']
    drop=(physical_P[:,s]-physical_P[:,t])/stats['p_std']
    forcing=np.broadcast_to(common['node_common_time'][0],(len(s),64,8)).copy()
    arrays=dict(common,edge_index=graph['edge_index'],
        edge_common_time=np.concatenate((forcing,(offset/stats['flow_std']).T[:,:,None]),axis=2),
        baseline_P=p,baseline_Q=qo,
        node_rom=np.column_stack((p.mean(0),p.std(0))),
        edge_rom=np.column_stack((qt.mean(0),qt.std(0),qo.mean(0),qo.std(0),drop.mean(0))),
        node_rom_time=np.stack((p.T,(p-p.mean(0)).T),axis=2),
        edge_rom_time=np.stack((qt.T,qo.T,drop.T),axis=2))
    return {key:np.asarray(value,dtype=np.int64 if key=='edge_index' else np.float32)
            for key,value in arrays.items()}


def target_arrays(case_id,stats):
    y=labels(case_id)
    y['P']=((y['P_raw']-stats['p_mean'])/stats['p_std']).astype(np.float32)
    y['Q']=((y['Q_raw']-stats['flow_mean'])/stats['flow_std']).astype(np.float32)
    y['local_target_norm']=((y['local_target_raw']-stats['flow_mean'])/stats['flow_std']).astype(np.float32)
    return y


def physical_inputs(case_id):
    folder=case_dir(case_id)
    graph,boundary=npz(folder/'graph.npz'),npz(folder/'boundary.npz')
    maps=npz(folder/'section_geometry.npz')
    cfg=read_json(folder/'configuration.json')
    period=float(boundary['period_seconds'])
    phase=(np.arange(64)*period/64+float(boundary['phase_offset_seconds']))%period
    qin=np.interp(phase,boundary['times_seconds'],boundary['inlet_flow_cm3_s'])
    return dict(graph=graph,boundary=boundary,maps=maps,source_configuration=cfg,
                phase_seconds=phase,prescribed_qin_cm3_s=qin)


@lru_cache(maxsize=1)
def model_module():
    spec=importlib.util.spec_from_file_location('phytcr_dataset_models',ROOT/'code/models.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(model='PhyTCR-Net'):
    configs=read_json(ROOT/'methods/model_configs.json')
    return model_module().FullCycleGraphModel(**configs[model])
