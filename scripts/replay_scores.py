"""Evaluate saved outer-test predictions and provenance-group summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
from public_data import ROOT,fit_statistics,labels,npz,read_json


def channel(prediction,truth,mask,scale):
    error=prediction[:,mask]-truth[:,mask]
    mean=error.mean(0)
    wave=error-mean
    true_wave=truth[:,mask]-truth[:,mask].mean(0)
    mse=float(np.mean(error**2))
    den=max(float(np.mean(true_wave**2)),1e-12)
    return dict(rmse=mse**.5,standardized_rmse=mse**.5/scale,mae=float(abs(error).mean()),
        mean_rmse=float(np.sqrt(np.mean(mean**2))),wave_rmse=float(np.sqrt(np.mean(wave**2))),
        wave_ratio=float(np.mean(wave**2))/den,wave_truth_MSE_denominator=den,
        signed_mean_error=float(mean.mean()),valid_positions=int(mask.sum()))


def score_case(case_id,p,q,stats):
    y=labels(case_id)
    if p.shape!=y['P_raw'].shape or q.shape!=y['Q_raw'].shape:
        raise ValueError('Prediction dimensions differ from section labels: '+case_id)
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        raise FloatingPointError('Prediction contains nonfinite values: '+case_id)
    ps=channel(p,y['P_raw'],y['P_mask'],stats['p_std'])
    qs=channel(q,y['Q_raw'],y['Q_mask'],stats['flow_std'])
    local=dict(sections=len(y['local_edges']))
    if len(y['local_edges']):
        rms=np.sqrt(np.mean((q[:,y['local_edges']]-y['local_target_raw'])**2,0))
        local.update(rms_mL_s=rms.tolist())
    folder=ROOT/'data/cases'/case_id
    graph=npz(folder/'graph.npz')
    common=npz(folder/'common_features.npz')
    s,t=graph['edge_index']
    valid=y['P_mask'][s]&y['P_mask'][t]
    delta=((p[:,s]-p[:,t])-(y['P_raw'][:,s]-y['P_raw'][:,t]))[:,valid]
    peaks={}
    for name,pred,truth,mask in (('P',p,y['P_raw'],y['P_mask']),('Q',q,y['Q_raw'],y['Q_mask'])):
        d=abs(np.argmax(pred[:,mask],0)-np.argmax(truth[:,mask],0))
        peaks[name]=float(np.minimum(d,64-d).mean()/64*float(common['global_common'][0]))
    return dict(case_id=case_id,P=ps,Q=qs,S=.5*(ps['standardized_rmse']**2+qs['standardized_rmse']**2),
        local_flow_consistency=local,geometry_adjacent_P_pairs=int(valid.sum()),
        adjacent_deltaP_rmse_mmHg=None if not valid.any() else float(np.sqrt(np.mean(delta**2))),
        peak_time_circular_MAE_seconds=peaks)


def difference(a,b):
    if isinstance(a,dict):
        assert set(a)<=set(b)
        return max((difference(v,b[k]) for k,v in a.items()),default=0.)
    if isinstance(a,list):
        assert len(a)==len(b)
        return max((difference(v,w) for v,w in zip(a,b)),default=0.)
    if isinstance(a,str) or a is None or isinstance(a,bool):
        assert a==b,(a,b)
        return 0.
    return abs(float(a)-float(b))


def bootstrap(values,fold_ids,stratified=True):
    rng=np.random.default_rng(20261029)
    n=len(values)
    if stratified:
        drawn=[]
        for fold in sorted(set(fold_ids)):
            members=np.flatnonzero(fold_ids==fold)
            drawn.append(members[rng.integers(0,len(members),(20000,len(members)))])
        indices=np.concatenate(drawn,axis=1)
    else:
        indices=rng.integers(0,n,(20000,n))
    return np.quantile(values[indices].mean(1),[.025,.975],method='linear').tolist()


def run():
    split=read_json(ROOT/'splits/five_fold.json')
    groups=split['case_to_source_group']
    index=read_json(ROOT/'results/prediction_index.json')
    fits={r['selection_id']:r for r in read_json(ROOT/'methods/model_selection.json')['fits']}
    stats={r['outer_fold']:fit_statistics(r['training']) for r in split['folds']}
    rows=[]
    diffs=[]
    for record in index['records']:
        fit=fits[record['selection_id']]
        assert difference(stats[record['fold']],fit['statistics'])<1e-12
        pred=npz(ROOT/record['prediction_path'])
        row=score_case(record['case_id'],pred['pressure_mmHg'],pred['flow_cm3_s'],stats[record['fold']])
        diffs.append(difference(row,read_json(ROOT/record['score_path'])))
        rows.append(dict(model=record['model'],fold=record['fold'],seed=record['seed'],**row))
    assert len(rows)==420 and max(diffs)<1e-10
    metrics={}
    group_values={}
    for model in ('PhyTCR-Net','D1'):
        subset=[r for r in rows if r['model']==model]
        ids=sorted({r['case_id'] for r in subset})
        assert len(ids)==70 and all(sum(r['case_id']==cid for r in subset)==3 for cid in ids)
        per_case={cid:{ch:float(np.mean([r[ch]['rmse'] for r in subset if r['case_id']==cid]))
                       for ch in ('P','Q')} for cid in ids}
        group_values[model]={g:{ch:float(np.mean([v[ch] for cid,v in per_case.items() if groups[cid]==g]))
                               for ch in ('P','Q')} for g in sorted(set(groups.values()))}
        metrics[model]={ch:dict(rmse=float(np.mean([v[ch] for v in group_values[model].values()])),
                               unit='mmHg' if ch=='P' else 'mL/s') for ch in ('P','Q')}
    names=sorted(group_values['PhyTCR-Net'])
    fold_map={groups[cid]:fold['outer_fold'] for fold in split['folds'] for cid in fold['outer_test']}
    fold_ids=np.array([fold_map[g] for g in names])
    contrasts={}
    for ch in ('P','Q'):
        values=np.array([group_values['D1'][g][ch]-group_values['PhyTCR-Net'][g][ch] for g in names])
        contrasts[ch]=dict(mean_D1_minus_PhyTCR_Net=float(values.mean()),CI95=bootstrap(values,fold_ids),
            unstratified_CI95=bootstrap(values,fold_ids,False),
            PhyTCR_Net_lower_error_groups=int((values>0).sum()),ties=int((values==0).sum()),
            unit='mmHg' if ch=='P' else 'mL/s')
    saved=read_json(ROOT/'results/primary_metrics.json')
    assert difference(metrics,saved['models'])<1e-10
    assert difference(contrasts,saved['paired_comparison'])<1e-10
    return dict(evaluation='Main model comparison',result='ok',model_case_results=len(rows),
        source_groups=len(names),seeds_per_geometry=3,score_file_max_absolute_difference=max(diffs),
        models=metrics,paired_comparison=contrasts)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    text=json.dumps(run(),indent=2,allow_nan=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(text+'\n',encoding='utf-8')
    print(text)
