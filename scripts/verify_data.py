"""Dataset file integrity, array structure and grouped split consistency."""
import hashlib
import json
from collections import Counter
import numpy as np
from public_data import ROOT,labels,npz,read_json


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def run():
    manifest=read_json(ROOT/'SHA256SUMS.json')
    listed={r['path'] for r in manifest['files']}
    present={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
             if p.is_file() and p.name!='SHA256SUMS.json' and '__pycache__' not in p.parts}
    assert listed==present
    for row in manifest['files']:
        path=ROOT/row['path']
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
    split=read_json(ROOT/'splits/five_fold.json')
    ids=set(split['case_to_source_group'])
    assert len(ids)==70 and len(set(split['case_to_source_group'].values()))==63
    assert split['training_seeds']==[7,19,43] and len(split['folds'])==5
    outer=Counter()
    role_sizes=[]
    for fold in split['folds']:
        roles=[set(fold[k]) for k in ('training','inner_validation','outer_test')]
        assert set.union(*roles)==ids and sum(map(len,roles))==70
        source_roles=[{split['case_to_source_group'][cid] for cid in role} for role in roles]
        for i in range(3):
            for j in range(i+1,3):
                assert not roles[i]&roles[j] and not source_roles[i]&source_roles[j]
        outer.update(fold['outer_test'])
        role_sizes.append(list(map(len,roles)))
    assert set(outer)==ids and set(outer.values())=={1}
    dictionary=read_json(ROOT/'data/data_dictionary.json')['fields']
    assert len(dictionary)==55
    local_geometries=local_sections=0
    for cid in sorted(ids):
        folder=ROOT/'data/cases'/cid
        graph,boundary=npz(folder/'graph.npz'),npz(folder/'boundary.npz')
        y=labels(cid)
        section=npz(folder/'section_geometry.npz')
        n=len(graph['radius_cm'])
        s,t=graph['edge_index']
        e=len(s)
        assert e==n-1 and min(s.min(),t.min())>=0 and max(s.max(),t.max())<n
        indegree=np.bincount(t,minlength=n)
        outdegree=np.bincount(s,minlength=n)
        roots=np.flatnonzero(indegree==0)
        assert len(roots)==1 and np.all(indegree[np.arange(n)!=roots[0]]==1)
        assert np.array_equal(np.flatnonzero(graph['inlet_mask']),roots)
        assert np.array_equal(np.flatnonzero(graph['outlet_mask']),np.flatnonzero(outdegree==0))
        assert set(boundary['outlet_nodes'])==set(np.flatnonzero(outdegree==0))
        reached={int(roots[0])}
        for _ in range(n):
            before=len(reached)
            reached|={int(b) for a,b in zip(s,t) if int(a) in reached}
            if len(reached)==before:
                break
        assert len(reached)==n
        symbols={'N':n,'E':e,'M':len(boundary['outlet_nodes']),'2*E':2*e,
                 '2*M-1':2*len(boundary['outlet_nodes'])-1,
                 'N_internal':len(section['interior_nodes']),
                 'T_inlet':len(boundary['times_seconds']),'K_local':len(y['local_edges'])}
        fields=set()
        for path in sorted(folder.glob('*.npz')):
            arrays=npz(path)
            for name,array in arrays.items():
                key=path.name+'/'+name
                fields.add(key)
                definition=dictionary[key]
                dims=[v.strip() for v in definition['shape'].strip('()').split(',') if v.strip()]
                shape=tuple(int(v) if v.isdigit() else symbols[v] for v in dims)
                assert array.shape==shape,(cid,key,array.shape,shape)
                assert str(array.dtype) in definition['dtypes'],(cid,key,array.dtype)
        assert fields==set(dictionary)
        assert np.isfinite(y['P_raw'][:,y['P_mask']]).all() and y['P_mask'].any()
        assert np.isfinite(y['Q_raw'][:,y['Q_mask']]).all() and y['Q_mask'].any()
        phase=npz(folder/'phases.npz')
        physical=npz(folder/'physics.npz')
        np.testing.assert_allclose(phase['times_seconds'],physical['times_seconds'][:64],atol=1e-9,rtol=0)
        np.testing.assert_allclose(np.diff(phase['times_seconds']),float(boundary['period_seconds'])/64,atol=1e-9,rtol=0)
        assert np.array_equal(section['physical_edge_index'],graph['edge_index'])
        local_geometries+=int(len(y['local_edges'])>0)
        local_sections+=len(y['local_edges'])
    index=read_json(ROOT/'results/prediction_index.json')
    assert len(index['records'])==420
    seen=set()
    for r in index['records']:
        key=(r['model'],r['fold'],r['seed'],r['case_id'])
        assert key not in seen and r['case_id'] in split['folds'][r['fold']]['outer_test']
        seen.add(key)
        assert (ROOT/r['prediction_path']).is_file() and (ROOT/r['score_path']).is_file()
    assert set(r['model'] for r in index['records'])=={'PhyTCR-Net','D1'}
    return dict(check='Data integrity check',result='ok',files_checked=len(listed),
        geometries=70,provenance_groups=63,folds=5,seeds=[7,19,43],
        fold_role_counts=role_sizes,outer_test_appearances_per_geometry=1,
        model_case_predictions=420,array_definitions=55,
        geometries_with_local_flow_data=local_geometries,local_flow_sections=local_sections)


if __name__=='__main__':
    print(json.dumps(run(),indent=2,allow_nan=False))
