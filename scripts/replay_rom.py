"""Recompute RI–RCR responses from geometry and prescribed boundary conditions."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from public_data import ROOT,npz,physical_inputs


def replay(case_id):
    spec=importlib.util.spec_from_file_location('phytcr_dataset_rom',ROOT/'code/rom.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model=module.RigidTreeRCR(physical_inputs(case_id))
    saved=npz(ROOT/'data/cases'/case_id/'physics.npz')
    initial=np.r_[np.zeros(model.k),np.full(model.m,model.cfg['initial_pressure_mmHg'])]
    states=model.integrate(initial,model.cfg['window']['end_seconds'],.0005,saved['times_seconds'])[:,:,0]
    observed=model.observe(states,saved['times_seconds'])
    errors={k:float(np.max(abs(observed[k]-saved[k]))) for k in ('pressure_mmHg','flow_cm3_s')}
    errors['states']=float(np.max(abs(states-saved['states'])))
    assert max(errors.values())<1e-5
    return dict(case_id=case_id,frames=65,maximum_absolute_differences=errors,result='ok')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',default=['aorta_017','aorta_004','aorta_026'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    text=json.dumps([replay(cid) for cid in args.cases],indent=2,allow_nan=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(text+'\n',encoding='utf-8')
    print(text)
