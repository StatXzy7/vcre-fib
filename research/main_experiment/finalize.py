"""Export traceable paper tables and test-first report from completed evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sfibai_b.bootstrap import paired_patient_cluster_bootstrap
from synap_search.io import sha256, write_json


def bootstrap_with_draws(candidate, reference, destination):
    result=paired_patient_cluster_bootstrap(candidate,reference,candidate_name='SYNAP',reference_name='SFIBAI',n_resamples=2000,seed=2026)
    patients=candidate.groupby('patient_uid',sort=True).agg(center_id=('center_id','first')).reset_index()
    rng=np.random.default_rng(2026); arrays={}
    for i, center in enumerate(sorted(patients.center_id.astype(str).unique())):
        indices=np.flatnonzero(patients.center_id.astype(str).to_numpy()==center)
        arrays[f'center_{i}']=indices[rng.integers(0,len(indices),size=(2000,len(indices)))]
    np.savez_compressed(destination/'paired_bootstrap_draws.npz',**arrays)
    patients.to_csv(destination/'paired_bootstrap_patient_index.csv',index=False)
    result.update(draws_sha256=sha256(destination/'paired_bootstrap_draws.npz'),
                  patient_index_sha256=sha256(destination/'paired_bootstrap_patient_index.csv'),
                  interpretation='descriptive patient sampling conditional on fitted models; not training randomness')
    return result


def export_split(root, methods, split):
    destination=root/'paper_ready'/split; destination.mkdir(parents=True,exist_ok=True)
    metrics={m:json.loads((root/m/'evaluation'/split/'metrics.json').read_text(encoding='utf-8')) for m in methods}
    frames={m:pd.read_csv(root/m/'evaluation'/split/'predictions.csv.gz',float_precision='round_trip',dtype={'image_uid':str,'patient_uid':str,'center_id':str}) for m in methods}
    bootstrap=None  # Optional bootstrap is outside the manuscript reproduction recipe.
    if bootstrap is not None: write_json(destination/'paired_bootstrap.json',bootstrap)
    ranking={m:i+1 for i,m in enumerate(sorted(methods,key=lambda m:metrics[m]['r_final']))}
    tables={name:[] for name in ('primary','per_seed','image_metrics','patient_metrics','auxiliary_metrics','checkpoints','training')}
    bindings={}
    for method in methods:
        m=metrics[method]; run=root/m; freeze=json.loads((run/'CHECKPOINT_FREEZE.json').read_text(encoding='utf-8'))
        base={'method_id':method,'method_name':'SynAP-Fib' if method=='SYNAP' else 'SFibAI (Data V4)','seed':2026,'status':'COMPLETE'}
        delta=m['r_final']-metrics['SFIBAI']['r_final'] if 'SFIBAI' in metrics else None
        relative=100*delta/metrics['SFIBAI']['r_final'] if delta is not None and metrics['SFIBAI']['r_final']!=0 else None
        main={**base,'r_final':m['r_final'],'image_cor':m['image']['cor'],'patient_max_cor':m['patient_max']['cor'],
              'patient_median_cor':m['patient_median']['cor'],'center_cor':m['center_balanced_patient_max']['cor'],
              'delta_abs_vs_sfibai':delta,'delta_rel_pct_vs_sfibai':relative}
        tables['primary'].append({**main,'rank':ranking[method]})
        ci=bootstrap['ci_95'] if method=='SYNAP' and bootstrap else ([0.,0.] if method=='SFIBAI' else [None,None])
        path=run/'evaluation'/split/'metrics.json'
        tables['per_seed'].append({**main,'paired_ci_low_vs_sfibai':ci[0],'paired_ci_high_vs_sfibai':ci[1],
            'n_images':m['image']['n'],'n_patients':m['patient_max']['n'],'n_centers':m['center_balanced_patient_max']['center_count'],
            'checkpoint_sha256':freeze['checkpoint_sha256'],'metrics_path':str(path),'metrics_sha256':sha256(path)})
        fields={'mae':'mae','acc_03':'accuracy_within_0_3','acc_05':'accuracy_within_0_5','accuracy_4grade':'grade_accuracy',
                'tmae':'tmae','severe_error_rate':'severe_error_rate','clinical_risk':'cor'}
        tables['image_metrics'].append({**base,'n_images':m['image']['n'],**{out:m['image'][key] for out,key in fields.items()}})
        patient={**base,'n_patients':m['patient_max']['n'],'n_centers':m['center_balanced_patient_max']['center_count'],
                 'center_clinical_risk':m['center_balanced_patient_max']['cor']}
        for agg in ('max','median'): patient.update({f'{agg}_{out}':m[f'patient_{agg}'][key] for out,key in fields.items()})
        tables['patient_metrics'].append(patient)
        tables['auxiliary_metrics'].append({**base,'status':'COMPLETE' if method=='SYNAP' else 'NOT_APPLICABLE',
             'n_view_images':m['position'].get('n'),'view_accuracy':m['position'].get('accuracy'),'view_macro_f1':m['position'].get('macro_f1'),
             'n_valid_weak_boxes':m['lesion'].get('valid_box_count'),'weak_box_dice_at_05':m['lesion'].get('dice_at_0_5'),
             'weak_box_iou_at_05':m['lesion'].get('iou_at_0_5')})
        identity=json.loads((run/'IDENTITY.json').read_text(encoding='utf-8'))
        tables['checkpoints'].append({**base,'actual_train_split':'train','actual_selection_split':'val','selected_epoch':freeze['epoch'],
             'validation_r_final':freeze['selection_key'][0],'validation_image_cor':freeze['selection_key'][1],
             'checkpoint_path':freeze['checkpoint'],'checkpoint_sha256':freeze['checkpoint_sha256'],'config_sha256':freeze['identity_sha256'],
             'split_manifest_sha256':identity['data_hashes']['images.csv'],'freeze_manifest_path':str(run/'CHECKPOINT_FREEZE.json'),
             'training_complete':True,'test_complete':True})
        for line in (run/'history.jsonl').read_text(encoding='utf-8').splitlines():
            item=json.loads(line); val=item['val']; epoch=item['epoch']
            cp=run/f'epoch_{epoch:03d}.pt'
            tables['training'].append({'method_id':method,'seed':2026,'epoch':epoch,'status':'COMPLETE',
                 'native_val_r_final':val['r_final'],'native_val_image_cor':val['image']['cor'],
                 'native_val_view_accuracy':val['position'].get('accuracy'),'native_val_weak_box_iou':val['lesion'].get('iou_at_0_5'),
                 'selection_eligible':epoch>=21,'checkpoint_sha256':sha256(cp) if cp.exists() else None})
        provenance=run/'evaluation'/split/'provenance.json'
        bindings[method]={'identity':str(run/'IDENTITY.json'),'freeze':freeze,'metrics':str(path),'metrics_sha256':sha256(path),
                          'predictions_provenance':json.loads(provenance.read_text(encoding='utf-8'))}
    for name,rows in tables.items(): pd.DataFrame(rows).to_csv(destination/f'{name}.csv',index=False)
    resources=[]
    for method in methods: resources.extend(json.loads((root/method/'resources.json').read_text(encoding='utf-8')))
    pd.DataFrame(resources).to_csv(destination/'resources.csv',index=False)
    write_json(destination/'source_bindings.json',bindings)
    return tables, bootstrap


def main():
    p=argparse.ArgumentParser(); p.add_argument('--round-root',type=Path,required=True);p.add_argument('--methods',nargs='+',default=['SFIBAI','SYNAP'])
    args=p.parse_args(); root=args.round_root
    for method in args.methods:
        run=root/method
        for name in ('TRAINING_COMPLETE.json','CHECKPOINT_FREEZE.json','EVALUATION_COMPLETE.json','resources.json'):
            if not (run/name).is_file(): raise RuntimeError(f'Missing completion artifact: {run/name}')
        freeze=json.loads((run/'CHECKPOINT_FREEZE.json').read_text(encoding='utf-8'))
        completed=json.loads((run/'TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
        history=[json.loads(line) for line in (run/'history.jsonl').read_text(encoding='utf-8').splitlines()]
        horizon = int(completed['epochs'])
        if not 21 <= horizon <= 120 or [h['epoch'] for h in history]!=list(range(1,horizon+1)) or not 21<=freeze['epoch']<=horizon:
            raise ValueError('Incomplete training horizon or invalid selected epoch')
        for epoch in range(10,horizon+1,10):
            if not (run/f'epoch_{epoch:03d}.pt').is_file(): raise ValueError('Missing periodic checkpoint')
        if sha256(run/'best.pt')!=freeze['checkpoint_sha256']: raise ValueError('Frozen checkpoint changed')
        for split in ('val','test'):
            dest=run/'evaluation'/split; provenance=json.loads((dest/'provenance.json').read_text(encoding='utf-8'))
            for filename,key in [('metrics.json','metrics_sha256'),('predictions.csv.gz','predictions_sha256')]:
                if sha256(dest/filename)!=provenance[key]: raise ValueError('Evaluation artifact changed')
            for agg,digest in provenance['patient_predictions_sha256'].items():
                if sha256(dest/f'patient_{agg}_predictions.csv.gz')!=digest: raise ValueError('Patient predictions changed')
    test,bootstrap=export_split(root,args.methods,'test'); val,_=export_split(root,args.methods,'val')
    lines=['# Test set 主结果','',f'Comparison round: {root.name}. Seed 2026. Frozen validation-selected best checkpoints.',
           'Test: 4,107 images / 240 patients / 4 observed centers. Lower R_final is better.',
           'R_final = 0.4 image COR + 0.4 patient-max COR + 0.2 sqrt-center-weighted patient-max COR.','']
    for name in ('primary','image_metrics','patient_metrics','auxiliary_metrics'):
        lines.extend([f'## {name}','',pd.DataFrame(test[name]).to_csv(index=False),''])
    if bootstrap: lines.extend(['## 配对患者区间','',json.dumps(bootstrap,ensure_ascii=False,indent=2),'',
                                'Descriptive patient sampling interval, not training-seed variability or external generalization.',''])
    lines.extend(['## checkpoint 选择/训练诊断','',pd.DataFrame(val['checkpoints']).to_csv(index=False),'',
                  'Every epoch validation metrics and TensorBoard are retained. Checkpoints at every ten completed epochs plus best/last.',
                  'Full metrics, confusion matrices and per-center results are in each evaluation/metrics.json.',
                  'Resources measured at batch 1 and 8 on the same GPU; optional causal/route diagnostics are not main endpoints.'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(root/'RUN_COMPLETE.json',{'status':'COMPLETE','methods':args.methods,'seed':2026,'test_complete':True,
               'report_sha256':sha256(root/'REPORT.md'),'scope':'main methods only; optional architecture ablations not executed'})


if __name__=='__main__': main()
