"""Export traceable paper tables and test-first report from completed evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from synap_search.io import sha256, write_json


def export_split(root, methods, split):
    destination=root/'paper_ready'/split; destination.mkdir(parents=True,exist_ok=True)
    metrics={m:json.loads((root/m/'evaluation'/split/'metrics.json').read_text(encoding='utf-8')) for m in methods}
    ranking={m:i+1 for i,m in enumerate(sorted(methods,key=lambda m:metrics[m]['r_final']))}
    tables={name:[] for name in ('primary','per_seed','image_metrics','patient_metrics','auxiliary_metrics','checkpoints','training')}
    bindings={}
    for method in methods:
        m=metrics[method]; run=root/method; freeze=json.loads((run/'CHECKPOINT_FREEZE.json').read_text(encoding='utf-8'))
        base={'method_id':method,'method_name':method,'seed':2026,'status':'COMPLETE'}
        delta=m['r_final']-metrics['FULL']['r_final'] if 'FULL' in metrics else None
        relative=100*delta/metrics['FULL']['r_final'] if delta is not None and metrics['FULL']['r_final']!=0 else None
        resource=json.loads((run/'resources.json').read_text(encoding='utf-8'))[0]
        main={**base,'comparison_round':root.name,'selected_epoch':freeze['epoch'],'checkpoint_frozen':True,
              'parameters':resource['parameters'],'checkpoint_sha256':freeze['checkpoint_sha256'],'r_final':m['r_final'],'image_cor':m['image']['cor'],'patient_max_cor':m['patient_max']['cor'],
              'patient_median_cor':m['patient_median']['cor'],'center_cor':m['center_balanced_patient_max']['cor'],
              'delta_abs_vs_full':delta,'delta_rel_pct_vs_full':relative}
        tables['primary'].append({**main,'rank':ranking[method]})
        path=run/'evaluation'/split/'metrics.json'
        tables['per_seed'].append({**main,
            'n_images':m['image']['n'],'n_patients':m['patient_max']['n'],'n_centers':m['center_balanced_patient_max']['center_count'],
            'checkpoint_sha256':freeze['checkpoint_sha256'],'metrics_path':str(path),'metrics_sha256':sha256(path)})
        fields={'mae':'mae','acc_03':'accuracy_within_0_3','acc_05':'accuracy_within_0_5','accuracy_4grade':'grade_accuracy',
                'tmae':'tmae','severe_error_rate':'severe_error_rate','clinical_risk':'cor'}
        tables['image_metrics'].append({**base,'n_images':m['image']['n'],**{out:m['image'][key] for out,key in fields.items()}})
        patient={**base,'n_patients':m['patient_max']['n'],'n_centers':m['center_balanced_patient_max']['center_count'],
                 'center_clinical_risk':m['center_balanced_patient_max']['cor']}
        for agg in ('max','median'): patient.update({f'{agg}_{out}':m[f'patient_{agg}'][key] for out,key in fields.items()})
        tables['patient_metrics'].append(patient)
        tables['auxiliary_metrics'].append({**base,'view_status':'COMPLETE' if m['position'].get('available') else 'N/A', 'weakloc_status':'COMPLETE' if m['lesion'].get('available') else 'N/A',
             'n_view_images':m['position'].get('n'),'view_accuracy':m['position'].get('accuracy'),'view_macro_f1':m['position'].get('macro_f1'),'view_ce':m['position'].get('ce'),
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
    for method in methods:
        resources.extend([{**row, 'method_id':method,'method_name':method} for row in json.loads((root/method/'resources.json').read_text(encoding='utf-8'))])
    pd.DataFrame(resources).to_csv(destination/'resources.csv',index=False)
    write_json(destination/'source_bindings.json',bindings)
    return tables


def main():
    p=argparse.ArgumentParser(); p.add_argument('--round-root',type=Path,required=True);p.add_argument('--methods',nargs='+',default=['FULL','RE_WO_VIEW','RE_WO_WEAKLOC'])
    args=p.parse_args(); root=args.round_root
    if args.methods != ['FULL','RE_WO_VIEW','RE_WO_WEAKLOC']:
        raise ValueError('Exactly Full reuse and two deletion variants required')
    if sha256(root/'FULL'/'best.pt') != 'fc2ab7c366ff8d981b07bb97d62c22782f2ff1d84e9f29e231b9d6f88d2407b3':
        raise ValueError('Full reference changed')
    for method in args.methods:
        run=root/method
        for name in ('TRAINING_COMPLETE.json','CHECKPOINT_FREEZE.json','EVALUATION_COMPLETE.json','resources.json'):
            if not (run/name).is_file(): raise RuntimeError(f'Missing completion artifact: {run/name}')
        freeze=json.loads((run/'CHECKPOINT_FREEZE.json').read_text(encoding='utf-8'))
        completed=json.loads((run/'TRAINING_COMPLETE.json').read_text(encoding='utf-8'))
        history=[json.loads(line) for line in (run/'history.jsonl').read_text(encoding='utf-8').splitlines()]
        if completed['epochs']!=120 or [h['epoch'] for h in history]!=list(range(1,121)) or not 21<=freeze['epoch']<=120:
            raise ValueError('Incomplete training horizon or invalid selected epoch')
        for epoch in range(10,121,10):
            if not (run/f'epoch_{epoch:03d}.pt').is_file(): raise ValueError('Missing periodic checkpoint')
        if sha256(run/'best.pt')!=freeze['checkpoint_sha256']: raise ValueError('Frozen checkpoint changed')
        for split in ('val','test'):
            dest=run/'evaluation'/split; provenance=json.loads((dest/'provenance.json').read_text(encoding='utf-8'))
            for filename,key in [('metrics.json','metrics_sha256'),('predictions.csv.gz','predictions_sha256')]:
                if sha256(dest/filename)!=provenance[key]: raise ValueError('Evaluation artifact changed')
            for agg,digest in provenance['patient_predictions_sha256'].items():
                if sha256(dest/f'patient_{agg}_predictions.csv.gz')!=digest: raise ValueError('Patient predictions changed')
    test=export_split(root,args.methods,'test'); val=export_split(root,args.methods,'val')
    lines=['# Test set 主结果','',f'Comparison round: {root.name}. Seed 2026. Frozen validation-selected best checkpoints. Full reused without retraining.',
           'Test: 4,107 images / 240 patients / 4 observed centers. Lower R_final is better.',
           'R_final = 0.4 image COR + 0.4 patient-max COR + 0.2 sqrt-center-weighted patient-max COR.','']
    for name in ('primary','image_metrics','patient_metrics','auxiliary_metrics'):
        lines.extend([f'## {name}','',pd.DataFrame(test[name]).to_csv(index=False),''])
    lines.extend(['## checkpoint 选择/训练诊断','',pd.DataFrame(val['checkpoints']).to_csv(index=False),'',
                  'Every epoch validation metrics and TensorBoard are retained. Checkpoints at epochs 10,20,...,120 plus best/last.',
                  'Full metrics, confusion matrices and per-center results are in each evaluation/metrics.json.',
                  'New variants resources measured at batch1/8 on A100 GPU0; Full resources reused from the preceding round.',
                  'Delta=R_variant-R_Full. Single-seed whole-module conditional effects; no synergy or isolated supervision/injection claim. No bootstrap in this round.'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(root/'RUN_COMPLETE.json',{'status':'COMPLETE','methods':args.methods,'seed':2026,'test_complete':True,
               'report_sha256':sha256(root/'REPORT.md'),'scope':'exactly two deletion trainings; Full reused; no bootstrap, scans or extra seeds'})


if __name__=='__main__': main()
