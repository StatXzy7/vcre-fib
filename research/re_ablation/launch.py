"""Durable bounded controller: train, freeze, evaluate, resources, final report."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from synap_search.io import write_json


def group(commands):
    running=[]
    try:
        for cmd,environment,log_path in commands:
            log_path.parent.mkdir(parents=True,exist_ok=True)
            handle=log_path.open('x',encoding='utf-8')
            proc=subprocess.Popen(cmd,env=environment,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            running.append((proc,handle,log_path))
        while running:
            for entry in running[:]:
                proc,handle,path=entry; code=proc.poll()
                if code is None: continue
                handle.close(); running.remove(entry)
                if code: raise RuntimeError(f'{path}: process exited {code}')
            time.sleep(5)
    finally:
        for proc,handle,_ in running:
            if proc.poll() is None: os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            handle.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--round-root',type=Path,required=True)
    p.add_argument('--allocation',required=True,help='JSON mapping method to physical GPU list');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args(); allocation=json.loads(args.allocation); scripts=Path(__file__).parent
    if args.round_root.exists(): raise FileExistsError('Never overwrite an existing round')
    if set(allocation) != {'RE_WO_VIEW','RE_WO_WEAKLOC'} or any(len(v)!=4 for v in allocation.values()):
        raise ValueError('Exactly two variants on four GPUs each')
    devices=[gpu for group in allocation.values() for gpu in group]
    if len(devices)!=len(set(devices)): raise ValueError('GPU allocations overlap')
    args.round_root.mkdir(parents=True)
    write_json(args.round_root/'LAUNCH.json',{'allocation':allocation,'workers_per_rank':args.workers,'pid':os.getpid(),'time':time.time()})
    full=args.root/'outputs/zgc_main_seed2026_20260925/SYNAP'
    (args.round_root/'FULL').symlink_to(full, target_is_directory=True)
    try:
        for mode in ('train','evaluate'):
            jobs=[]
            for method,gpus in allocation.items():
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),NCCL_DEBUG='WARN',PYTHONUNBUFFERED='1')
                cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(gpus)}',str(scripts/'runtime.py'),mode,
                     '--root',str(args.root),'--output',str(args.round_root/method),'--method',method,'--round',args.round_root.name,'--workers',str(args.workers)]
                jobs.append((cmd,env,args.round_root/'logs'/f'{method}_{mode}.log'))
            group(jobs)
        # Same physical GPU and sequential measurements for fair resource rows.
        for method in allocation:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(devices[0]))
            group([([sys.executable,str(scripts/'profile_resources.py'),'--root',str(args.root),'--output',str(args.round_root/method),'--method',method],
                    env,args.round_root/'logs'/f'{method}_resources.log')])
        subprocess.run([sys.executable,str(scripts/'summarize.py'),'--round-root',str(args.round_root),'--methods','FULL','RE_WO_VIEW','RE_WO_WEAKLOC'],check=True)
    except Exception as exc:
        write_json(args.round_root/'FAILED.json',{'error':repr(exc),'time':time.time(),'automatic_retry':False})
        raise


if __name__=='__main__': main()

