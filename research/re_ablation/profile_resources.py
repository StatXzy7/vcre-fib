"""Measure both frozen models on the same GPU after scoring finishes."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import runtime as rt


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--method', required=True)
    args = p.parse_args(); rank, world, device = rt.setup()
    if world != 1: raise ValueError('Resource measurements require a single idle GPU')
    model, _, _ = rt.build(args.method, args.root, device, 1, distributed=False)
    model.load_state_dict(torch.load(args.output/'best.pt', map_location='cpu', weights_only=False)['model'])
    model.eval(); count = [0]
    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d): count[0] += output.numel() * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear): count[0] += output.numel() * module.in_features
    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model(torch.zeros(1,3,512,512,device=device))
    for h in hooks: h.remove()
    rows = []
    for batch in (1,8):
        image=torch.zeros(batch,3,512,512,device=device)
        for _ in range(10): model(image)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); elapsed=[]
        for _ in range(50):
            begin=time.perf_counter(); model(image); torch.cuda.synchronize(); elapsed.append((time.perf_counter()-begin)*1000)
        rows.append({'method_id':args.method,'method_name':args.method,
                     'batch_size':batch,'status':'COMPLETE',
                     'parameters':sum(p.numel() for p in model.parameters()), 'active_parameters':sum(p.numel() for p in model.parameters()),
                     'gmacs':count[0]/1e9,'latency_median_ms':float(np.median(elapsed)),'latency_p95_ms':float(np.percentile(elapsed,95)),
                     'peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'device':torch.cuda.get_device_name(device),
                     'precision':'fp32','software_fingerprint':rt.sha256(args.output/'IDENTITY.json'),
                     'mac_definition':'per image, executed Conv2d and Linear only; excludes elementwise/pooling/normalization',
                     'memory_unit':'MiB','warmup':10,'repeats':50})
    rt.write_json(args.output/'resources.json',rows)


if __name__=='__main__': main()
