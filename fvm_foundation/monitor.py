import json, sys, torch
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src'))
sys.path.insert(0, str(HERE.parents[0] / 'fvm_gen'))
sys.path.insert(0, str(HERE.parents[0] / 'fvm_solver'))

from helper import N_CHANNELS, INPUT_STATS_PATH, DELTA_STATS_PATH
from data import FVMDataModule
from lightning_model import FVMLightningModel

DATA   = HERE.parent / 'data' / 'fvm_gen_datasets'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

dm = FVMDataModule(data_dir=DATA, batch_size=4, num_workers=0)
dm.setup()
loader = dm.train_dataloader()

batches = []
for w, t in loader:
    batches.append((w.to(device), t.to(device)))
    if len(batches) == 2: break
fixed_w, fixed_t = batches[0]

model = FVMLightningModel(lr=1e-4)
with open(DELTA_STATS_PATH) as f: ds = json.load(f)
model.delta_mean.copy_(torch.tensor(ds['mean']).view(N_CHANNELS,1,1))
model.delta_std.copy_(torch.tensor(ds['std']).view(N_CHANNELS,1,1))
with open(INPUT_STATS_PATH) as f: ins = json.load(f)
model.input_mean.copy_(torch.tensor(ins['mean']).view(N_CHANNELS,1,1))
model.input_std.copy_(torch.tensor(ins['std']).view(N_CHANNELS,1,1))
model = model.to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-4)

def compute_loss(w, t):
    pred  = model(w)
    tn    = (t - model.delta_mean) / model.delta_std
    valid = torch.isfinite(tn)
    return (pred - tn).abs()[valid].mean()

def grad_norm():
    return sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5

def weight_norm():
    return sum(p.norm().item()**2 for p in model.parameters())**0.5

print("\n--- target magnitude distribution across 100 batches ---")
batch_losses = []
for i, (w, t) in enumerate(loader):
    tn    = (t.to(device) - model.delta_mean) / model.delta_std
    valid = torch.isfinite(tn)
    batch_losses.append(tn[valid].abs().mean().item())
    if i >= 99: break
batch_losses.sort()
n = len(batch_losses)
for label, idx in [("p10", n//10), ("p50", n//2), ("p90", int(n*0.9)), ("p99", int(n*0.99)), ("max", -1)]:
    print(f"  {label}: {batch_losses[idx]:.4f}")
print(f"  spike ratio p90/p50: {batch_losses[int(n*0.9)] / batch_losses[n//2]:.1f}x")

print("\nstep  train_loss  fixed_loss  grad_norm  weight_norm")
step = 0
for _ in range(6):
    for w, t in loader:
        w, t = w.to(device), t.to(device)
        model.train()
        opt.zero_grad()
        tr_loss = compute_loss(w, t)
        tr_loss.backward()
        gn = grad_norm()
        opt.step()
        model.eval()
        with torch.no_grad():
            fi_loss = compute_loss(fixed_w, fixed_t)
        if step % 5 == 0:
            print(f"{step:4d}  {tr_loss.item():.5f}     {fi_loss.item():.5f}     {gn:.4f}    {weight_norm():.2f}")
        step += 1
        if step >= 100: break
    if step >= 100: break
