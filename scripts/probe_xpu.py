"""Probe whether the FVM solver's device-dependent ops work on this machine's XPU.

Run on the cluster node:  python scripts/probe_xpu.py

Tests, in the order the solver hits them:
  1. basic dense ops on xpu
  2. sparse COO construction + coalesce + to_sparse_csr on xpu   (solver setup)
  3. sparse CSR mm/addmm/mv/addmv on xpu                         (SPMGeneral hot loop)
  4. triton import + the solver's ELL SpMV kernel on xpu         (SPMCuda alternative)
  5. torch.compile of a small function on xpu

If (2)+(3) pass -> FVM_DEVICE=xpu should work as-is.
If they fail but (4) passes -> the triton ELL path can be enabled for xpu instead.
If both fail -> run FVM_DEVICE=cpu (or ask for a CPU-side sparse fallback).
"""
import sys

import torch

results = {}


def step(name):
    def deco(fn):
        try:
            fn()
            results[name] = "OK"
            print(f"  [OK]   {name}")
        except Exception as e:
            results[name] = f"FAIL: {type(e).__name__}: {e}"
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        return fn
    return deco


print(f"torch {torch.__version__}")
has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
print(f"torch.xpu available: {has_xpu}")
if not has_xpu:
    print("No XPU visible — nothing to probe (check the module/env, e.g. `module load "
          "intel-oneapi` and a torch build with xpu support).")
    sys.exit(1)
print(f"device: {torch.xpu.get_device_name(0)}\n")

dev = torch.device("xpu")

A_cpu = torch.zeros(64, 64)
A_cpu[torch.arange(64), torch.arange(64)] = 2.0
A_cpu[torch.arange(63), torch.arange(1, 64)] = -1.0
x_cpu = torch.randn(64)
X_cpu = torch.randn(64, 4)
want_mv = A_cpu @ x_cpu
want_mm = A_cpu @ X_cpu


@step("dense ops on xpu")
def _dense():
    a = torch.randn(256, 256, device=dev)
    ((a @ a).relu().sum() * 2).cpu()


@step("sparse COO build + coalesce + to_sparse_csr on xpu")
def _sparse_build():
    coo = A_cpu.to_sparse_coo().to(dev).coalesce()
    csr = coo.to_sparse_csr()
    assert csr.device.type == "xpu"


@step("sparse CSR mm/addmm/mv/addmv on xpu (SPMGeneral hot loop)")
def _sparse_ops():
    A = A_cpu.to_sparse_csr()
    A = torch.sparse_csr_tensor(A.crow_indices().to(torch.int32).to(dev),
                                A.col_indices().to(torch.int32).to(dev),
                                A.values().to(dev), size=A.size(), device=dev)
    x, X = x_cpu.to(dev), X_cpu.to(dev)
    assert torch.allclose(torch.mv(A, x).cpu(), want_mv, atol=1e-4)
    assert torch.allclose(torch.sparse.mm(A, X).cpu(), want_mm, atol=1e-4)
    b1, bM = torch.zeros(64, device=dev), torch.zeros(64, 4, device=dev)
    torch.addmv(b1, A, x)
    torch.addmm(bM, A, X)


@step("triton import + solver ELL SpMV kernel on xpu")
def _triton():
    import os
    solver_dir = os.environ.get(
        "FVM_SOLVER_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fvm_solver")))
    sys.path.insert(0, solver_dir)
    from time_fvm.utils.ell_kernel import csr_to_ell, ell_spmv
    csr = A_cpu.to_sparse_csr()
    vals, cols = csr_to_ell(csr)           # built on CPU, shipped to xpu
    y = ell_spmv(vals.to(dev), cols.to(dev), x_cpu.to(dev))
    assert torch.allclose(y.cpu(), want_mv, atol=1e-4)


@step("torch.compile on xpu")
def _compile():
    def f(t):
        return (t * 2 + 1).sin().sum()
    fc = torch.compile(f)
    a = torch.randn(128, 128, device=dev)
    assert torch.allclose(fc(a).cpu(), f(a).cpu(), atol=1e-4)


print("\n--- verdict ---")
sp = results.get("sparse COO build + coalesce + to_sparse_csr on xpu", "") == "OK" and \
     results.get("sparse CSR mm/addmm/mv/addmv on xpu (SPMGeneral hot loop)", "") == "OK"
tr = results.get("triton import + solver ELL SpMV kernel on xpu", "") == "OK"
if sp:
    print("FVM_DEVICE=xpu should work as-is (SPMGeneral path).")
    if not results.get("torch.compile on xpu", "") == "OK":
        print('torch.compile failed: set "compile": false in the gen config.')
elif tr:
    print("Sparse CSR unsupported, but triton ELL works: the solver's SPMCuda/ELL path "
          "can be enabled for xpu (small change to time_fvm/utils/sparse.py).")
else:
    print("Neither sparse CSR nor triton ELL works on xpu here: run FVM_DEVICE=cpu, "
          "or the sparse mats need a CPU/dense fallback.")
