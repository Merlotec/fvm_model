"""Fail unless everything the FVM generation path imports is present.

Run by scripts/isambard_pair.slurm before it writes the .installed marker, so a
partially built venv is never mistaken for a finished one.  That failure mode is
easy to hit: requirements.txt carries a git+https dependency, `uv pip install -r`
is a single transaction, and a blocked git fetch therefore leaves the venv with
torch (installed by an earlier command) and nothing else.
"""
import importlib.util
import sys

# cprint is the git dependency (Maccy-Z/c_print); fvm_equation.py and
# fvm_mesh.py both import it, so it is not optional for generation.
REQUIRED = ['numpy', 'scipy', 'torch', 'gmsh', 'shapely', 'cprint', 'matplotlib',
            'codetiming']

missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
if missing:
    print(f'venv is INCOMPLETE — missing: {", ".join(missing)}', file=sys.stderr)
    print(f'interpreter: {sys.executable}', file=sys.stderr)
    sys.exit(1)

import torch                                                   # noqa: E402
print(f'venv ok — python {sys.version.split()[0]}, torch {torch.__version__}, '
      f'cuda available {torch.cuda.is_available()}')
