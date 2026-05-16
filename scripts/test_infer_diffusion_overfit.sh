cd "$(dirname "$0")/.." || exit 1

python fvm_foundation/infer_diffusion.py \
    $(ls fvm_foundation/checkpoints_diffusion/*.ckpt | sort | tail -n 1) \
    out/test_infer_diffusion_overfit \
    -r 1 \
    --data-dir data/fvm_gen_overfit
