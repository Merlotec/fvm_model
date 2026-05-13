python fvm_foundation/infer.py \
$(ls fvm_foundation/checkpoints/*.ckpt | sort -t'=' -k3,3n | head -n 1) \
out/test_infer_overfit -r 1 --data-dir data/fvm_gen_overfit
