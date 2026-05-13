python fvm_foundation/infer.py fvm_foundation/checkpoints/model-epoch=057-train_loss=0.03995.ckpt out/test_infer -r 1 --data-dir data/overfit

python fvm_foundation/infer.py \
$(ls fvm_foundation/checkpoints/*.ckpt | sort -t'=' -k3,3n | head -n 1) \
out/test_infer -r 1 --data-dir data/fvm_gen_datasets
