python fvm_latent/infer.py \
  $(ls fvm_latent/checkpoints/*.ckpt | grep -v last | sort | tail -n 1) \
  out/test_infer_latent -r 1 --data-dir data/fvm_gen_datasets
