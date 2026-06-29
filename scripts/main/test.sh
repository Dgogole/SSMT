GPU_IDS="0"
cfg="config/SSMT.yaml"
saveroot="experiments"
exp_name="ssmt_test"
ckpts="experiments/ssmt/20260616_164223/checkpoints/ckpt-best.pth"
encoder_ckpts="experiments/tadpm_pretrain/20260111_153140/checkpoints/ckpt-best.pth"
run_id=$(date +%Y%m%d_%H%M%S)

export CUDA_VISIBLE_DEVICES=$GPU_IDS

python main.py --config $cfg \
    --save_root $saveroot \
    --exp_name $exp_name \
    --run_id $run_id \
    --encoder_ckpts $encoder_ckpts \
    --ckpts $ckpts \
    --test
