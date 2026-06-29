RUN_ID=$(date +%Y%m%d_%H%M%S)

GPU_IDS="2,3"
NUM_GPUS=2
cfg="config/SSMT.yaml"
saveroot="experiments"
exp_name="ssmt"
encoder_ckpts="experiments/tadpm_pretrain/20260111_153140/checkpoints/ckpt-best.pth"
num_workers=16

export CUDA_VISIBLE_DEVICES=$GPU_IDS
accelerate launch --num_processes=$NUM_GPUS --num_machines=1 --mixed_precision=no main.py \
    --config $cfg \
    --num_workers $num_workers \
    --save_root $saveroot \
    --exp_name $exp_name \
    --run_id $RUN_ID \
    --encoder_ckpts $encoder_ckpts
