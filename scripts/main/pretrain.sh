export CUDA_VISIBLE_DEVICES=0,1
ENT="python pretrain.py "
cfg="config/pretrain.yaml"
saveroot="experiments"
exp_name="ssmt_pretrain"

NCCL_DEBUG=INFO $ENT --config $cfg \
    --save_root $saveroot \
    --exp_name $exp_name 
