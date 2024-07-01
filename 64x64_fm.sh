export OPENAI_LOGDIR=/data/openai_log/64x64_fm
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --log_interval 10 --save_interval 1000 --warm_checkpoint /data/checkpoint/models/64x64_diffusion.pt"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=1 python -m scripts.image_train --data_dir /data/imagenet/sub $MODEL_FLAGS $TRAIN_FLAGS

# from 64x64_fm 
# 64x64_fm은 ema가 초반에 빠르게 수렴하지 않으므로, 64x64_fm의 step 5000에서부터 warm start
export OPENAI_LOGDIR=/data/openai_log/64x64_fm_warm
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --log_interval 10 --save_interval 1000 --warm_checkpoint /data/openai_log/64x64_fm/model005000.pt"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=1 python -m scripts.image_train --data_dir /data/imagenet/sub $MODEL_FLAGS $TRAIN_FLAGS

#위에꺼 이어서
export OPENAI_LOGDIR=/data/openai_log/64x64_fm_warm2
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --log_interval 10 --save_interval 1000 --resume_checkpoint /data/openai_log/64x64_fm_warm/model111000.pt"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=1 python -m scripts.image_train --data_dir /data/imagenet/sub $MODEL_FLAGS $TRAIN_FLAGS

#위에꺼 이어서
export OPENAI_LOGDIR=/data/openai_log/64x64_fm_warm3
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --log_interval 10 --save_interval 1000 --resume_checkpoint /data/openai_log/64x64_fm_warm2/model340000.pt"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=1 python -m scripts.image_train --data_dir /data/imagenet/sub $MODEL_FLAGS $TRAIN_FLAGS

#위에꺼 이어서
export OPENAI_LOGDIR=/data/openai_log/64x64_fm_warm4
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --log_interval 10 --save_interval 1000 --resume_checkpoint /data/openai_log/64x64_fm_warm3/model444000.pt"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=1 python -m scripts.image_train --data_dir /data/imagenet/sub $MODEL_FLAGS $TRAIN_FLAGS

#위에꺼 이어서
export OPENAI_LOGDIR=/data/openai_log/64x64_fm_warm5
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --log_interval 10 --save_interval 1000 --resume_checkpoint /data/openai_log/64x64_fm_warm4/model504000.pt"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=1 python -m scripts.image_train --data_dir /data/imagenet/sub $MODEL_FLAGS $TRAIN_FLAGS

#위에꺼 이어서
export OPENAI_LOGDIR=/data/openai_log/64x64_fm_warm6
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --log_interval 10 --save_interval 1000 --resume_checkpoint /data/openai_log/64x64_fm_warm5/model727000.pt"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=1 python -m scripts.image_train --data_dir /data/imagenet/sub $MODEL_FLAGS $TRAIN_FLAGS

