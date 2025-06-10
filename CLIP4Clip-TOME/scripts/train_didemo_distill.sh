DATA_PATH=data/didemo
CUDA_VISIBLE_DEVICES=3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29502 \ 
main_task_distill.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=32 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/videos_compress \
--output_dir ckpts/ckpt_didemo_retrieval_looseType \
--lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 16 \
--datatype didemo --feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16 \
--distill_scale 0.07 \
--distill_type mse \
--distill_weight 1 \
--enable_tome --tome_hierarchy \
--tome_type="soft_s_multi_cls" --frame_flatten_type="keep" --tome_average_type="AVG" \
--tome_r 130 48 48 48 48 32 32 32 32 32 32 32 \
--tome_prop_attn \
--frame_flatten_layer 0 4 8 \
--flatten_layer_merge_type "T" "T" "S" \
--init_teacher_model /mnt/main/zjcai/CLIP4Clip/ckpts/pytorch_model.bin.2 \