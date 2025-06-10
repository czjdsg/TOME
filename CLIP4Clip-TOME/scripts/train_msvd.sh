DATA_PATH=/mnt/main/zjcai/msvd
CUDA_VISIBLE_DEVICES=3,4 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=32 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/videos_compress \
--output_dir ckpts/ckpt_msrvtt_retrieval_msvd \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 8 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16 \
