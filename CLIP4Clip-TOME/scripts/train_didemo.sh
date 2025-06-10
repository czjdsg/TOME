DATA_PATH=/mnt/main/zjcai/LocalizingMoments-master
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29502 \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=5 --batch_size=16 --n_display=50 \
--data_path ${DATA_PATH}/data \
--features_path ${DATA_PATH}/videos \
--output_dir ckpts/ckpt_didemo_retrieval \
--lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 16 \
--datatype didemo --feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16 \
