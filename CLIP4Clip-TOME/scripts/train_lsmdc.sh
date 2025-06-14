DATA_PATH=data/lsmdc
python -m torch.distributed.launch --nproc_per_node=1 \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=5 --batch_size=16 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/videos_compress \
--output_dir ckpts/ckpt_lsmdc_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype lsmdc --feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16 --video_reader cv2