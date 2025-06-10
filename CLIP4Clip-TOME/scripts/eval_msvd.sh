DATA_PATH=data/msvd
python -m torch.distributed.launch --nproc_per_node=1 \
main_task_retrieval.py --do_eval --num_thread_reader=4 \
--epochs=5 --batch_size=64 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/videos_compress \
--output_dir ckpts/ckpt_msvd_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32 \
--init_model output/msvd/clip4clip/pytorch_model.bin.1 --video_padding True
# --enable_tome --tome_hierarchy \
# --tome_type="soft_s_multi_cls" --frame_flatten_type="keep" --tome_average_type="AVG" \
# --tome_r 36 8 8 8 6 4 4 4 4 4 4 4 \
# --tome_prop_attn \
# --frame_flatten_layer 0 4 8 \
# --flatten_layer_merge_type "T" "T" "S" \