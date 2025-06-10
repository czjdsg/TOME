EXP_NAME="clip4clip_vitb16_12frame_2x_distill_pad12"

DATA_PATH=data/lsmdc
python -m torch.distributed.launch --nproc_per_node=1 \
main_task_distill.py --do_train --num_thread_reader=4 \
--epochs=5 --batch_size=32 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/videos_compress \
--output_dir ${LOGDIR}${EXP_NAME} \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype lsmdc --feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16 \
--enable_tome --tome_hierarchy \
--tome_type="soft_s_multi_cls" --frame_flatten_type="keep" --tome_average_type="AVG" \
--tome_r 130 32 32 32 32 24 24 24 24 24 24 24 \
--frame_flatten_layer 0 4 8 \
--tome_prop_attn \
--flatten_layer_merge_type "T" "T" "S" --padding_base=12