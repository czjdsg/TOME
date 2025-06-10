DATA_PATH=data/msrvtt
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 \
main_task_retrieval.py --do_eval --num_thread_reader=4 \
--epochs=5 --batch_size=32 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/MSRVTT_Videos_Compression \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16 \
--init_model output/msrvtt/clip4clip_vitb16_12frame_3x_pad12/pytorch_model.bin.2 \
# --enable_tome --tome_hierarchy \
# --tome_type="soft_s_multi_cls" --frame_flatten_type="keep" --tome_average_type="AVG" \
# --tome_r 0 0 0 0 0 0 0 0 0 0 0 0 \
# --tome_prop_attn \
# --frame_flatten_layer 0 4 8 \
# --flatten_layer_merge_type "T" "T" "S" \

# --init_model output/msrvtt/clip4clip_vitb16_12frame_3x/pytorch_model.bin.1 \

# --init_model output/msrvtt/clip4clip/pytorch_model.bin.1 \

# --frame_mask \
# --merge_frame_mask_layer 1 2 3 5 6 7 9 10 11 \
# --attn_frame_mask_layer 1 5 9 \
# --tome_prop_attn \
# 130 32 32 32 32 24 24 24 24 24 24 24
# 130 48 48 48 48 32 32 32 32 32 32 32
# --tome_prop_attn \
# --frame_flatten_layer 0 4 8 \
# --flatten_layer_merge_type "T" "T" "S" \
# --merge_frame_mask_layer 1 2 3 5 6 7 9 10 11 \
# --frame_mask \
# --attn_frame_mask_layer 1 5 9 \
# --enable_frame_pos_embed \
# --binary_frame_mask \
# --merge_frame_mask_layer 2 3
# --binary_frame_mask
# --frame_mask
# --tome_prop_attn 
# 24 24 24 64 32 32 32 64 64 64 64 64
# 24 24 24 24 24 24 24 24 24 24 24 24
# 16 16 16 16 16 16 16 16 16 16 16 16
# 0 0 0 0 0 0 0 0 0 0 0 0
# 192 96 10 10 10 10 10 10 10 10 10 10
# 32 32 32 32 32 32 32 32 32 32 32 32
# 160 32 30 28 128 20 20 20 20 20 20 20
# 160 32 30 160 20 20 20 20 20 20 20 20
# --tome_token_importance --tome_importance_alpha=2.
# output/msrvtt/clip4clip_vitb16/pytorch_model.bin.0

# 参数说明：
# --frame_mask：是否维护frame_idx并计算frame_mask，默认为False
# --attn_frame_mask_layer: 在哪些层的attention中，使用frame_mask
# --merge_frame_mask_kayer: 在哪些层的merge中，使用frame_mask
# --binary_frame_mask: 是否使用0/1 mask，0/1 mask在计算每次时序merge之前，都会重新初始化frame_idx
# --tome_prop_attn: 是否使用ToMe里提出的size mask，只要使用空间的merge，就需要打开该设置