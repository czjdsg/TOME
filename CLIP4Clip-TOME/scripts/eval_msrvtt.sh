DATA_PATH=data/msrvtt
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 \
main_task_retrieval.py --do_eval --num_thread_reader=4 \
--epochs=5 --batch_size=64 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/MSRVTT_Videos_Compression \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32 --video_reader decord \
--init_model ckpts/ckpt_msrvtt_retrieval_looseType/pytorch_model.bin.4 --video_padding True \

# output/msrvtt/clip4clip/pytorch_model.bin.1
# output/msrvtt/clip4clip_vitb16/pytorch_model.bin.0
# --enable_tome --tome_hierarchy --tome_prop_attn \
# --tome_type="soft_s" --frame_flatten_type="keep" --tome_average_type="AVG" \
# --tome_r 16 16 16 16 16 16 16 16 16 16 16 16
# --frame_flatten_layer 0
# --tome_prop_attn 
# 24 24 24 64 32 32 32 64 64 64 64 64
# 24 24 24 24 24 24 24 24 24 24 24 24
# 16 16 16 16 132 16 16 16 16 16 16 16
# 0 0 0 0 0 0 0 0 0 0 0 0
# 192 96 10 10 10 10 10 10 10 10 10 10
# --tome_token_importance --tome_importance_alpha=2.