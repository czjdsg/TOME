DATA_PATH=/mnt/main/zjcai/msrvtt
CUDA_VISIBLE_DEVICES=6 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 \
main_task_retrieval.py --do_eval --num_thread_reader=4 \
--epochs=5 --batch_size=32 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/MSRVTT_Videos_Compression \
--output_dir ckpts/ckpt_msrvtt_weime \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 32 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/16 \
--enable_weime --weime_r_neuron 100 --weime_r_head 0 --max_samples 1000 \
--init_model /mnt/main/zjcai/CLIP4Clip/ckpts/pytorch_model.bin.2 \
# --enable_tome --tome_hierarchy \
# --tome_type="soft_s_multi_cls" --frame_flatten_type="keep" --tome_average_type="AVG" \
# --tome_prop_attn \
# --tome_r 37 16 16 16 16 16 16 8 8 8 8 8 \
# --frame_flatten_layer 0 4 8 \
# --flatten_layer_merge_type "T" "T" "S" \

# --enable_frame_pos_embed