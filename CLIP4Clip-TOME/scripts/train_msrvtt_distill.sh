DATA_PATH=/mnt/main/zjcai/msrvtt
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 \
main_task_distill.py --do_train --num_thread_reader=4 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/MSRVTT_Videos_Compression \
--output_dir ckpts/ckpt_msrvtt_retrieval_msrvtt_tome_3x_distill \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32 \
--distill_scale 0.07 \
--distill_type mse \
--distill_weight 1 \
--init_teacher_model /mnt/main/zjcai/CLIP4Clip/ckpts/pytorch_model.bin.2 \
--enable_tome --tome_hierarchy \
--tome_type="soft_s_multi_cls" --frame_flatten_type="keep" --tome_average_type="AVG" \
--tome_r 130 48 48 48 48 32 32 32 32 32 32 32 \
--tome_prop_attn \
--frame_flatten_layer 0 4 8 \
--flatten_layer_merge_type "T" "T" "S" \
# --enable_weime --weime_r_neuron 0 --weime_r_head 0 --max_samples 1000 \
# --init_model /mnt/main/zjcai/CLIP4Clip/ckpts/pytorch_model.bin.2 \
# --weime_pretrained_model /mnt/main/zjcai/CLIP4Clip/ckpts/ckpt_msrvtt_weime_vit16_merge/pytorch_model.bin.pretrained 
