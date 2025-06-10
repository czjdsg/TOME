DATA_PATH=data/msrvtt
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 \
main_task_retrieval.py --do_eval --num_thread_reader=4 \
--epochs=5 --batch_size=16 --n_display=50 \
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
--init_model output/msrvtt/clip4clip_vitb16/pytorch_model.bin.0 \
--pretrained_clip_name ViT-B/16 --freeze_clip  --atp --atp_input_layer=0