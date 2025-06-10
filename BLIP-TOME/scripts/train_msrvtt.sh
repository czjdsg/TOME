CUDA_VISIBLE_DEVICES=5 \
python -m torch.distributed.run --nproc_per_node=1 --master_addr 127.0.0.8 --master_port 29508 train_video_retrieval.py --config configs/retrieval_msrvtt_ToMe.yaml --output_dir ./output/video_retrieval_msrvtt/Original_f12 --accumulation_steps 1 \
# --low_resource_eval 
#  --mixed_precision_method apex