CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --nproc_per_node=4 --master_addr 127.0.0.8 --master_port 29507 train_video_retrieval.py --config configs/retrieval_didemo_ToMe.yaml --output_dir ./output/video_retrieval_didemo/ToMe+WeiMe_base_f48 --accumulation_steps 1 \
# --low_resource_eval \
#  --mixed_precision_method apex 