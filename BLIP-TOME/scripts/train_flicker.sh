CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.run --nproc_per_node=4 --master_addr 127.0.0.8 --master_port 29507 train_video_retrieval.py --config configs/retrieval_flicker_ToMe.yaml --output_dir ./output/video_retrieval_flicker/ToMe_base \
#  --mixed_precision_method apex