from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import time
import torch
import numpy as np
import random
import tqdm
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_distill import CLIP4Clip
from modules.optimization import BertAdam
import tome
from weime.merge import neuron_merge, attention_head_merge

from util import parallel_apply, get_logger, str2bool
from dataloaders.data_dataloaders import DATALOADER_DICT

torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_speed_test", action='store_true', help="Whether to test inference speed.")
    parser.add_argument("--enable_tome", action='store_true', help="Whether to enable token merging.")
    parser.add_argument("--tome_prop_attn", action='store_true', help="Whether to enable proportion attention for token merging.")
    parser.add_argument("--tome_hierarchy", action='store_true', help="Whether to flatten and combine frame pairs for token merging.")
    parser.add_argument("--tome_token_importance", action='store_true', help="Whether to add token importance score for token merging.")
    parser.add_argument("--frame_selection", action='store_true', help="Whether to enable global CLS for frame selection.")
    parser.add_argument("--atp", action='store_true', help="Whether to enable atp for frame selection.")
    parser.add_argument("--freeze_clip", action='store_true', help="Whether to freeze clip.")
    parser.add_argument("--frame_mask", action='store_true', help="Whether to maintain the frame idx of each token for masked ToMe and SA.")
    parser.add_argument("--binary_frame_mask", action='store_true', help="Whether to use binary frame mask for ToMe and SA.")
    parser.add_argument("--enable_frame_pos_embed", action='store_true', help="Whether to use frame pos embeds at the clip input layer.")
    parser.add_argument("--sta_prune", action='store_true', help="Whether to use sta token pruning method.")
    parser.add_argument("--trace_source", action='store_true', help="Whether to trace source for visualization.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    parser.add_argument('--tome_type', type=str, default='soft_s', help='token merging type for ToMe')
    parser.add_argument('--frame_flatten_type', type=str, default='as_patch', help='different approaches to due with two frame CLS')
    parser.add_argument('--tome_average_type', type=str, default='AVG', help='averge type for merge src and dst token')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')
    parser.add_argument('--tome_r', nargs="+", type=int, default=2, help='Num of token to merge per layer')
    parser.add_argument('--tome_min_token', type=int, default=1, help='Num of token to merge per layer')
    parser.add_argument('--tome_importance_alpha', type=float, default=6.0, help='Num of token to merge per layer')
    parser.add_argument('--frame_flatten_layer', nargs="+", type=int, default=[], help='Num of token to merge per layer')
    parser.add_argument('--atp_num_layers', type=int, default=3, help='Num of token to merge per layer')
    parser.add_argument('--atp_input_layer', type=int, default=0, help='Num of token to merge per layer')
    parser.add_argument('--attn_frame_mask_layer', nargs="+", type=int, default=[], help='layer indices for applying frame mask to SA')
    parser.add_argument('--merge_frame_mask_layer', nargs="+", type=int, default=[], help='layer indices for applying frame mask to ToMe')
    parser.add_argument('--flatten_layer_merge_type', nargs="+", type=str, default=[], help='merge type in the frame flatten layer')
    parser.add_argument("--enable_weime", action='store_true', help="Whether to enable weight merging.")
    parser.add_argument('--weime_r_neuron', type=int, default=100, help='Num of neurons to merge per layer')
    parser.add_argument('--weime_r_head', type=int, default=2, help='Num of heads to merge per layer')
    parser.add_argument('--weime_layers', nargs="+", type=int, help='Layers to apply weime')
    parser.add_argument('--max_samples', type=int, default=1000, help='Num of samples to use for calulating neuron similarity')
        
    parser.add_argument('--video_reader', type=str, default="decord", help='approaches to load video')
    parser.add_argument('--video_padding', type=str2bool, default=True, help='whether to pad video')
    parser.add_argument('--padding_base', type=int, default=4, help='base num for video padding')
    parser.add_argument('--distill_weight', type=float, default=1., help='base num for video padding')
    parser.add_argument('--distill_scale', type=float, default=1., help='base num for video padding')

    parser.add_argument("--output_dir", default=None, type=str, required=True, # True
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--weime_pretrained_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--init_teacher_model", default=None, type=str, required=False, help="weights of teacher model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument("--distill_type", default="contrast", choices=["contrast", "mse", "cosine"], type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local-rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval and not args.do_speed_test:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    # n_gpu = torch.cuda.device_count()
    n_gpu = int(os.environ['WORLD_SIZE'])
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_teacher_model(args, device, n_gpu, local_rank):
    if args.init_teacher_model:
        model_state_dict = torch.load(args.init_teacher_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    model.eval()
    model.to(device)
    print("Teacher model device: {}".format(device))

    return model

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    if args.enable_frame_pos_embed:
            os.environ["enable_frame_pos_embed"] = "True"
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    if args.enable_tome:
        logger.warning("Enable Token Merging with setting as:")
        logger.warning("\t Reduction number: {}".format(args.tome_r))
        logger.warning("\t Prop attention: {}".format(args.tome_prop_attn))
        logger.warning("\t Merge type: {}".format(args.tome_type))
        logger.warning("\t Hierarchy: {}".format(args.tome_hierarchy))
        logger.warning("\t Token importance: {}".format(args.tome_token_importance))
        logger.warning("\t Token importance alpha: {}".format(args.tome_importance_alpha))
        logger.warning("\t Frame Flatten Type: {}".format(args.frame_flatten_type))
        logger.warning("\t Frame Flatten Layer: {}".format(args.frame_flatten_layer))
        logger.warning("\t Merge Average Type: {}".format(args.tome_average_type))
        logger.warning("\t Enable frame pos embed: {}".format(args.enable_frame_pos_embed))
        # patch_func = tome.patch.clip if not args.frame_mask else tome.patch.frame_mask_clip
        if args.sta_prune:
            patch_func = tome.patch.sta_clip
        else:
            patch_func = tome.patch.frame_mask_clip
        patch_func(
            model.clip.visual, 
            trace_source=args.trace_source,
            r=args.tome_r, 
            prop_attn=args.tome_prop_attn, 
            type=args.tome_type,
            min_token=args.tome_min_token,
            hierarchy=args.tome_hierarchy,
            token_importance=args.tome_token_importance,
            importance_alpha=args.tome_importance_alpha,
            frame_flatten_type=args.frame_flatten_type,
            average_type=args.tome_average_type,
            frame_flatten_layer=args.frame_flatten_layer,
            enable_frame_mask=args.frame_mask,
            attn_frame_mask_layer=args.attn_frame_mask_layer,
            merge_frame_mask_layer=args.merge_frame_mask_layer,
            binary_frame_mask=args.binary_frame_mask,
            flatten_layer_merge_type=args.flatten_layer_merge_type,
            )
    if args.frame_selection:
        logger.warning("Enable Frame Selection.")
        tome.patch.fs_clip(
            model.clip.visual,
            r=0,
            prop_attn=False,
        )
    model.to(device)
    print("Model device: {}".format(device))

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)        
        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, teacher_model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        # if n_gpu == 1:
            # multi-gpu does scattering it-self
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask = batch

        # forward teacher
        with torch.no_grad():
            teacher_visual_output, _, teacher_hidden_states = teacher_model.get_visual_output(video, video_mask, return_hidden_states=True)
        loss, loss_gf = model(input_ids, segment_ids, input_mask, video, video_mask, args.frame_selection, teacher_hidden_states=teacher_hidden_states, return_hidden_states=True)
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
            loss_gf = loss_gf.mean() if loss_gf is not None else None
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                if loss_gf is None:
                    logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                                args.epochs, step + 1,
                                len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                                float(loss),
                                (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                else:
                    logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Loss Distill: %f, Time/step: %f", epoch + 1,
                                args.epochs, step + 1,
                                len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                                float(loss),
                                float(loss_gf),
                                (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
                                                                     loose_type=model.loose_type)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def _topk_accuracy(preds, gts, topk=[1, 3]):
    # preds: B,num_layer,T
    # gts: B,1,T
    pred_idx = preds.argsort(dim=-1, descending=True) # B,num_layer,T
    gt_idx = gts.argsort(dim=-1, descending=True) # B,1,T
    topk_accs = []
    for k in topk:
        tmp_pred_idx = pred_idx[:, :, :k] # B,num_layer,k
        tmp_gt_idx = gt_idx[:, :, :k] # B,1,k
        all_accs = []
        for l in range(pred_idx.shape[1]):
            layer_accs = []
            for b in range(pred_idx.shape[0]):
                pred = set(tmp_pred_idx[b, l].tolist())
                gt = set(tmp_gt_idx[b, 0].tolist())
                acc = len(pred & gt) / k
                layer_accs.append(acc)
            all_accs.append(sum(layer_accs) / len(layer_accs))
        topk_accs.append(torch.Tensor(all_accs))
    return torch.stack(topk_accs, dim=0) # topk,num_layer


def eval_epoch(args, model, test_dataloader, device, n_gpu):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        frame_select_accs = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(tqdm.tqdm(test_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output, _, _ = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                sequence_output, visual_output, gf_sims, visual_hidden_states = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask, 
                                                            frame_selection=args.frame_selection, return_gf_sims=True, return_hidden_states=args.atp)
                if gf_sims is not None:
                    gf_sim_target = model.construct_gf_sim_target(sequence_output, visual_output, video_mask.squeeze(1))
                    frame_select_acc = _topk_accuracy(gf_sims, gf_sim_target)
                    frame_select_accs.append(frame_select_acc)
                if visual_hidden_states is not None:
                    atp_input = visual_hidden_states[model.atp_input_layer][0].view(sequence_output.shape[0], visual_output.shape[1], -1).transpose(0, 1) # L,BT,C ==> T,B,C
                    # atp_output = model.atp((atp_input.float(), visual_output.shape[1]))[0].transpose(0, 1) # T,B,C ==> B,T,C
                    atp_output = atp_input.float().transpose(0, 1)
                    atp_output = model.atp_linear(atp_output.float()).squeeze(-1) # B,T
                    gf_sim_tgt = model.construct_gf_sim_target(sequence_output, visual_output, video_mask.squeeze(1)).squeeze(1)
                    gf_sims = atp_output + (1 - video_mask.squeeze(1)) * -1e4
                    frame_select_acc = _topk_accuracy(gf_sims.unsqueeze(1), gf_sim_tgt.unsqueeze(1))
                    frame_select_accs.append(frame_select_acc)
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            # print("{}/{}\r".format(bid, len(test_dataloader)), end="")
        if len(frame_select_accs) > 0:
            frame_select_accs = torch.stack(frame_select_accs, dim=0).mean(0) # topk,num_layer
        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                      batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        else:
            sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
    
    if len(frame_select_accs) > 0:
        logger.info("Frame Selection Accuracy:")
        logger.info("\t>>> top-1: {} ".format(frame_select_accs[0]))
        logger.info("\t>>> top-3: {}".format(frame_select_accs[1],))

    R1 = tv_metrics['R1']
    return R1

def precompute_tensor_shape(args, num_frame=12, num_patch=197, num_layer=12):
    num_cls = 1
    ff_layer = args.frame_flatten_layer
    r = args.tome_r
    for _ in range(num_layer):
        pass

def speed_test(args, model, test_dataloader, device, n_gpu):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        warmup_step = 5
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch
            visual_output = model.get_visual_output(video, video_mask, frame_selection=args.frame_selection)
            if bid >= warmup_step:
                break
        
        repeat_time = 100
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(repeat_time):
            visual_output = model.get_visual_output(video, video_mask, frame_selection=args.frame_selection)
        torch.cuda.synchronize()
        end_time = time.time()
        duration = (end_time - start_time) / repeat_time
        logger.info("inference speed: {:.2f} ms/batch".format(duration * 1000))
        logger.info("Throughout: {:.2f} videos/s".format(1 / duration * args.batch_size_val))

    vit_num_layer = len(model.clip.visual.transformer.resblocks)
    if hasattr(model.clip.visual, "_tome_info"):
        logger.info("Tensor shape of different layers:")
        for l in range(vit_num_layer):
            logger.info("\t layer: {},  TxL: {} ---> {}".format(l, model.clip.visual._tome_info['tensor_shape'][l], model.clip.visual._tome_info['tensor_shape'][l+1]))

def compute_corr(args, model, test_dataloader, device, n_gpu):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
        

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)
    teacher_model = init_teacher_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    if args.freeze_clip:
        for name, param in model.clip.named_parameters():
            param.requires_grad = False
    
    # freeze teacher model:
    for name, param in teacher_model.named_parameters():
        param.requires_grad = False

    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']
        
        global_step = 0
        if args.enable_weime:
            logger.info("merging model parameters...")
            logger.info("Model parameters before merging {}".format(sum(p.numel() for p in model.parameters())))
            model = neuron_merge(args, model, args.weime_r_neuron, device, train_dataloader, args.weime_layers)
            model = attention_head_merge(args, model, args.weime_r_head, device, train_dataloader, args.weime_layers)
            if args.weime_pretrained_model:
                model_state_dict = torch.load(args.weime_pretrained_model, map_location='cpu')
                model.load_state_dict(model_state_dict, strict=False)
            optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
            logger.info("Model parameters after merging {}".format(sum(p.numel() for p in model.parameters())))
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            # if args.enable_weime:
            #     logger.info("merging model parameters...")
            #     logger.info("Model parameters before merging {}".format(sum(p.numel() for p in model.parameters())))
            #     model = neuron_merge(args, model, args.weime_r_neuron // args.epochs, device, train_dataloader, args.weime_layers)
            #     model = attention_head_merge(args, model, args.weime_r_head // args.epochs, device, train_dataloader, args.weime_layers)
            #     optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
            #     logger.info("Model parameters after merging {}".format(sum(p.numel() for p in model.parameters())))
                
            tr_loss, global_step = train_epoch(epoch, args, model, teacher_model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                ## Run on val dataset, this process is *TIME-consuming*.
                # logger.info("Eval on val dataset")
                # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)

                R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

        ## Uncomment if want to test on the best checkpoint
        # if args.local_rank == 0:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu)
            speed_test(args, model, test_dataloader, device, n_gpu)
        
    elif args.do_speed_test:
        if args.local_rank == 0:
            speed_test(args, model, test_dataloader, device, n_gpu)


if __name__ == "__main__":
    main()
