import torch
import torch.nn as nn
from tqdm import tqdm


def batch_cosine_similarity(x, type='neuron'):
    if type == 'neuron':
        x = x.transpose(1, 2)
    # x shape: (b, d, l)
    # 转置为 (b, l, d)
    else:
        x = x.transpose(1, 2) # shape: (b, l, d)
        B, L, D = x.shape
        x = x.view(B, -1, 64, D).sum(dim=2)
    
    # print(x.shape)
    # 计算 L2 范数
    x_norm = x / torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-7)  # shape: (b, l, 1)
    a = x_norm[..., ::2, :]
    b = x_norm[..., 1::2, :]
    # 计算点积
    scores = a @ b.transpose(1, 2)  # shape: (b, l, l)

    return scores

def merge(x, r, metric=None, scores=None, mode='mean'):
    with torch.no_grad():
        if scores is None:
            if metric is None:
                metric = x
            metric = metric / metric.norm(dim=-1, keepdim=True) # B,L,C
            a, b = metric[..., ::2, :], metric[..., 1::2, :] # B,L//2,C
            scores = a @ b.transpose(-1, -2) # B,L//2,L//2
        scores = 2 * torch.rand(scores.shape).to(scores.device) - 1
        node_max, node_idx = scores.max(dim=-1) # B,L//2
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # B,L//2,1

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens, B,L//2 - r, 1
        src_idx = edge_idx[..., :r, :]  # Merged Tokens, B,r,1
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx) # B,r,1
        
        # print(x.shape, scores.shape)
        x = x.transpose(0, 1)
        src, dst = x[::2], x[1::2] # L//2,B,C
        t1, n, c = src.shape
        unm = src.gather(dim=0, index=unm_idx.transpose(0, 1).expand(t1 - r, n, c))
        src = src.gather(dim=0, index=src_idx.transpose(0, 1).expand(r, n, c))
        # dst = dst.scatter_reduce(0, dst_idx.transpose(0, 1).expand(r, n, c), src, reduce=mode)

        

        return torch.cat([unm, dst], dim=0)


def neuron_merge(args, model, r, device, train_dataloader, merge_layer_idx=None):
    def activation_hook(name):
        def hook(model, input, output):
            activations_dict[name] = input[0]
        return hook
    if r == 0:
        return model
    if merge_layer_idx is None:
        merge_layer_idx = list(range(12))
    model.eval()
    with torch.no_grad():
        params = {idx: [] for idx in merge_layer_idx}
        n = 0
        activations_dict = {}
        scores_sum = [None] * len(merge_layer_idx)
        if hasattr(model, 'module'):
            for layer_idx in merge_layer_idx:
                model.module.clip.visual.transformer.resblocks[layer_idx].mlp.c_proj.register_forward_hook(activation_hook(layer_idx))

            for batch in tqdm(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, video, video_mask = batch
                _ = model.module.get_visual_output(video, video_mask, frame_selection=args.frame_selection)
                for i, layer_idx in enumerate(merge_layer_idx):
                    # print(activations_dict[layer_idx].shape)
                    feature = model.module.clip.visual.transformer.resblocks[layer_idx].mlp.c_proj.weight[None, :, :] * activations_dict[layer_idx][:, 0][:, None, :]
                    # print(feature.shape)
                    if scores_sum[i] is None:
                        scores_sum[i] = batch_cosine_similarity(feature).sum(dim=0, keepdim=True)
                    else:
                        scores_sum[i] += batch_cosine_similarity(feature).sum(dim=0, keepdim=True)

                n += video.shape[0]
                if n >= args.max_samples:
                    break
            for i in range(len(scores_sum)):
                scores_sum[i] /= n
            scores = torch.cat(scores_sum, dim=0)
                
            for name, param in model.named_parameters():
                if 'mlp' in name:
                    l = name.split('.')
                    if l[2] == 'visual':
                        layer_idx = int(l[5])
                        if layer_idx in merge_layer_idx:
                            sub_idx = l[7]
                            param_type = l[8]
                            if sub_idx == 'c_fc' and param_type == 'weight':
                                params[layer_idx].append(param)
                            elif sub_idx == 'c_fc' and param_type == 'bias':
                                params[layer_idx].append(param.unsqueeze(1))
                            elif sub_idx == 'c_proj' and param_type == 'weight':
                                params[layer_idx].append(param.t())
                            
            all_neurons = []
            for p in params.values():
                all_neurons.append(torch.cat(p, dim=1).unsqueeze(0))
            
            all_neurons = torch.cat(all_neurons)
            merged_weight = merge(all_neurons, r, scores=scores).transpose(0, 1)
            # print(merged_weight.shape)
            for i, layer_idx in enumerate(merge_layer_idx):
                weight1 = merged_weight[i, :, :768]
                bias1 = merged_weight[i, :, 768:769].squeeze(1)
                weight2 = merged_weight[i, :, 769:].t()
                # print(weight2.shape)
            
                new_fc1 = nn.Linear(768, 3072-r)
                new_fc2 = nn.Linear(3072-r, 768)
                new_fc1.weight = nn.Parameter(weight1)
                new_fc1.bias = nn.Parameter(bias1)
                new_fc2.weight = nn.Parameter(weight2)
                new_fc2.bias = model.module.clip.visual.transformer.resblocks[layer_idx].mlp.c_proj.bias
                model.module.clip.visual.transformer.resblocks[layer_idx].mlp.c_fc = new_fc1
                model.module.clip.visual.transformer.resblocks[layer_idx].mlp.c_proj = new_fc2
        else:

            for layer_idx in merge_layer_idx:
                model.clip.visual.transformer.resblocks[layer_idx].mlp.c_proj.register_forward_hook(activation_hook(layer_idx))

            for batch in tqdm(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, video, video_mask = batch
                _ = model.get_visual_output(video, video_mask, frame_selection=args.frame_selection)
                for i, layer_idx in enumerate(merge_layer_idx):
                    # print(activations_dict[layer_idx].shape)
                    feature = model.clip.visual.transformer.resblocks[layer_idx].mlp.c_proj.weight[None, :, :] * activations_dict[layer_idx][:, 0][:, None, :]
                    # print(feature.shape)
                    if scores_sum[i] is None:
                        scores_sum[i] = batch_cosine_similarity(feature).sum(dim=0, keepdim=True)
                    else:
                        scores_sum[i] += batch_cosine_similarity(feature).sum(dim=0, keepdim=True)

                n += video.shape[0]
                if n >= args.max_samples:
                    break
            for i in range(len(scores_sum)):
                scores_sum[i] /= n
            scores = torch.cat(scores_sum, dim=0)
                
            for name, param in model.named_parameters():
                if 'mlp' in name:
                    l = name.split('.')
                    if l[1] == 'visual':
                        layer_idx = int(l[4])
                        if layer_idx in merge_layer_idx:
                            sub_idx = l[6]
                            param_type = l[7]
                            if sub_idx == 'c_fc' and param_type == 'weight':
                                params[layer_idx].append(param)
                            elif sub_idx == 'c_fc' and param_type == 'bias':
                                params[layer_idx].append(param.unsqueeze(1))
                            elif sub_idx == 'c_proj' and param_type == 'weight':
                                params[layer_idx].append(param.t())
                            
            all_neurons = []
            for p in params.values():
                all_neurons.append(torch.cat(p, dim=1).unsqueeze(0))
            
            all_neurons = torch.cat(all_neurons)
            merged_weight = merge(all_neurons, r, scores=scores).transpose(0, 1)
            # print(merged_weight.shape)
            for i, layer_idx in enumerate(merge_layer_idx):
                weight1 = merged_weight[i, :, :768]
                bias1 = merged_weight[i, :, 768:769].squeeze(1)
                weight2 = merged_weight[i, :, 769:].t()
                # print(weight2.shape)
            
                new_fc1 = nn.Linear(768, 3072-r)
                new_fc2 = nn.Linear(3072-r, 768)
                new_fc1.weight = nn.Parameter(weight1)
                new_fc1.bias = nn.Parameter(bias1)
                new_fc2.weight = nn.Parameter(weight2)
                new_fc2.bias = model.clip.visual.transformer.resblocks[layer_idx].mlp.c_proj.bias
                model.clip.visual.transformer.resblocks[layer_idx].mlp.c_fc = new_fc1
                model.clip.visual.transformer.resblocks[layer_idx].mlp.c_proj = new_fc2
    return model


def attention_head_merge(args, model, r, device, train_dataloader, merge_layer_idx=None):
    def activation_hook(name):
        def hook(model, input, output):
            activations_dict[name] = input[0]
        return hook
    if r == 0:
        return model
    if merge_layer_idx is None:
        merge_layer_idx = list(range(12))
    model.eval()
    with torch.no_grad():
        params = {idx: [] for idx in merge_layer_idx}
        n = 0
        activations_dict = {}
        scores_sum = [None] * len(merge_layer_idx)
        
        if hasattr(model, 'module'):
            for layer_idx in merge_layer_idx:
                model.module.clip.visual.transformer.resblocks[layer_idx].attn.out_proj.register_forward_hook(activation_hook(layer_idx))

            for batch in tqdm(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, video, video_mask = batch
                _ = model.module.get_visual_output(video, video_mask, frame_selection=args.frame_selection)

                for i, layer_idx in enumerate(merge_layer_idx):
                    feature = model.module.clip.visual.transformer.resblocks[layer_idx].attn.out_proj.weight[None, :, :] * activations_dict[layer_idx].view(197, -1, model.module.clip.visual.transformer.resblocks[layer_idx].attn.out_proj.weight.shape[-1])[:, 0][:, None, :]
                    if scores_sum[i] is None:
                        scores_sum[i] = batch_cosine_similarity(feature, type='head').sum(dim=0, keepdim=True)
                    else:
                        scores_sum[i] += batch_cosine_similarity(feature, type='head').sum(dim=0, keepdim=True)

                n += video.shape[0]
                if n >= args.max_samples:
                    break
            for i in range(len(scores_sum)):
                scores_sum[i] /= n
            scores = torch.cat(scores_sum, dim=0)
            # print(scores.shape)
        
            num_heads = model.module.clip.visual.transformer.resblocks[0].attn.num_heads
            new_num_heads = num_heads - r
            # d_model = model.module.visual.transformer.resblocks[0].attn.embed_dim
            
            # for in_proj_weights & bias
            in_params = {idx: {'q': [], 'k': [], 'v':[]} for idx in merge_layer_idx}
            out_params = {}
            for name, param in model.named_parameters():
                if 'attn' in name:
                    l = name.split('.')
                    if l[2] == 'visual':
                        layer_idx = int(l[5])
                        if layer_idx in merge_layer_idx:
                            if l[7] == 'in_proj_weight':
                                in_params[layer_idx]['q'].append(param[0:num_heads*64, :].view(num_heads, -1))
                                in_params[layer_idx]['k'].append(param[num_heads*64:2*num_heads*64, :].view(num_heads, -1))
                                in_params[layer_idx]['v'].append(param[2*num_heads*64:3*num_heads*64, :].view(num_heads, -1))
                            if l[7] == 'in_proj_bias':
                                in_params[layer_idx]['q'].append(param[0:num_heads*64].view(num_heads, -1))
                                in_params[layer_idx]['k'].append(param[num_heads*64:2*num_heads*64].view(num_heads, -1))
                                in_params[layer_idx]['v'].append(param[2*num_heads*64:3*num_heads*64].view(num_heads, -1))
                            if l[7] == 'out_proj' and l[8] == 'weight':
                                out_params[layer_idx] = param.t().reshape(num_heads, -1)
        
            merged_in_weights = {idx: [] for idx in merge_layer_idx}     
            merged_in_bias = {idx: [] for idx in merge_layer_idx}      
            for type in ['q', 'k', 'v']:               
                all_heads = []
                for p in in_params.values():
                    all_heads.append(torch.cat(p[type], dim=1).unsqueeze(0))
            
                all_heads = torch.cat(all_heads)
                # print(all_heads.shape, scores.shape)
                merged_weight = merge(all_heads, r, scores=scores).transpose(0, 1)
                # print(merged_weight.shape)
                for i, layer_idx in enumerate(merge_layer_idx):
                    merged_in_weights[layer_idx].append(merged_weight[i, :, :64*768].reshape(new_num_heads*64, 768))
                    merged_in_bias[layer_idx].append(merged_weight[i, :, 64*768:].squeeze(1).flatten())

                    # print(weight2.shape)
                    
                
            merged_out_weights = {}    
            all_heads = []
            for p in out_params.values():
                all_heads.append(p.unsqueeze(0))
                
            all_heads = torch.cat(all_heads)
            merged_weight = merge(all_heads, r, scores=scores).transpose(0, 1)
            for i, layer_idx in enumerate(merge_layer_idx):
                merged_out_weights[layer_idx]= merged_weight[i, :, :64*768].reshape(new_num_heads*64, 768).t()


            for layer_idx in merge_layer_idx:
                new_multiheadattention = nn.MultiheadAttention(768, new_num_heads)
                new_multiheadattention.head_dim = 64
                new_multiheadattention.in_proj_weight = nn.Parameter(torch.cat(merged_in_weights[layer_idx], dim=0))
                new_multiheadattention.in_proj_bias = nn.Parameter(torch.cat(merged_in_bias[layer_idx], dim=0))
                new_multiheadattention.out_proj.weight = nn.Parameter(merged_out_weights[layer_idx])
                new_multiheadattention.out_proj.bias = model.module.clip.visual.transformer.resblocks[layer_idx].attn.out_proj.bias
                model.module.clip.visual.transformer.resblocks[layer_idx].attn = new_multiheadattention

        else:
            
            for layer_idx in merge_layer_idx:
                model.clip.visual.transformer.resblocks[layer_idx].attn.out_proj.register_forward_hook(activation_hook(layer_idx))

            for batch in tqdm(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, video, video_mask = batch
                _ = model.get_visual_output(video, video_mask, frame_selection=args.frame_selection)

                for i, layer_idx in enumerate(merge_layer_idx):
                    feature = model.clip.visual.transformer.resblocks[layer_idx].attn.out_proj.weight[None, :, :] * activations_dict[layer_idx].view(197, -1, model.clip.visual.transformer.resblocks[layer_idx].attn.out_proj.weight.shape[-1])[:, 0][:, None, :]
                    if scores_sum[i] is None:
                        scores_sum[i] = batch_cosine_similarity(feature, type='head').sum(dim=0, keepdim=True)
                    else:
                        scores_sum[i] += batch_cosine_similarity(feature, type='head').sum(dim=0, keepdim=True)

                n += video.shape[0]
                if n >= args.max_samples:
                    break
            for i in range(len(scores_sum)):
                scores_sum[i] /= n
            scores = torch.cat(scores_sum, dim=0)
            # print(scores.shape)
        
            num_heads = model.clip.visual.transformer.resblocks[0].attn.num_heads
            new_num_heads = num_heads - r
            # d_model = model.module.visual.transformer.resblocks[0].attn.embed_dim
            
            # for in_proj_weights & bias
            in_params = {idx: {'q': [], 'k': [], 'v':[]} for idx in merge_layer_idx}
            out_params = {}
            for name, param in model.named_parameters():
                if 'attn' in name:
                    l = name.split('.')
                    if l[1] == 'visual':
                        layer_idx = int(l[4])
                        if layer_idx in merge_layer_idx:
                            if l[6] == 'in_proj_weight':
                                in_params[layer_idx]['q'].append(param[0:num_heads*64, :].view(num_heads, -1))
                                in_params[layer_idx]['k'].append(param[num_heads*64:2*num_heads*64, :].view(num_heads, -1))
                                in_params[layer_idx]['v'].append(param[2*num_heads*64:3*num_heads*64, :].view(num_heads, -1))
                            if l[6] == 'in_proj_bias':
                                in_params[layer_idx]['q'].append(param[0:num_heads*64].view(num_heads, -1))
                                in_params[layer_idx]['k'].append(param[num_heads*64:2*num_heads*64].view(num_heads, -1))
                                in_params[layer_idx]['v'].append(param[2*num_heads*64:3*num_heads*64].view(num_heads, -1))
                            if l[6] == 'out_proj' and l[7] == 'weight':
                                out_params[layer_idx] = param.t().reshape(num_heads, -1)
        
            merged_in_weights = {idx: [] for idx in merge_layer_idx}     
            merged_in_bias = {idx: [] for idx in merge_layer_idx}      
            for type in ['q', 'k', 'v']:               
                all_heads = []
                for p in in_params.values():
                    all_heads.append(torch.cat(p[type], dim=1).unsqueeze(0))
            
                all_heads = torch.cat(all_heads)
                # print(all_heads.shape, scores.shape)
                merged_weight = merge(all_heads, r, scores=scores).transpose(0, 1)
                # print(merged_weight.shape)
                for i, layer_idx in enumerate(merge_layer_idx):
                    merged_in_weights[layer_idx].append(merged_weight[i, :, :64*768].reshape(new_num_heads*64, 768))
                    merged_in_bias[layer_idx].append(merged_weight[i, :, 64*768:].squeeze(1).flatten())

                    # print(weight2.shape)
                    
                
            merged_out_weights = {}    
            all_heads = []
            for p in out_params.values():
                all_heads.append(p.unsqueeze(0))
                
            all_heads = torch.cat(all_heads)
            merged_weight = merge(all_heads, r, scores=scores).transpose(0, 1)
            for i, layer_idx in enumerate(merge_layer_idx):
                merged_out_weights[layer_idx]= merged_weight[i, :, :64*768].reshape(new_num_heads*64, 768).t()

            for layer_idx in merge_layer_idx:
                new_multiheadattention = nn.MultiheadAttention(768, new_num_heads)
                new_multiheadattention.head_dim = 64
                new_multiheadattention.in_proj_weight = nn.Parameter(torch.cat(merged_in_weights[layer_idx], dim=0))
                new_multiheadattention.in_proj_bias = nn.Parameter(torch.cat(merged_in_bias[layer_idx], dim=0))
                new_multiheadattention.out_proj.weight = nn.Parameter(merged_out_weights[layer_idx])
                new_multiheadattention.out_proj.bias = model.clip.visual.transformer.resblocks[layer_idx].attn.out_proj.bias
                model.clip.visual.transformer.resblocks[layer_idx].attn = new_multiheadattention
    return model