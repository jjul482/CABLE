'''
Adapter boilerplate: https://github.com/sun-hailong/CVPR24-Ease
'''
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath

import timm
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from timm.models.registry import register_model
import torch.nn.functional as F
import numpy as np

import logging
import os
from collections import OrderedDict
import torch
import copy


class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        # lora init
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x, adapters=None, gates=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        # Gated adapter combination
        if adapters is not None and gates is not None:
            adapt_outputs = []
            if adapters is not None and isinstance(adapters, nn.Module) and not isinstance(adapters, nn.ModuleList):
                adapters = [adapters]
            for i, adapt in enumerate(adapters):
                adapt_out = adapt(x, add_residual=False)
                adapt_outputs.append(adapt_out * gates[i])
            adapt_x = sum(adapt_outputs)
            if self.config.ffn_adapt:
                if self.config.ffn_option == 'sequential':
                    x = x + adapt_x
                elif self.config.ffn_option == 'parallel':
                    x = x + adapt_x
                else:
                    raise ValueError(self.config.ffn_adapt)
        elif adapters is not None:
            # Default: use first adapter
            adapt_x = adapters[0](x, add_residual=False)
            if self.config.ffn_adapt:
                if self.config.ffn_option == 'sequential':
                    x = adapters[0](x)
                elif self.config.ffn_option == 'parallel':
                    x = x + adapt_x
                else:
                    raise ValueError(self.config.ffn_adapt)

        x = residual + x
        return x



class VisionTransformer(nn.Module):
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=1024, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()

        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # self.init_weights(weight_init)

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        ######## Adapter begins #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)
        
        self.config = tuning_config
        self._device = tuning_config._device
        self.adapter_list = []
        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter()

    def set_adapter_gates(self, fisher_traces, temperature=3.0):
        # Set gating weights for adapters based on fisher traces.
        fisher_tensor = torch.tensor(fisher_traces, dtype=torch.float32)
        self.adapter_gates = F.softmax(fisher_tensor / temperature, dim=0).cpu().numpy()  # shape: [num_adapters]

    def compute_forgetting_score_adapters(self, forgetting_traces, inputs, targets, loss_fn, lr=1e-3):
        """
        Computes the forgetting score for each adapter:
        Forgetting_A = sum_i F_i^(A) * (theta_i - theta_i^t)^2
        - forgetting_traces: list of tensors, one per adapter, shape matches adapter parameters
        - inputs: input batch tensor
        - targets: target batch tensor
        - loss_fn: loss function to use
        - lr: learning rate for simulated update
        Returns: list of forgetting scores, one per adapter
        """
        self.eval()
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        targets = targets.long()

        # Get current parameters
        theta_list = []
        for adapter in self.cur_adapter:
            theta_list.append([p.detach().clone() for p in adapter.parameters()])

        # Simulate a gradient update (do not update actual model)
        # 1. Forward pass
        self.zero_grad()
        outputs = self.forward_train(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        # 2. Compute updated parameters for each adapter
        theta_t_list = []
        for adapter in self.cur_adapter:
            updated_params = []
            for p in adapter.parameters():
                if p.grad is not None:
                    updated_params.append(p.detach() - lr * p.grad.detach())
                else:
                    updated_params.append(p.detach().clone())
            theta_t_list.append(updated_params)

        # 3. Compute forgetting scores
        scores = []
        for F, theta, theta_t in zip(forgetting_traces, theta_list, theta_t_list):
            score = 0.0
            for f, p, p_t in zip(F, theta, theta_t):
                score += torch.sum(f * (p - p_t) ** 2).item()
            scores.append(score)
        return self.zscore_normalize(scores)
    
    def zscore_normalize(self, scores):
        scores = np.array(scores)
        mean = scores.mean()
        std = scores.std()
        if std > 0:
            return ((scores - mean) / std).tolist()
        else:
            return [0.0 for _ in scores]
    
    def sum_fisher_traces(self, traces):
        """
        Given a list of lists of tensors (traces) for each adapter,
        returns a list of floats, one per adapter (sum of all elements in all parameter traces).
        """
        sum_traces = []
        for adapter_traces in traces:
            total = 0.0
            for param_trace in adapter_traces:
                total += param_trace.sum().item()
            sum_traces.append(total)
        return sum_traces

    def compute_fisher_trace_adapters(self, dataloader, loss_fn, device=None, num_batches=10):
        """
        Approximates the Fisher Information trace for each adapter over a few batches.
        Returns a list of lists of tensors, one list per adapter, each tensor matches a parameter's shape.
        """
        device = device or self._device
        self.eval()
        # Initialize traces: list of lists of tensors (one list per adapter, one tensor per parameter)
        traces = [
            [torch.zeros_like(param) for param in adapter.parameters()]
            for adapter in self.cur_adapter
        ]
        for batch_idx, (inputs, _, targets) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.long()
            self.zero_grad()
            outputs = self.forward_train(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            for i, adapter in enumerate(self.cur_adapter):
                for j, param in enumerate(adapter.parameters()):
                    if param.grad is not None:
                        traces[i][j] += (param.grad.detach() ** 2)
        # Average over batches
        for i in range(len(traces)):
            for j in range(len(traces[i])):
                traces[i][j] /= num_batches
        self.set_adapter_gates([trace[0].sum().item() for trace in traces])  # Optionally gate by first param's sum
        return traces

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist           

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True
        
    def get_new_adapter(self):
        config = self.config
        self.cur_adapter = nn.ModuleList()
        if config.ffn_adapt:
            for i in range(len(self.blocks)):
                adapter = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num, d_model=self.embed_dim,
                                        init_option=config.ffn_adapter_init_option,
                                        adapter_scalar=config.ffn_adapter_scalar,
                                        adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                        ).to(self._device)
                self.cur_adapter.append(adapter)
            self.cur_adapter.requires_grad_(True)
        else:
            print("====Not use adapter===")

    def add_adapter_to_list(self):
        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()
    
    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if self.config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            # Gated adapter logic
            if hasattr(self, "adapter_gates"):
                x = blk(x, self.cur_adapter, self.adapter_gates)
            else:
                x = blk(x, self.cur_adapter)
            if self.config.vpt_on:
                x = x[:, self.config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_test(self, x, use_init_ptm=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)
        
        features = []
        
        if use_init_ptm:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            features.append(x)
        
        for i in range(len(self.adapter_list)):
            x = copy.deepcopy(x_init)
            for j in range(len(self.blocks)):
                adapt = self.adapter_list[i][j]
                x = self.blocks[j](x, adapt, self.adapter_gates)
            x = self.norm(x)
            features.append(x)
        
        x = copy.deepcopy(x_init)
        for i in range(len(self.blocks)):
            adapt = self.cur_adapter[i]
            x = self.blocks[i](x, adapt, self.adapter_gates)
        x = self.norm(x)
        features.append(x)
        
        return features

    def forward(self, x, test=False, use_init_ptm=False):
        if not test:
            output = self.forward_train(x)
        else:
            features = self.forward_test(x, use_init_ptm)
            output = torch.Tensor().to(features[0].device)
            for x in features:
                cls = x[:, 0, :]
                output = torch.cat((
                    output,
                    cls
                ), dim=1)

        return output

    def forward_proto(self, x, adapt_index):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)
        
        # the init_PTM's feature
        if adapt_index == -1:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            output = x[:, 0, :]
            return output

        i = adapt_index
        x = copy.deepcopy(x_init)
        for j in range(len(self.blocks)):
            if i < len(self.adapter_list):
                adapt = self.adapter_list[i][j]
            else:
                adapt = self.cur_adapter[j]
            x = self.blocks[j](x, adapt, self.adapter_gates)
        x = self.norm(x)
        output = x[:, 0, :]
        
        return output
        
def vit_base_patch16_224_cable(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model=timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model

def vit_large_patch14_clip_224_dfn2b_s39b(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # checkpoint_model = torch.load('./pretrained_models/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz')
    checkpoint_model=timm.create_model("vit_large_patch14_clip_224.dfn2b_s39b", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:1024]
            k_weight = qkv_weight[1024:1024*2]
            v_weight = qkv_weight[1024*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:1024]
            k_bias = qkv_bias[1024:1024*2]
            v_bias = qkv_bias[1024*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model

def resnet50d_ra4_e3600_r224_in1k(pretrained=False, **kwargs):
    
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    checkpoint_model=timm.create_model("resnet50d.ra4_e3600_r224_in1k", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False 
    return model

