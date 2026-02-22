#文件路径: src/models/newmodel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import os
import torch.distributed as dist
from src.models.components.VPT_pool import GumbelPromptPool
from src.models.components.Classprompt import PrototypePromptGenerator
from src.models.components.PCPvisionprompt import PCPVisualPromptGenerator
from src.models.components.shareprompt_new import SharedPromptNetwork
from src.models.components.PCPtextprompt import LightweightMetaNet
from models.components.residual_adapter import ClipAdapter
from src.utils.loss import InternalLossWeighter

class PrototypeGuidedCLIP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.model.num_classes

        
        # (Full Model)
        self.use_adapter = True       #  Adapter
        self.use_shared_space = True  #  Shared Prompt Network
        self.use_hierarchical = True  #  Texture/Global Pools
        self.use_prototypes = True    # Prototype 
        # ==========================================
        self.prompt_length = config.model.prompt.prompt_len
        self.pool_size = config.model.prompt.prompt_pool_size
        self.top_k = config.model.prompt.top_k
        self.clip_model_name = config.model.original_clip_name
        # ===  CLIP ===
        self.original_clip, _ = clip.load(self.clip_model_name, device='cpu', jit=False)
        self.vision_layers = len(self.original_clip.visual.transformer.resblocks)
        self.text_layers = len(self.original_clip.transformer.resblocks)
        
        self.visual_width = self.original_clip.visual.conv1.out_channels
        self.text_width = self.original_clip.transformer.width 
        
        self.original_clip.float() 

        # =================================================================================
        # 
        # =================================================================================
        
        # 
        # Texture Start
        self.v_idx_texture = self.vision_layers//3
        if self.v_idx_texture < 0: self.v_idx_texture = 0
        
        # Adapter Insert
        self.v_idx_adapter_insert_at = (2 * self.vision_layers) // 3 - 1
        
        # Global Start
        self.v_idx_global = self.v_idx_adapter_insert_at + 1 

        
        self.t_idx_p1 = self.text_layers // 3
        if self.t_idx_p1 < 0: self.t_idx_p1 = 0
        self.t_idx_adapter = (2 * self.text_layers) // 3 - 1
        self.t_idx_p2 = self.t_idx_adapter + 1

       
        self.v_layers_to_inject = [0] + \
                                list(range(self.v_idx_texture, self.v_idx_adapter_insert_at + 1)) + \
                                list(range(self.v_idx_global, self.vision_layers))
                                
        self.t_layers_to_inject = [0] + \
                                list(range(self.t_idx_p1, self.t_idx_adapter + 1)) + \
                                list(range(self.t_idx_p2, self.text_layers))

        print(f"Visual Injection Layers: {self.v_layers_to_inject}")
        print(f"Text Injection Layers: {self.t_layers_to_inject}")

        
        self.register_buffer("prototype_centers", self._load_prototype_centers(config.model.prototype_centers_path))
        proto_dim = self.prototype_centers.shape[-1]

                
        self.v_layer_mapping = {}
        
        
        self.v_layer_mapping[0] = 'input_stage'
        
        
        for i in range(self.v_idx_texture, self.v_idx_adapter_insert_at + 1):
            self.v_layer_mapping[i] = 'texture_stage'
            
        
        for i in range(self.v_idx_global, self.vision_layers):
            self.v_layer_mapping[i] = 'global_stage'

        
        self.t_layer_mapping = {}
        
       
        self.t_layer_mapping[0] = 'start_stage'
        
        
        for i in range(self.t_idx_p1, self.t_idx_adapter + 1):
            self.t_layer_mapping[i] = 'p1_stage'
            
        
        for i in range(self.t_idx_p2, self.text_layers):
            self.t_layer_mapping[i] = 'p2_stage'

        print(f"Visual Sharing Groups: {self.v_layer_mapping}")
        print(f"Text Sharing Groups: {self.t_layer_mapping}")

        
        self.register_buffer("prototype_centers", self._load_prototype_centers(config.model.prototype_centers_path))
        proto_dim = self.prototype_centers.shape[-1]

        
        self.shared_net = SharedPromptNetwork(
            shared_dim=1024, 
            visual_dim=self.visual_width, 
            text_dim=self.text_width,                 
            prompt_len=self.prompt_length,
            
            visual_layer_mapping=self.v_layer_mapping,
            text_layer_mapping=self.t_layer_mapping,
            dropout=0.1  
        )
        
        
        # self.shared_net = SharedPromptNetwork(
        #     shared_dim=512, 
        #     visual_dim=self.visual_width, 
        #     text_dim=self.text_width,                 
        #     prompt_len=self.prompt_length,
        #     visual_layers_to_inject=self.v_layers_to_inject,
        #     text_layers_to_inject=self.t_layers_to_inject
        # )
        
        
        self.text_private_prompts = nn.Parameter(torch.empty(3, self.prompt_length, self.text_width))
        self.text_generator = LightweightMetaNet(input_dim=self.text_width, output_dim=self.text_width, prompt_len=self.prompt_length, hidden_dim=64)
        
        
        self.vision_prompt_generator_input = PCPVisualPromptGenerator(
            visual_dim=self.visual_width, proto_dim=proto_dim, prompt_len=self.prompt_length
        )
        self.texture_prompt_pool = GumbelPromptPool(
            pool_size=self.pool_size, prompt_length=self.prompt_length, embed_dim=self.visual_width, top_k=self.top_k, embedding_key='cls'
        )
        self.global_prompt_pool = GumbelPromptPool(
            pool_size=self.pool_size, prompt_length=self.prompt_length, embed_dim=self.visual_width, top_k=self.top_k, embedding_key='cls'
        )
        
        
        self.random_input_prompt = nn.Parameter(torch.randn(self.prompt_length, self.visual_width))
        nn.init.normal_(self.random_input_prompt, std=0.02)
        


        
        self.mid_adapter_vision = ClipAdapter(self.visual_width, bottleneck_dim=self.visual_width//2)
        self.final_adapter_vision = ClipAdapter(self.visual_width, bottleneck_dim=self.visual_width//2)
        self.mid_adapter_text = ClipAdapter(self.text_width, bottleneck_dim=self.text_width//2)
        self.final_adapter_text = ClipAdapter(self.text_width, bottleneck_dim=self.text_width//2)

        
        self.gate_input = nn.Parameter(torch.ones(1) * 0.5)   
        self.gate_texture = nn.Parameter(torch.ones(1) * 0.5) 
        self.gate_global = nn.Parameter(torch.ones(1) * 0.5)  

        # 6. Freeze CLIP
        for param in self.original_clip.parameters():
            param.requires_grad = False
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_weighter = InternalLossWeighter(num_losses=3)

        self._init_positional_embeddings()
        self._init_prompt_text_weights()

    def _init_prompt_text_weights(self):
        nn.init.normal_(self.text_private_prompts, std=0.02)

    def _load_prototype_centers(self, path):
        if not os.path.exists(path):
            print(f"Warning: Prototype centers not found at {path}. Using random init.")
            return torch.randn(self.num_classes, self.visual_width)
        centers = np.load(path)
        return torch.from_numpy(centers).float()

    def _get_text_context(self, text):
        with torch.no_grad():
            x = self.original_clip.token_embedding(text).type(self.original_clip.dtype)
            x = x + self.original_clip.positional_embedding.type(self.original_clip.dtype)
            x = x.permute(1, 0, 2)
            x = self.original_clip.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.original_clip.ln_final(x).type(self.original_clip.dtype)
            eot_indices = text.argmax(dim=-1)
            context = x[torch.arange(x.shape[0]), eot_indices]
            if self.original_clip.text_projection is not None:
                context = context @ self.original_clip.text_projection
            context = context.detach()
        return context

    def _init_positional_embeddings(self):
        original_vit = self.original_clip.visual
        hidden_dim = original_vit.conv1.out_channels
        original_pos_embed = original_vit.positional_embedding
        extra_tokens = self.prompt_length
        new_seq_len = original_pos_embed.shape[0] + extra_tokens
        self.pos_embed = nn.Parameter(torch.zeros(new_seq_len, hidden_dim))
        self.pos_embed.data[0:1, :] = original_pos_embed.data[0:1, :] 
        self.pos_embed.data[1+extra_tokens:, :] = original_pos_embed.data[1:, :]
        nn.init.normal_(self.pos_embed.data[1:1+extra_tokens, :], std=0.02)

    def get_prototype_class_ids(self, images):
        with torch.no_grad():
            features = self.original_clip.encode_image(images)
            features = F.normalize(features, dim=-1)
            centers = F.normalize(self.prototype_centers, dim=-1)
            similarities = torch.matmul(features, centers.t())
            class_ids = torch.argmax(similarities, dim=1)
        return class_ids

    def forward(self, batch):
        images = batch["images"]
        text_tokens = batch.get("text_tokens")
        batch_size = images.shape[0]
        device = images.device
        dtype = self.original_clip.dtype
        # 1.  (Teacher)
        with torch.no_grad():
            teacher_image_features = self.original_clip.encode_image(images)
            teacher_image_features = F.normalize(teacher_image_features, dim=-1).detach()
            if text_tokens is not None:
                teacher_text_features = self.original_clip.encode_text(text_tokens)
                teacher_text_features = F.normalize(teacher_text_features, dim=-1).detach()

        
        if self.use_shared_space:
            v_shared_prompts, t_shared_prompts = self.shared_net(batch_size)
        else:
            
            v_shared_prompts = {
                k: torch.zeros(self.prompt_length, batch_size, self.visual_width, device=device, dtype=dtype) 
                for k in self.v_layers_to_inject
            }
            t_shared_prompts = {
                k: torch.zeros(self.prompt_length, batch_size, self.text_width, device=device, dtype=dtype) 
                for k in self.t_layers_to_inject
            }
        # ====================================================
        # v_shared_prompts={}
        # t_shared_prompts={}
        
        class_ids = self.get_prototype_class_ids(images)
        image_features, texture_loss, global_loss = self.encode_image_with_hierarchical_prompts(
            images, class_ids, v_shared_prompts
        )
        
        
        if text_tokens is None:
            return {'image_features': image_features, 'class_ids': class_ids}
        
        
        text_context = self._get_text_context(text_tokens) 
        dynamic_bias = self.text_generator(text_context.float()) 
        text_features = self.encode_text_hierarchical(text_tokens, t_shared_prompts, dynamic_bias)
        text_features = F.normalize(text_features, dim=-1)
        
        # 5. Logits
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        
        if dist.is_available() and dist.is_initialized():
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
            logits_per_image = logit_scale * image_features @ all_text_features.t()
            logits_per_text = logit_scale * text_features @ all_image_features.t()
            rank = dist.get_rank()
            labels = torch.arange(batch_size, device=images.device) + rank * batch_size
        else:
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(images.shape[0], device=images.device)

        # 6. Losses
        loss_ce = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        
        if dist.is_available() and dist.is_initialized():
            loss_hard = (self.compute_hard_negative_loss(image_features, all_text_features, labels) + 
                        self.compute_hard_negative_loss(text_features, all_image_features, labels)) / 2
        else:
            loss_hard = (self.compute_hard_negative_loss(image_features, text_features, labels) + 
                        self.compute_hard_negative_loss(text_features, image_features, labels)) / 2

        loss_distill = (1 - torch.mean(F.cosine_similarity(image_features, teacher_image_features, dim=1)) +
                        1 - torch.mean(F.cosine_similarity(text_features, teacher_text_features, dim=1))) / 2
        # loss_distill = torch.tensor(0.0, device=device, dtype=dtype)
        losses = [loss_ce, loss_distill, loss_hard]
        total_loss = self.loss_weighter(losses)
        # loss = loss_ce + 0.5 * loss_hard + 2.0 * loss_distill

        return {
            'loss': total_loss,
            'aux_loss': texture_loss + global_loss,
            'image_features': image_features,
            'text_features': text_features,
            'logits_per_image': logits_per_image,
            'class_ids': class_ids
        }

    def compute_hard_negative_loss(self, query_features, key_features, labels, margin=0.2):
        scores = query_features @ key_features.t()
        pos_scores = scores.gather(1, labels.view(-1, 1)) 
        cost = (margin + scores - pos_scores).clamp(min=0)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, labels.view(-1, 1), True)
        cost = cost.masked_fill(mask, 0)
        loss = cost.max(1)[0].mean()
        return loss

    def encode_image_with_hierarchical_prompts(self, images: torch.Tensor, class_ids: torch.Tensor, v_shared_prompts):
        
        batch_size = images.shape[0]
        vit = self.original_clip.visual
        dtype = self.original_clip.dtype
        device = images.device

        # === 1. Input Stage ===
        x = vit.conv1(images.type(dtype)) 
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) 
        #  3. Prototypes ===

        if self.use_prototypes:
            
            image_global_feat = x.mean(dim=1)
            selected_prototypes = self.prototype_centers[class_ids].to(dtype)
            class_prompts = self.vision_prompt_generator_input(image_global_feat, selected_prototypes)
            class_prompts = class_prompts * self.gate_input 
        else:
            
            # class_prompts = self.random_input_prompt.unsqueeze(0).repeat(batch_size, 1, 1).to(dtype)
            class_prompts = torch.zeros(batch_size, self.prompt_length, self.visual_width, device=device, dtype=dtype)
        # ========================================
        # image_global_feat = x.mean(dim=1)
        # selected_prototypes = self.prototype_centers[class_ids].to(dtype)
        # class_prompts = self.vision_prompt_generator_input(image_global_feat, selected_prototypes)
        # class_prompts = class_prompts * self.gate_input 
        
        shared_p0 = v_shared_prompts[0].to(dtype)
        prompt_input = class_prompts.permute(1, 0, 2) + shared_p0 
        len_input = prompt_input.shape[0]
        
        class_token = vit.class_embedding.to(dtype) + torch.zeros(batch_size, 1, x.shape[-1], dtype=dtype, device=device)
        #  Input Prompt: [CLS, Input, Patches]
        x = torch.cat([class_token, prompt_input.permute(1,0,2), x], dim=1) 
        
        x = x + self.pos_embed.to(dtype).to(device) 
        x = vit.ln_pre(x)
        x = x.permute(1, 0, 2) 

        blocks = vit.transformer.resblocks
        cached_texture_base = None
        len_texture = 0
        cached_global_base = None
        len_global = 0
        
        texture_loss = torch.tensor(0.0, device=device, dtype=dtype)
        global_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for i, block in enumerate(blocks):
           
            current_shared_p = v_shared_prompts.get(i)
            if current_shared_p is not None:
                current_shared_p = current_shared_p.to(dtype)

            
            if i == self.v_idx_texture:
                
                
                if self.use_hierarchical:
                    x_nld = x.permute(1, 0, 2)
                    cls_features = x_nld[:, 0, :]
                    texture_out = self.texture_prompt_pool(x_nld, cls_features)
                    texture_prompts = texture_out['batched_prompt'].to(dtype).permute(1, 0, 2)
                    texture_prompts = texture_prompts * self.gate_texture
                    # texture_loss = texture_out['orth_loss']
                else:
                    
                    texture_prompts = torch.zeros(self.prompt_length, batch_size, self.visual_width, device=device, dtype=dtype)
                # =====================================================
                # x_nld = x.permute(1, 0, 2)
                # cls_features = x_nld[:, 0, :]
                # texture_out = self.texture_prompt_pool(x_nld, cls_features)
                # texture_prompts = texture_out['batched_prompt'].to(dtype).permute(1, 0, 2)
                # texture_prompts = texture_prompts * self.gate_texture
                
                
                cached_texture_base = texture_prompts
                
                final_texture_prompt = texture_prompts
                if current_shared_p is not None:
                    if self.use_hierarchical and self.top_k > 1:
                        current_shared_p_expanded = current_shared_p.repeat(self.top_k, 1, 1)
                        final_texture_prompt = final_texture_prompt + current_shared_p_expanded
                    else:
                        
                        final_texture_prompt = final_texture_prompt + current_shared_p
                    # if self.top_k > 1 and self.use_hierarchical: 
                    #      
                    #      
                    #     pass
                    # elif self.top_k > 1: current_shared_p = current_shared_p.repeat(self.top_k, 1, 1)
                    # final_texture_prompt = final_texture_prompt + current_shared_p
                
                len_texture = final_texture_prompt.shape[0]
                
                
                x = torch.cat([x[:1], final_texture_prompt, x[1+len_input:]], dim=0)
            
            elif i > self.v_idx_texture and i <= self.v_idx_adapter_insert_at:
                # Deep Prompting
                if current_shared_p is not None:
                    if self.top_k > 1 and self.use_hierarchical: 
                        current_shared_p = current_shared_p.repeat(self.top_k, 1, 1)
                        layer_specific_texture = cached_texture_base + current_shared_p
                    else:
                        layer_specific_texture = cached_texture_base + current_shared_p
                else:
                    layer_specific_texture = cached_texture_base
                x[1 : 1 + len_texture] = layer_specific_texture

            # === Stage 3: Adapter & Handover ===
            if i == self.v_idx_adapter_insert_at:
                x = block(x)
               
                
                if self.use_adapter:
                    adapter_out = self.mid_adapter_vision(x)
                    x = adapter_out
                # ======================================
                # adapter_out = self.mid_adapter_vision(x)
                # x = adapter_out
                
                
                x = torch.cat([x[:1], x[1+len_texture:]], dim=0)
                continue 

            # === Stage 4: Global Prompt ===
            if i == self.v_idx_global:
               
                
                if self.use_hierarchical:
                    x_nld = x.permute(1, 0, 2)
                    cls_features = x_nld[:, 0, :]
                    global_out = self.global_prompt_pool(x_nld, cls_features)
                    global_prompts = global_out['batched_prompt'].to(dtype).permute(1, 0, 2)
                    global_prompts = global_prompts * self.gate_global
                    
                else:
                    global_prompts = torch.zeros(self.prompt_length, batch_size, self.visual_width, device=device, dtype=dtype)
                # ====================================================
                
                # x_nld = x.permute(1, 0, 2)
                # cls_features = x_nld[:, 0, :]
                # global_out = self.global_prompt_pool(x_nld, cls_features)
                # global_prompts = global_out['batched_prompt'].to(dtype).permute(1, 0, 2)
                # global_prompts = global_prompts * self.gate_global
                
                
                cached_global_base = global_prompts
                
                final_global_prompt = global_prompts
                if current_shared_p is not None:
                    if self.use_hierarchical and self.top_k > 1:
                        current_shared_p_expanded = current_shared_p.repeat(self.top_k, 1, 1)
                        final_global_prompt = final_global_prompt + current_shared_p_expanded
                    else:
                        final_global_prompt = final_global_prompt + current_shared_p
                else:
                    final_global_prompt = final_global_prompt
                    # if self.top_k > 1: current_shared_p = current_shared_p.repeat(self.top_k, 1, 1)
                    # final_global_prompt = final_global_prompt + current_shared_p
                
                len_global = final_global_prompt.shape[0]
                
                #  Global Prompt
                x = torch.cat([x[:1], final_global_prompt, x[1:]], dim=0)
            
            elif i > self.v_idx_global:
                # Deep Prompting
                if current_shared_p is not None:
                    if self.top_k > 1 and self.use_hierarchical: 
                        current_shared_p = current_shared_p.repeat(self.top_k, 1, 1)
                        layer_specific_global = cached_global_base + current_shared_p
                    else:
                        layer_specific_global = cached_global_base + current_shared_p
                else:
                    layer_specific_global = layer_specific_global
                x[1 : 1 + len_global] = layer_specific_global

            x = block(x)
        # === [ABLATION MOD] 7.  Final Adapter ===
        if self.use_adapter:
            final_adapter_out = self.final_adapter_vision(x)
            x = final_adapter_out
        # ============================================
        # final_adapter_out = self.final_adapter_vision(x)
        # x = final_adapter_out
        
        x = x[0, :, :] 
        x = vit.ln_post(x)
        if vit.proj is not None: x = x @ vit.proj
        return F.normalize(x, dim=-1), 0, 0

    def build_causal_attention_mask(self, length, device):
        mask = torch.empty(length, length, device=device)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def encode_text_hierarchical(self, text, t_shared_prompts, dynamic_bias):
        
        batch_size = text.shape[0]
        original_clip = self.original_clip
        dtype = original_clip.dtype
        device = text.device
        prompt_len = self.prompt_length
        
        x = original_clip.token_embedding(text).type(dtype)
        x = x + original_clip.positional_embedding.type(dtype)
        
        if self.use_prototypes:
            
            shared_p0 = t_shared_prompts[0].to(dtype)
            bias = dynamic_bias.permute(1, 0, 2).to(dtype)
            
            p0 = self.text_private_prompts[0].unsqueeze(1) + shared_p0 + bias
            p0 = p0.permute(1, 0, 2) # [B, L, D]
            
        else:
            
            
            # p0 = self.text_private_prompts[0].unsqueeze(0).repeat(batch_size, 1, 1).to(dtype)
            
            
            p0 = torch.zeros(batch_size, self.prompt_length, self.text_width, device=device, dtype=dtype)
            # p0 = p0.permute(1, 0, 2) # [B, L, D]
        # ======================================================
        # P0
        # shared_p0 = t_shared_prompts[0].to(dtype)
        # bias = dynamic_bias.permute(1, 0, 2).to(dtype)
        # p0 = self.text_private_prompts[0].unsqueeze(1) + shared_p0 + bias
        # p0 = p0.permute(1, 0, 2)
        
        x = x.permute(1, 0, 2) 
        x = torch.cat([x[:1], p0.permute(1,0,2), x[1:]], dim=0)
        
        layers = original_clip.transformer.resblocks
        cached_p1_base = None
        cached_p2_base = None

        for i, layer in enumerate(layers):
            current_shared_p = t_shared_prompts.get(i)
            if current_shared_p is not None:
                current_shared_p = current_shared_p.to(dtype)

            # === Stage P1 ===
            if i == self.t_idx_p1:
                p1_base = self.text_private_prompts[1].unsqueeze(1) 
                cached_p1_base = p1_base
                
                p1 = p1_base
                if current_shared_p is not None:
                    p1 = p1 + current_shared_p
                
                x = torch.cat([x[:1], p1, x[1:]], dim=0)
                
            elif i > self.t_idx_p1 and i <= self.t_idx_adapter:
                if current_shared_p is not None:
                    p1_layer = cached_p1_base + current_shared_p
                else:
                    p1_layer = cached_p1_base
                x[1 : 1 + prompt_len] = p1_layer

            # === Adapter ===
            if i == self.t_idx_adapter:
                seq_len = x.shape[0]
                mask = self.build_causal_attention_mask(seq_len, device).to(dtype)
                original_mask = layer.attn_mask
                try:
                    layer.attn_mask = mask
                    x = layer(x)
                finally:
                    layer.attn_mask = original_mask
                # === [ABLATION MOD] 8.  Text Adapter ===
                if self.use_adapter:
                    adapter_out = self.mid_adapter_text(x)
                    x = adapter_out
                # ===========================================    
                # adapter_out = self.mid_adapter_text(x)
                # x = adapter_out
                
                # Prepare P2 for next stage
                next_shared_p = t_shared_prompts.get(i+1)
                if next_shared_p is not None: next_shared_p = next_shared_p.to(dtype)
                
                p2_base = self.text_private_prompts[2].unsqueeze(1)
                cached_p2_base = p2_base
                
                p2 = p2_base
                if next_shared_p is not None:
                    p2 = p2 + next_shared_p
                
                # Replace P1 -> P2
                x = torch.cat([x[:1], x[1+prompt_len:]], dim=0)
                x = torch.cat([x[:1], p2, x[1:]], dim=0)
                continue 

            # === Stage P2 ===
            elif i > self.t_idx_adapter:
                if current_shared_p is not None:
                    p2_layer = cached_p2_base + current_shared_p
                else:
                    p2_layer = cached_p2_base
                x[1 : 1 + prompt_len] = p2_layer

            seq_len = x.shape[0]
            mask = self.build_causal_attention_mask(seq_len, device).to(dtype)
            original_mask = layer.attn_mask
            try:
                layer.attn_mask = mask
                x = layer(x)
            finally:
                layer.attn_mask = original_mask
        # === [ABLATION MOD] 9.  Final Text Adapter ===
        if self.use_adapter:
            final_adapter_out = self.final_adapter_text(x)
            x = final_adapter_out
        # =================================================
        # final_adapter_out = self.final_adapter_text(x)
        # x = final_adapter_out

        x = x.permute(1, 0, 2)
        x = original_clip.ln_final(x).type(dtype)

        total_offset = prompt_len * 2 
        original_eot_indices = text.argmax(dim=-1)
        eot_indices = torch.clamp(original_eot_indices + total_offset, max=x.shape[1]-1)
        
        x = x[torch.arange(batch_size), eot_indices] 
        if original_clip.text_projection is not None:
            x = x @ original_clip.text_projection

        return x

    # ===  (System Validation) ===
    def encode_text(self, text_tokens):
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        dtype = self.original_clip.dtype
        
        with torch.no_grad():
            if self.use_shared_space:
                _, t_shared_prompts = self.shared_net(batch_size)
            else:
                t_shared_prompts = {
                    k: torch.zeros(self.prompt_length, batch_size, self.text_width, device=device, dtype=dtype) 
                    for k in self.t_layers_to_inject
                }
            
            text_context = self._get_text_context(text_tokens)
            dynamic_bias = self.text_generator(text_context.float())
            text_features = self.encode_text_hierarchical(text_tokens, t_shared_prompts, dynamic_bias)
        return F.normalize(text_features, dim=-1)
    
    # def encode_text(self, text_tokens):
    #     batch_size = text_tokens.shape[0]
    #     device = text_tokens.device
    #     dtype = self.original_clip.dtype
    #     USE_SHARED_PROMPTS = True  
        
    #     # =================================
    #     
    #     # v_shared_prompts, t_shared_prompts = self.shared_net(batch_size)
        
    #     with torch.no_grad():
    #         if USE_SHARED_PROMPTS:
    #             _, t_shared_prompts = self.shared_net(batch_size)
    #         else:
    #             t_shared_prompts = {
    #                 k: torch.zeros(self.prompt_length, batch_size, self.text_width, device=device, dtype=dtype) 
    #                 for k in self.t_layers_to_inject
    #             }
    #         
    #         # _, t_shared_prompts = self.shared_net(batch_size)
    #         text_context = self._get_text_context(text_tokens)
    #         dynamic_bias = self.text_generator(text_context.float())
    #         text_features = self.encode_text_hierarchical(text_tokens, t_shared_prompts, dynamic_bias)
    #     return F.normalize(text_features, dim=-1)
    def encode_image(self, images):
        batch_size = images.shape[0]
        device = images.device
        dtype = self.original_clip.dtype
        with torch.no_grad():
            if self.use_shared_space:
                v_shared_prompts, _ = self.shared_net(batch_size)
            else:
                v_shared_prompts = {
                    k: torch.zeros(self.prompt_length, batch_size, self.visual_width, device=device, dtype=dtype) 
                    for k in self.v_layers_to_inject
                }
            class_ids = self.get_prototype_class_ids(images)
            features, _, _ = self.encode_image_with_hierarchical_prompts(images, class_ids, v_shared_prompts)
        return features
# ==============================================================================

# ==============================================================================
def gather_features(features):

    if not (dist.is_available() and dist.is_initialized()):
        return features

    
    features = features.contiguous()
    
    
    world_size = dist.get_world_size()
    gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
    
    
    torch.distributed.all_gather(gathered_features, features)
    
    # 拼接
    all_features = torch.cat(gathered_features, dim=0)
    
    
    
    rank = dist.get_rank()
    all_features[rank * len(features) : (rank + 1) * len(features)] = features
    
    return all_features

import torch.distributed.nn 

# class CLIPFullFinetune(nn.Module):
#     
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.clip_model_name = config.model.original_clip_name
        
#         print(f"🔥 [Baseline] Loading: {self.clip_model_name} (Strict CUSA Reproduction)")
        
#         
#         model, _ = clip.load(self.clip_model_name, device='cpu', jit=False)
#         if hasattr(model, 'visual'): model.visual = model.visual.float()
#         if hasattr(model, 'transformer'): model.transformer = model.transformer.float()
#         self.model = model.float()
#         self.embed_dim = self.model.text_projection.shape[1]

#         
#         self.ln_image = nn.LayerNorm(self.embed_dim)
#         self.proj_image = nn.Linear(self.embed_dim, self.embed_dim)
#         self.ln_text = nn.LayerNorm(self.embed_dim)
#         self.proj_text = nn.Linear(self.embed_dim, self.embed_dim)
        
#         # ======================================================
#         
#         # ======================================================
#         print("⚠️ [Initialization] Using CUSA-style Random Normal Init (std=0.02). Expect low initial scores.")
#         nn.init.normal_(self.proj_image.weight, std=0.02)
#         nn.init.zeros_(self.proj_image.bias)
        
#         nn.init.normal_(self.proj_text.weight, std=0.02)
#         nn.init.zeros_(self.proj_text.bias)
        
#         # LayerNorm 
#         nn.init.ones_(self.ln_image.weight)
#         nn.init.zeros_(self.ln_image.bias)
#         nn.init.ones_(self.ln_text.weight)
#         nn.init.zeros_(self.ln_text.bias)

#         # 3. Logit Scale
#         
#         self.model.logit_scale.data.fill_(np.log(1 / 0.07))

#         
#         for param in self.parameters():
#             param.requires_grad = True

#     def forward(self, batch):
#         images = batch["images"]
#         text_tokens = batch.get("text_tokens")
        
#         
#         image_features = self.model.encode_image(images)
#         image_features = self.ln_image(image_features)
#         image_features = self.proj_image(image_features)
#         image_features = F.normalize(image_features, dim=-1)
        
#         if text_tokens is None: 
#             return {'image_features': image_features}
            
#         text_features = self.model.encode_text(text_tokens)
#         text_features = self.ln_text(text_features)
#         text_features = self.proj_text(text_features)
#         text_features = F.normalize(text_features, dim=-1)

#         
#         logit_scale = self.model.logit_scale.exp().clamp(max=100.0)
#         batch_size = images.shape[0]
#         device = images.device

#         if dist.is_available() and dist.is_initialized():
#             
#             
#             gathered_image_features = torch.distributed.nn.all_gather(image_features)
#             gathered_text_features = torch.distributed.nn.all_gather(text_features)
            
#             
#             all_image_features = torch.cat(gathered_image_features, dim=0)
#             all_text_features = torch.cat(gathered_text_features, dim=0)
            
#             
#             rank = dist.get_rank()
#             labels = torch.arange(batch_size, device=device) + rank * batch_size
#         else:
#             
#             all_image_features = image_features
#             all_text_features = text_features
#             labels = torch.arange(batch_size, device=device)

#         # === 3.  Global Loss ===
#         # [Batch, Global_Batch]
#         logits_per_image = logit_scale * image_features @ all_text_features.t()
#         logits_per_text = logit_scale * text_features @ all_image_features.t()
        
#         loss_ce = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        
#         
#         return {
#             'loss': loss_ce,
#             'aux_loss': torch.tensor(0.0, device=device),
#             'image_features': image_features,    
#             'text_features': text_features,
#             'logits_per_image': logits_per_image 
#         }

#     
#     def encode_image(self, images):
#         with torch.no_grad():
#             features = self.model.encode_image(images)
#             features = self.ln_image(features)
#             features = self.proj_image(features) 
#             return F.normalize(features, dim=-1)

#     def encode_text(self, text_tokens):
#         with torch.no_grad():
#             features = self.model.encode_text(text_tokens)
#             features = self.ln_text(features)
#             features = self.proj_text(features) 
#             return F.normalize(features, dim=-1)




class CLIPFullFinetune(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.clip_model_name = config.model.original_clip_name
        
        print(f"🔥 [CUSA-Style Finetune] Loading: {self.clip_model_name}")
        
       
        model, _ = clip.load(self.clip_model_name, device='cpu', jit=False)
        
     
        if hasattr(model, 'visual'):
            model.visual = model.visual.float()
        if hasattr(model, 'transformer'):
            model.transformer = model.transformer.float()
        self.model = model.float()
        
        self.embed_dim = self.model.text_projection.shape[1] # 512 for ViT-B/32

   
        print("🔧 [Architecture] Adding CUSA-style Projection Layers (LN + FC)...")
        
        
        self.ln_image = nn.LayerNorm(self.embed_dim)
        self.proj_image = nn.Linear(self.embed_dim, self.embed_dim)
        
     
        self.ln_text = nn.LayerNorm(self.embed_dim)
        self.proj_text = nn.Linear(self.embed_dim, self.embed_dim)
        
      
        
        nn.init.normal_(self.proj_image.weight, std=0.02)
        nn.init.zeros_(self.proj_image.bias) 
        
        nn.init.normal_(self.proj_text.weight, std=0.02)
        nn.init.zeros_(self.proj_text.bias)
        # ======================================================

       
        for param in self.parameters():
            param.requires_grad = True
            
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔥 Total Trainable Parameters: {trainable_params:,}")

    def forward(self, batch):
        images = batch["images"]
        text_tokens = batch.get("text_tokens")
        
        
        image_features = self.model.encode_image(images)
        
        
        image_features = self.ln_image(image_features)
        image_features = self.proj_image(image_features)
        
        
        image_features = F.normalize(image_features, dim=-1)
        
        if text_tokens is None:
            return {'image_features': image_features}
            
        
        text_features = self.model.encode_text(text_tokens)
        
       
        text_features = self.ln_text(text_features)
        text_features = self.proj_text(text_features)
        
        text_features = F.normalize(text_features, dim=-1)
        
        
        logit_scale = self.model.logit_scale.exp()
        
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=images.device)
             
        loss_ce = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        
        return {
            'loss': loss_ce,
            'image_features': image_features,
            'text_features': text_features,
            'logits_per_image': logits_per_image
        }

    
    def encode_image(self, images):
        with torch.no_grad():
            features = self.model.encode_image(images)
            
            features = self.ln_image(features)
            features = self.proj_image(features)
            return F.normalize(features, dim=-1)

    def encode_text(self, text_tokens):
        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
            
            features = self.ln_text(features)
            features = self.proj_text(features)
            return F.normalize(features, dim=-1)