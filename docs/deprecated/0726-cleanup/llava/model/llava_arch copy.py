"""作用：实现 LLaVA/HiDe-LLaVA 模型加载、结构封装、多模态输入整理和权重转换工具。"""

#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import inspect

from .multimodal_encoder.builder import build_vision_tower, build_text_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from HiDe.peft.tuners import HiDeMOELoraModel
from collections import deque


class LlavaMetaModel:

    """作用：LlavaMetaModel 类封装模型结构、配置或前向传播相关逻辑。"""
    def __init__(self, config):
        """作用：初始化对象状态、保存配置参数，并构建后续方法需要使用的成员变量。"""
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        
        if hasattr(config, "mm_text_tower"):
            self.text_tower = build_text_tower(config, delay_load=True)

    def get_vision_tower(self):
        """作用：读取、筛选或组装指定对象并返回给调用方。"""
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_text_tower(self):
        """作用：读取、筛选或组装指定对象并返回给调用方。"""
        text_tower = getattr(self, 'text_tower', None)
        if type(text_tower) is list:
            text_tower = text_tower[0]
        return text_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        """作用：执行 initialize_vision_modules 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                """作用：读取、筛选或组装指定对象并返回给调用方。"""
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'),strict = False)

    def initialize_text_modules(self, model_args, fsdp=None):
        """作用：执行 initialize_text_modules 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        text_tower = model_args.text_tower

        if self.get_text_tower() is None:
            text_tower = build_text_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.text_tower = [text_tower]
            else:
                self.text_tower = text_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                text_tower = self.text_tower[0]
            else:
                text_tower = self.text_tower
            text_tower.load_model()


class LlavaMetaForCausalLM(ABC):
    """作用：LlavaMetaForCausalLM 类封装模型结构、配置或前向传播相关逻辑。"""
    @abstractmethod
    def get_model(self):
        """作用：读取、筛选或组装指定对象并返回给调用方。"""
        pass

    def get_vision_tower(self):
        """作用：读取、筛选或组装指定对象并返回给调用方。"""
        return self.get_model().get_vision_tower()

    def get_text_tower(self):
        """作用：读取、筛选或组装指定对象并返回给调用方。"""
        return self.get_model().get_text_tower()

    def encode_images(self, images):
        """作用：执行 encode_images 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        clip_image_features, image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return clip_image_features.to(self.device), image_features.to(self.device)

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        """
        作用：将文本 token、图像特征和标签整理成 LLaVA 前向传播所需的多模态输入。
        
        训练阶段会在线更新当前任务的 image/text 高斯统计量；推理阶段会根据统计量计算样本到各 expert 的相似度，再结合 adaptive_w_img 完成 expert 路由。
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_guide_features, image_features = self.encode_images(images)

        assert image_features.shape[1] == 576, 'vision tower not a withprojection version.'
        text_tower = self.get_text_tower()

        # with torch.no_grad():
        #     # image_guide_features: bs, 4096
        #     image_guide_features = image_features[:,0]
        
        input_pad = np.where(input_ids.cpu().detach().numpy()!=-200,input_ids.cpu().detach().numpy(),self.tokenizer.pad_token_id)
        decoded_inputs = self.tokenizer.batch_decode(input_pad, skip_special_tokens=True)
        decoded_hidden_inputs = ['\n'.join(decode_input.split('\n')[1:]) for decode_input in decoded_inputs]
        decoded_clip_inputs = [decode_input.split(' ASSISTANT')[0] for decode_input in decoded_hidden_inputs]

        clip_text_inputs = self.clip_tokenizer(
                decoded_clip_inputs,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )

        # text_guide_features: bs, 768
        text_guide_features = text_tower(clip_text_inputs)

        # 原始代码
        # region
        # if self.training:

        #     current_image_features = image_guide_features  # [batch_size, feature_dim]
        #     current_text_features = text_guide_features  # [batch_size, feature_dim]
        #     task_id = self.cur_task

        #     image_sum = self.image_anchors[task_id] * self.image_boundary[task_id] + current_image_features.sum(dim=0)
        #     text_sum = self.text_anchors[task_id] * self.text_boundary[task_id] + current_text_features.sum(dim=0)

        #     self.image_boundary[task_id].data += current_image_features.shape[0]
        #     self.text_boundary[task_id].data += current_text_features.shape[0]

        #     self.image_anchors[task_id] = image_sum / self.image_boundary[task_id]
        #     self.text_anchors[task_id] = text_sum / self.text_boundary[task_id]
        # else:
        #     image_sim = []
        #     text_sim = []
        #     for image_anchor in self.image_anchors:
        #         image_sims = F.cosine_similarity(image_guide_features.unsqueeze(1), image_anchor, dim=2)
        #         image_sim.append(image_sims.max().item())
        #     for text_anchor in self.text_anchors:
        #         text_sims = F.cosine_similarity(text_guide_features.unsqueeze(1), text_anchor, dim=2)
        #         text_sim.append(text_sims.max().item())

        #     image_sim = np.array(image_sim[:self.expert_num]) 
        #     text_sim = np.array(text_sim[:self.expert_num])  

        #     sim = (image_sim + text_sim) / 2

        #     sim_tensor = torch.tensor(sim, dtype=torch.float32)

        #     sim_softmax = F.softmax(sim_tensor / 0.1)
        #     # breakpoint()
        #     # compute_expert_weight = torch.sigmoid(shifted_conf).tolist()
        #     compute_expert_weight = sim_softmax.tolist()
        #     # print(compute_expert_weight)

        #     proj_names = [
        #         'q_proj', 'k_proj', 'v_proj', 'o_proj',  # self_attn 
        #         'gate_proj', 'up_proj', 'down_proj'      # mlp 
        #     ]
        #     for proj_name in proj_names:
        #         if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        #             proj_layer = getattr(self.model.layers[-1].self_attn, proj_name)
        #         else:
        #             proj_layer = getattr(self.model.layers[-1].mlp, proj_name)

        #         proj_layer.expert_weight = compute_expert_weight
        #         # print(proj_layer.expert_weight)
        # endregion

        # =========================
        # ----- TRAIN MODE --------
        # =========================
        if self.training:

            task_id = self.cur_task

            img = image_guide_features          # [B, D]
            txt = text_guide_features           # [B, D]
            B = img.shape[0]

            # --------- image mean/var/count ----------
            old_count = self.image_count[task_id].item()
            new_count = old_count + B

            old_mean = self.image_mean[task_id].data
            old_var  = self.image_var[task_id].data

            batch_mean = img.mean(dim=0, keepdim=True)
            batch_var  = img.var(dim=0, unbiased=False, keepdim=True)

            new_mean = (old_mean * old_count + batch_mean * B) / new_count
            new_var  = (
                old_var * old_count +
                batch_var * B +
                (old_mean - batch_mean).pow(2) * (old_count * B / new_count)
            ) / new_count

            self.image_mean[task_id].data.copy_(new_mean)
            self.image_var[task_id].data.copy_(new_var)
            self.image_count[task_id].data.fill_(new_count)

            # -------- text mean/var/count ----------
            old_count = self.text_count[task_id].item()
            new_count = old_count + B

            old_mean = self.text_mean[task_id].data
            old_var  = self.text_var[task_id].data

            batch_mean = txt.mean(dim=0, keepdim=True)
            batch_var  = txt.var(dim=0, unbiased=False, keepdim=True)

            new_mean = (old_mean * old_count + batch_mean * B) / new_count
            new_var  = (
                old_var * old_count +
                batch_var * B +
                (old_mean - batch_mean).pow(2) * (old_count * B / new_count)
            ) / new_count

            self.text_mean[task_id].data.copy_(new_mean)
            self.text_var[task_id].data.copy_(new_var)
            self.text_count[task_id].data.fill_(new_count)

            # 原 boundary 更新保持不变
            self.image_boundary[task_id].data += B
            self.text_boundary[task_id].data += B


        # =========================
        # ----- TEST MODE ---------
        # =========================
        else:
            image_sim = []
            text_sim = []

            img = image_guide_features         # [1, D]
            txt = text_guide_features          # [1, D]

            # 1. 自适应 Scale 计算模块
            valid_img_vars = []
            valid_txt_vars = []
            cur_task_num = self.expert_num - 1

            # 遍历所有可能的 Expert，收集已训练任务的方差
            for t in range(len(self.image_mean)): # 或者 self.expert_num
                # 判断该 Expert 是否已初始化/训练 (用均值是否为0或方差是否有效来判断)
                # 假设未训练的 Expert 均值/方差全是 0，或者你可以维护一个 trained_task_ids 列表
                if self.image_mean[t].abs().sum().item() > 1e-6: 
                    valid_img_vars.append(self.image_var[t].sum().item())
                    valid_txt_vars.append(self.text_var[t].sum().item())
                else:
                    cur_task_num = t-1 if cur_task_num==self.expert_num - 1 else cur_task_num

            # 计算自适应 scale (防止除以0)
            # 设定基准常数 K=1.0。
            # K=1.0 意味着平均方差会被映射到半径 r = exp(-1) ≈ 0.368
            # K=0.5 意味着平均方差会被映射到半径 r = exp(-0.5) ≈ 0.606 (更靠近边缘)
            K = 1.0 

            if len(valid_img_vars) > 0:
                avg_var_img = sum(valid_img_vars) / len(valid_img_vars)
                scale_img = K / (avg_var_img + 1e-6)
            else:
                scale_img = 0.007 # 默认备用值

            if len(valid_txt_vars) > 0:
                avg_var_txt = sum(valid_txt_vars) / len(valid_txt_vars)
                scale_txt = K / (avg_var_txt + 1e-6)
            else:
                scale_txt = 0.003 # 默认备用值

            # 2. 先把测试样本映射好 (因为它对所有 Expert 都是一样的，不用重复算)
            # img: [1, D], txt: [1, D]
            # 测试样本方差为0，直接映射到边界
            z_test_img = map_to_poincare(img, var_sum=None)
            z_test_txt = map_to_poincare(txt, var_sum=None)
            breakpoint()

            for t in range(self.expert_num):

                # region
                # # -------- image Gaussian similarity --------
                # mean_i = self.image_mean[t]         # [1, D]
                # var_i  = self.image_var[t] + 1e-6   # [1, D]

                # diff = img - mean_i
                # sim_i = - (diff.pow(2) / var_i).sum().item()
                # image_sim.append(sim_i)

                # # -------- text Gaussian similarity --------
                # mean_t = self.text_mean[t]
                # var_t  = self.text_var[t] + 1e-6

                # diff = txt - mean_t
                # sim_t = - (diff.pow(2) / var_t).sum().item()
                # text_sim.append(sim_t)
                # endregion

                # region
                # # Wasserstein Distance
                # mean_i = self.image_mean[t]         # [1, D]
                # var_i  = self.image_var[t]      # [1, D]
                # sim_i = - ( (mean_i - img).pow(2).sum().item() + var_i.sum().item()*0.5 )
                # image_sim.append(sim_i)
                # # 方差统计结果：[106.7500, 172.0000, 149.0000, 154.5000, 129.0000, 152.7500, 156.7500, 141.7500]
                # # 方差统计结果：[114.6250, 64.3125, 105.0625, 80.8125, 13.1641, 116.8750]

                # mean_t = self.text_mean[t]
                # var_t  = self.text_var[t]
                # sim_t = - ( (mean_t - txt).pow(2).sum().item() + var_t.sum().item()*0.5 )
                # text_sim.append(sim_t)
                # # 方差统计结果：[548.0000, 465.7500,   4.4961, 403.7500, 357.0000, 308.2500, 407.0000, 22.2500]
                # # 方差统计结果：[3.1758, 307.7500, 2.8789, 359.5000, 223.0000, 2.8730]
                # endregion

                # 双曲空间庞加莱距离

                # 3. 循环计算距离
                # ---------------- Image ----------------
                mean_i = self.image_mean[t]         # [1, D]
                var_i  = self.image_var[t]          # [1, D]
                var_sum_i = var_i.sum().item()
                if t > cur_task_num:
                    # 说明该 Expert 没有被训练过，跳过
                    image_sim.append(-float('inf'))
                    text_sim.append(-float('inf'))
                    continue
                
                # 映射 Expert 到双曲空间
                z_expert_i = map_to_poincare(mean_i, var_sum=var_sum_i, scale=scale_img)
                
                # 计算庞加莱距离
                dist_i = poincare_dist(z_test_img, z_expert_i).item()
                
                # 注意：我们要的是"相似度"，距离越小越好，所以取负
                sim_i = -dist_i
                image_sim.append(sim_i)

                # ---------------- Text ----------------
                mean_t = self.text_mean[t]
                var_t  = self.text_var[t]
                var_sum_t = var_t.sum().item()
                
                # 映射 Expert 到双曲空间
                # 这里会自动处理 ImageNet (Var=4.5 -> r≈0.99) 和 VQA (Var=548 -> r≈0.2)
                z_expert_t = map_to_poincare(mean_t, var_sum=var_sum_t, scale=scale_txt)
                
                # 计算庞加莱距离
                dist_t = poincare_dist(z_test_txt, z_expert_t).item()
                
                sim_t = -dist_t
                text_sim.append(sim_t)

            image_sim = np.array(image_sim)
            image_sim_tensor = torch.tensor(image_sim, dtype=torch.float32)
            text_sim = np.array(text_sim)
            text_sim_tensor = torch.tensor(text_sim, dtype=torch.float32)

            # image_softmax = F.softmax(image_sim_tensor / np.sqrt(768), dim=0)
            # text_softmax = F.softmax(text_sim_tensor / np.sqrt(768), dim=0)
            image_softmax = F.softmax(image_sim_tensor / torch.std(image_sim_tensor[:cur_task_num+1]), dim=0)
            text_softmax = F.softmax(text_sim_tensor / torch.std(text_sim_tensor[:cur_task_num+1]), dim=0)

            # image_weight = [
            #     [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            #     [0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0],
            #     [0.9861,0.6459,0.0077,0.0,0.0,0.0,0.0,0.0],
            #     [0.9963,0.6336,0.0014,0.6000,0.0,0.0,0.0,0.0],
            #     [0.9952,0.6105,0.0003,0.6055,0.8828,0.0,0.0,0.0],
            #     [0.9987,0.6445,0.0001,0.4945,0.9256,0.3373,0.0,0.0],
            #     [0.9996,0.6518,0.0000,0.5168,0.9541,0.3062,0.5168,0.0],
            #     [0.9992,0.6529,0.0003,0.5172,0.9430,0.3556,0.5172,0.1229]
            # ]
            image_weight = [
                [1.0,0.0,0.0,0.0,0.0,0.0],
                [0.5,0.5,0.0,0.0,0.0,0.0],
                [0.3396,0.8026,0.3236,0.0,0.0,0.0],
                [0.0928,0.9635,0.0604,0.8521,0.0,0.0],
                [0.0199,0.7544,0.0114,0.6316,0.9988,0.0],
                [0.0000,0.7534,0.6129,0.6056,0.9996,0.5437]
            ]
            weight_arr = np.zeros(self.expert_num,dtype=np.float32)
            weight_arr[:cur_task_num+1] = 1.0
            text_weight = weight_arr - np.array(image_weight[cur_task_num])
            sim = image_softmax * np.array(image_weight[cur_task_num]) + text_softmax * np.array(text_weight)
            # sim_softmax = sim / (sim.sum() + 1e-9)
            sim_softmax = torch.zeros_like(sim)
            sim_softmax[sim.argmax()] = 1.0

            # sim = (image_sim + text_sim) / 2.0
            # sim_tensor = torch.tensor(sim, dtype=torch.float32)
            # sim_softmax = F.softmax(sim_tensor / 768, dim=0)
            compute_expert_weight = sim_softmax.tolist()
            # print(compute_expert_weight)

            # compute_expert_weight = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]  # for upperbound test
            # breakpoint()
            proj_names = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ]
            for proj_name in proj_names:
                if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    proj_layer_list = [getattr(self.model.layers[i].self_attn, proj_name) for i in range(len(self.model.layers))]
                else:
                    proj_layer_list = [getattr(self.model.layers[i].mlp, proj_name) for i in range(len(self.model.layers))]

                for proj_layer in proj_layer_list:
                    proj_layer.expert_weight = compute_expert_weight






        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        """作用：执行 initialize_vision_tokenizer 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


# --- 辅助函数定义 (建议放在类外或作为静态方法) ---

def map_to_poincare(vector, var_sum=None, scale=1.0):
    """作用：把单个特征向量或 expert 均值映射到 Poincare ball，供双曲距离路由使用。"""
    # 1. 计算方向 (Direction)
    norm = vector.norm(p=2, dim=-1, keepdim=True) + 1e-6
    direction = vector / norm
    
    # 2. 计算半径 (Radius)
    if var_sum is None:
        # 测试样本：方差为0 -> 半径接近边界 (1.0)
        # 为了数值稳定性，取 0.999
        radius = 0.999
    else:
        # Expert：方差越大，越靠近圆心(0)；方差越小，越靠近边界(1)
        # r = exp(-scale * sum(var))
        radius = torch.exp(torch.tensor(-scale * var_sum))
        # 截断以防数值溢出
        radius = torch.clamp(radius, min=0.001, max=0.999).item()
        
    return direction * radius

def poincare_dist(u, v):
    """作用：计算两个 Poincare ball 向量之间的双曲距离，并做数值稳定裁剪。"""
    # 欧氏距离平方 ||u-v||^2
    sq_dist = (u - v).pow(2).sum(dim=-1)
    
    # 模长平方 ||u||^2, ||v||^2
    u_sq = u.pow(2).sum(dim=-1)
    v_sq = v.pow(2).sum(dim=-1)
    
    # 公式: arccosh( 1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)) )
    numerator = 2 * sq_dist
    denominator = (1 - u_sq) * (1 - v_sq)
    denominator = torch.clamp(denominator, min=1e-7) # 防止除零
    
    arg = 1 + numerator / denominator
    arg = torch.clamp(arg, min=1.0 + 1e-7) # 防止精度误差导致小于1
    
    return torch.acosh(arg)

