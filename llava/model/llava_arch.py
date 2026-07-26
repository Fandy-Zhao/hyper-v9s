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
import os
import json

from .multimodal_encoder.builder import build_vision_tower, build_text_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from Hyper.peft.tuners import HyperMOELoraModel
from collections import deque


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        
        if hasattr(config, "mm_text_tower"):
            self.text_tower = build_text_tower(config, delay_load=True)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_text_tower(self):
        text_tower = getattr(self, 'text_tower', None)
        if type(text_tower) is list:
            text_tower = text_tower[0]
        return text_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
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
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'),strict = False)

    def initialize_text_modules(self, model_args, fsdp=None):
        text_tower = model_args.text_tower
        mm_text_select_layer = getattr(model_args, 'mm_text_select_layer', -1)

        self.config.mm_text_tower = text_tower
        self.config.mm_text_select_layer = mm_text_select_layer

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
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_text_tower(self):
        return self.get_model().get_text_tower()

    def encode_images(self, images):
        clip_image_features, image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return clip_image_features.to(self.device), image_features.to(self.device)

    def _infer_valid_task_ids(self):
        max_tasks = min(self.expert_num, len(self.image_mean), len(self.text_mean))
        valid_task_ids = []
        for task_id in range(max_tasks):
            has_image_stats = self.image_mean[task_id].detach().abs().sum().item() > 1e-6
            has_text_stats = self.text_mean[task_id].detach().abs().sum().item() > 1e-6
            if has_image_stats or has_text_stats:
                valid_task_ids.append(task_id)
            elif valid_task_ids:
                break
        if not valid_task_ids:
            valid_task_ids = [0]
        return valid_task_ids

    def _get_prior_img_per_task(self, valid_task_ids, device, dtype):
        adaptive_w_img = getattr(self.config, "adaptive_w_img", None)
        values = []
        for task_id in valid_task_ids:
            if adaptive_w_img is not None and task_id < len(adaptive_w_img):
                values.append(float(adaptive_w_img[task_id]))
            else:
                values.append(0.5)
        return torch.tensor(values, device=device, dtype=dtype)

    def _normalized_entropy(self, p):
        eps = 1e-8
        num_tasks = max(p.size(-1), 2)
        denom = torch.log(torch.tensor(float(num_tasks), device=p.device, dtype=p.dtype))
        return -(p * (p + eps).log()).sum(dim=-1, keepdim=True) / denom

    def _confidence_and_margin(self, p):
        if p.size(-1) == 1:
            conf = torch.ones(p.size(0), 1, device=p.device, dtype=p.dtype)
            return conf, torch.ones_like(conf)
        top2 = torch.topk(p, k=2, dim=-1).values
        conf = top2[:, 0:1]
        margin = top2[:, 0:1] - top2[:, 1:2]
        return conf, margin

    def _distribution_from_scores(self, scores):
        if scores.size(-1) == 1:
            return torch.ones_like(scores)
        std = torch.std(scores, dim=-1, keepdim=True, unbiased=False)
        std = torch.where(torch.isnan(std) | (std < 1e-6), torch.ones_like(std), std)
        return F.softmax(scores / std, dim=-1)

    def _compute_modality_scores(self, image_guide_features, text_guide_features, valid_task_ids, use_hyperbolic=True):
        img = image_guide_features
        txt = text_guide_features

        valid_img_vars = [self.image_var[t].detach().sum().item() for t in valid_task_ids]
        valid_txt_vars = [self.text_var[t].detach().sum().item() for t in valid_task_ids]
        scale_img = 1.0 / ((sum(valid_img_vars) / len(valid_img_vars)) + 1e-6) if valid_img_vars else 0.007
        scale_txt = 1.0 / ((sum(valid_txt_vars) / len(valid_txt_vars)) + 1e-6) if valid_txt_vars else 0.003

        z_test_img = map_to_poincare(img, var_sum=None) if use_hyperbolic else None
        z_test_txt = map_to_poincare(txt, var_sum=None) if use_hyperbolic else None

        image_scores = []
        text_scores = []
        for task_id in valid_task_ids:
            mean_i = self.image_mean[task_id].to(device=img.device, dtype=img.dtype)
            mean_t = self.text_mean[task_id].to(device=txt.device, dtype=txt.dtype)
            if use_hyperbolic:
                var_sum_i = self.image_var[task_id].detach().sum().item()
                var_sum_t = self.text_var[task_id].detach().sum().item()
                z_expert_i = map_to_poincare(mean_i, var_sum=var_sum_i, scale=scale_img)
                z_expert_t = map_to_poincare(mean_t, var_sum=var_sum_t, scale=scale_txt)
                image_scores.append(-poincare_dist(z_test_img, z_expert_i))
                text_scores.append(-poincare_dist(z_test_txt, z_expert_t))
            else:
                image_scores.append(-(img - mean_i).pow(2).sum(dim=-1))
                text_scores.append(-(txt - mean_t).pow(2).sum(dim=-1))

        return torch.stack(image_scores, dim=-1), torch.stack(text_scores, dim=-1)

    def compute_modality_route(
        self,
        image_guide_features,
        text_guide_features,
        valid_task_ids,
        routing_mode=None,
        use_hyperbolic=True,
    ):
        routing_mode = routing_mode or getattr(self.config, "modality_routing_mode", "task")
        if routing_mode not in ("task", "sample", "sample_rule"):
            raise ValueError(f"Unsupported modality routing mode: {routing_mode}")

        image_scores, text_scores = self._compute_modality_scores(
            image_guide_features,
            text_guide_features,
            valid_task_ids,
            use_hyperbolic=use_hyperbolic,
        )
        p_v = self._distribution_from_scores(image_scores)
        p_s = self._distribution_from_scores(text_scores)
        prior_img_per_task = self._get_prior_img_per_task(valid_task_ids, p_v.device, p_v.dtype)

        if routing_mode == "task":
            w_img = prior_img_per_task.unsqueeze(0)
            alpha = None
            p_d = w_img * p_v + (1.0 - w_img) * p_s
        elif routing_mode == "sample_rule":
            h_v = self._normalized_entropy(p_v)
            h_s = self._normalized_entropy(p_s)
            c_v, m_v = self._confidence_and_margin(p_v)
            c_s, m_s = self._confidence_and_margin(p_s)
            alpha = torch.softmax(torch.cat([-h_v + c_v + m_v, -h_s + c_s + m_s], dim=-1), dim=-1)
            p_d = alpha[:, 0:1] * p_v + alpha[:, 1:2] * p_s
        else:
            instance_router = getattr(self, "instance_router", None)
            if instance_router is None:
                raise ValueError("modality_routing_mode='sample' requires instance_router to be initialized.")
            q = 0.5 * (p_v.detach() + p_s.detach())
            prior_img = (q * prior_img_per_task.unsqueeze(0)).sum(dim=-1, keepdim=True)
            prior_txt = 1.0 - prior_img
            prior_alpha = torch.cat([prior_img, prior_txt], dim=-1)

            h_v = self._normalized_entropy(p_v)
            h_s = self._normalized_entropy(p_s)
            c_v, m_v = self._confidence_and_margin(p_v)
            c_s, m_s = self._confidence_and_margin(p_s)
            route_feat = torch.cat([h_v, h_s, c_v, c_s, m_v, m_s, prior_img, prior_txt], dim=-1)
            router_param = next(instance_router.parameters(), None)
            router_device = router_param.device if router_param is not None else route_feat.device
            router_dtype = router_param.dtype if router_param is not None else route_feat.dtype
            alpha = instance_router(
                route_feat.detach().to(device=router_device, dtype=router_dtype),
                prior_alpha.detach().to(device=router_device, dtype=router_dtype),
            ).to(device=p_v.device, dtype=p_v.dtype)
            p_d = alpha[:, 0:1] * p_v.detach() + alpha[:, 1:2] * p_s.detach()

        # Keep the final routing distribution normalized in mixed-precision runs.
        p_d = p_d.float()
        p_d = p_d / p_d.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if os.environ.get("C1_DUMP_ROUTE", "0") == "1":
            try:
                dump_path = os.environ.get("C1_DUMP_ROUTE_PATH")
                if dump_path:
                    limit = int(os.environ.get("C1_DUMP_ROUTE_LIMIT", "3000"))
                    dump_count = int(getattr(self, "_c1_dump_route_count", 0))
                    if dump_count < limit:
                        target_task_id = os.environ.get("C1_DUMP_TARGET_TASK_ID", "")
                        branch = os.environ.get("C1_DUMP_BRANCH", "")
                        task_name = os.environ.get("C1_DUMP_TASK", "")
                        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
                        task_ids_tensor = torch.tensor(valid_task_ids, device=p_d.device, dtype=torch.long)

                        def _entropy(prob):
                            prob_f = prob.float().clamp_min(1e-8)
                            return (-(prob_f * prob_f.log()).sum(dim=-1)).detach().cpu().tolist()

                        def _top1(prob):
                            vals, idx = prob.float().max(dim=-1)
                            ids = task_ids_tensor[idx].detach().cpu().tolist()
                            return ids, vals.detach().cpu().tolist()

                        p_v_top1, p_v_prob = _top1(p_v)
                        p_s_top1, p_s_prob = _top1(p_s)
                        p_d_top1, p_d_prob = _top1(p_d)
                        p_v_entropy = _entropy(p_v)
                        p_s_entropy = _entropy(p_s)
                        p_d_entropy = _entropy(p_d)
                        alpha_to_dump = alpha
                        if alpha_to_dump is None:
                            selected_prior = prior_img_per_task[torch.argmax(p_d, dim=-1)]
                            alpha_to_dump = torch.stack([selected_prior, 1.0 - selected_prior], dim=-1)
                        alpha_cpu = alpha_to_dump.detach().float().cpu()
                        p_v_cpu = p_v.detach().float().cpu()
                        p_s_cpu = p_s.detach().float().cpu()
                        p_d_cpu = p_d.detach().float().cpu()
                        prior_cpu = prior_img_per_task.detach().float().cpu().tolist()
                        rows = []
                        for i in range(p_d.size(0)):
                            if dump_count + len(rows) >= limit:
                                break
                            target_int = None
                            try:
                                target_int = int(target_task_id)
                            except Exception:
                                pass
                            row = {
                                "dump_index": dump_count + len(rows),
                                "branch": branch,
                                "task": task_name,
                                "mode": routing_mode,
                                "target_task_id": target_int if target_int is not None else target_task_id,
                                "valid_task_ids": list(valid_task_ids),
                                "alpha_img": float(alpha_cpu[i, 0].item()),
                                "alpha_text": float(alpha_cpu[i, 1].item()),
                                "prior_alpha_img_by_task": prior_cpu,
                                "prior_alpha_text_by_task": [1.0 - x for x in prior_cpu],
                                "p_v": p_v_cpu[i].tolist() if p_v_cpu.size(-1) <= 16 else None,
                                "p_s": p_s_cpu[i].tolist() if p_s_cpu.size(-1) <= 16 else None,
                                "p_d": p_d_cpu[i].tolist() if p_d_cpu.size(-1) <= 16 else None,
                                "p_v_top1": p_v_top1[i],
                                "p_s_top1": p_s_top1[i],
                                "p_d_top1": p_d_top1[i],
                                "p_v_top1_prob": float(p_v_prob[i]),
                                "p_s_top1_prob": float(p_s_prob[i]),
                                "p_d_top1_prob": float(p_d_prob[i]),
                                "p_v_entropy": float(p_v_entropy[i]),
                                "p_s_entropy": float(p_s_entropy[i]),
                                "p_d_entropy": float(p_d_entropy[i]),
                                "top1_match_pv_ps": bool(p_v_top1[i] == p_s_top1[i]),
                                "top1_match_pd_target_task": bool(target_int is not None and p_d_top1[i] == target_int),
                                "selected_expert_or_topk_experts": [p_d_top1[i]],
                            }
                            rows.append(row)
                        with open(dump_path, "a", encoding="utf-8") as f:
                            for row in rows:
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        self._c1_dump_route_count = dump_count + len(rows)
            except Exception as exc:
                if not getattr(self, "_c1_dump_route_warned", False):
                    print(f"[C1_ROUTE_DUMP_WARNING] {exc}")
                    self._c1_dump_route_warned = True
        return {
            "p_v": p_v,
            "p_s": p_s,
            "p_d": p_d,
            "alpha": alpha,
            "prior_img_per_task": prior_img_per_task,
        }

    def _sample_router_replay_features(
        self,
        replay_task_ids,
        valid_task_ids,
        samples_per_task,
        device,
        dtype,
    ):
        image_samples = []
        text_samples = []
        targets = []
        task_to_index = {task_id: index for index, task_id in enumerate(valid_task_ids)}

        for task_id in replay_task_ids:
            image_mean = self.image_mean[task_id].detach().to(device=device, dtype=torch.float32)
            image_std = self.image_var[task_id].detach().to(device=device, dtype=torch.float32).clamp_min(1e-6).sqrt()
            text_mean = self.text_mean[task_id].detach().to(device=device, dtype=torch.float32)
            text_std = self.text_var[task_id].detach().to(device=device, dtype=torch.float32).clamp_min(1e-6).sqrt()

            image_noise = torch.randn(samples_per_task, image_mean.size(-1), device=device)
            text_noise = torch.randn(samples_per_task, text_mean.size(-1), device=device)
            image_samples.append((image_mean + image_noise * image_std).to(dtype=dtype))
            text_samples.append((text_mean + text_noise * text_std).to(dtype=dtype))
            targets.append(
                torch.full(
                    (samples_per_task,),
                    task_to_index[task_id],
                    device=device,
                    dtype=torch.long,
                )
            )

        return (
            torch.cat(image_samples, dim=0),
            torch.cat(text_samples, dim=0),
            torch.cat(targets, dim=0),
        )

    def _resolve_eval_modality_routing_mode(self):
        train_mode = getattr(self.config, "modality_routing_mode", "task")
        eval_mode = getattr(self.config, "eval_modality_routing_mode", "same")
        return train_mode if eval_mode == "same" else eval_mode

    def _apply_expert_weight(self, expert_weight):
        compute_expert_weight = expert_weight.detach().float().cpu().tolist()
        if len(compute_expert_weight) == 1:
            compute_expert_weight = compute_expert_weight[0]
        proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        for proj_name in proj_names:
            if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj_layer_list = [getattr(self.model.layers[i].self_attn, proj_name) for i in range(len(self.model.layers))]
            else:
                proj_layer_list = [getattr(self.model.layers[i].mlp, proj_name) for i in range(len(self.model.layers))]

            for proj_layer in proj_layer_list:
                proj_layer.expert_weight = compute_expert_weight

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
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
        self.router_aux_loss = None
        self.router_replay_loss = None
        self.router_log_dict = {}

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
            # breakpoint()
            # 原 boundary 更新保持不变
            self.image_boundary[task_id].data += B
            self.text_boundary[task_id].data += B

            routing_mode = getattr(self.config, "modality_routing_mode", "task")
            if routing_mode == "sample":
                valid_task_ids = list(range(min(task_id + 1, self.expert_num)))
                min_count = getattr(self.config, "router_min_count", 128)
                image_count = self.image_count[task_id].detach().item()
                text_count = self.text_count[task_id].detach().item()
                if len(valid_task_ids) > 1 and image_count >= min_count and text_count >= min_count:
                    route_out = self.compute_modality_route(
                        image_guide_features=img.detach(),
                        text_guide_features=txt.detach(),
                        valid_task_ids=valid_task_ids,
                        routing_mode="sample",
                    )
                    p_d = route_out["p_d"]
                    target_index = valid_task_ids.index(task_id)
                    target = torch.full(
                        (p_d.size(0),),
                        fill_value=target_index,
                        device=p_d.device,
                        dtype=torch.long,
                    )
                    self.router_aux_loss = F.nll_loss(torch.log(p_d + 1e-8), target)
                    self.router_log_dict = {
                        "router_aux_loss": self.router_aux_loss.detach(),
                        "router_alpha": route_out["alpha"].detach() if route_out["alpha"] is not None else None,
                    }

                    replay_weight = getattr(self.config, "router_replay_weight", 0.0)
                    replay_samples = getattr(self.config, "router_replay_samples", 4)
                    replay_task_ids = [
                        replay_task_id
                        for replay_task_id in valid_task_ids
                        if replay_task_id != task_id
                        and self.image_count[replay_task_id].detach().item() >= min_count
                        and self.text_count[replay_task_id].detach().item() >= min_count
                    ]
                    if replay_weight > 0 and replay_samples > 0 and replay_task_ids:
                        replay_img, replay_txt, replay_target = self._sample_router_replay_features(
                            replay_task_ids=replay_task_ids,
                            valid_task_ids=valid_task_ids,
                            samples_per_task=replay_samples,
                            device=img.device,
                            dtype=img.dtype,
                        )
                        replay_out = self.compute_modality_route(
                            image_guide_features=replay_img,
                            text_guide_features=replay_txt,
                            valid_task_ids=valid_task_ids,
                            routing_mode="sample",
                        )
                        self.router_replay_loss = F.nll_loss(
                            torch.log(replay_out["p_d"] + 1e-8),
                            replay_target,
                        )
                        self.router_log_dict["router_replay_loss"] = self.router_replay_loss.detach()

            # =====================================================================
            # [新增] 收集并保存 Raw Features，用于 Rebuttal 的高斯假设验证
            # =====================================================================
            # 为了防止 OOM，设置采样率，比如只保存 10% 的 batch 特征
            # sample_rate = 1.0 
            # if torch.rand(1).item() < sample_rate:
            #     # 将特征移到 CPU 并转为 float16 节省内存
            #     if not hasattr(self, "saved_img_features"):
            #         self.saved_img_features = []
            #         self.saved_txt_features =[]
            #         self.feature_save_dir = "./runs/rebuttal_features" # 你可以改成你的 output_dir
            #         os.makedirs(self.feature_save_dir, exist_ok=True)
                    
            #     self.saved_img_features.append(img.detach().cpu().half())
            #     self.saved_txt_features.append(txt.detach().cpu().half())
                
            #     # 当攒够一定数量（例如 50 个 batch）时，落盘保存并清空内存
            #     if len(self.saved_img_features) >= 25:
            #         img_tensor = torch.cat(self.saved_img_features, dim=0)
            #         txt_tensor = torch.cat(self.saved_txt_features, dim=0)
                    
            #         # 使用时间戳或随机数避免覆盖
            #         import uuid
            #         uid = uuid.uuid4().hex[:6]
            #         img_path = os.path.join(self.feature_save_dir, f"task_{task_id}_img_{uid}.pt")
            #         txt_path = os.path.join(self.feature_save_dir, f"task_{task_id}_txt_{uid}.pt")
                    
            #         torch.save(img_tensor, img_path)
            #         torch.save(txt_tensor, txt_path)
                    
            #         # 【新增】控制台输出，确认保存成功
            #         print(f"\n[Feature Save] Success! Task {task_id} saved {img_tensor.shape[0]} samples. Path: {img_path}")
                    
            #         self.saved_img_features = []
            #         self.saved_txt_features =[]


        # =========================
        # ----- TEST MODE ---------
        # =========================
        else:
            valid_task_ids = self._infer_valid_task_ids()
            routing_mode = self._resolve_eval_modality_routing_mode()
            route_out = self.compute_modality_route(
                image_guide_features=image_guide_features,
                text_guide_features=text_guide_features,
                valid_task_ids=valid_task_ids,
                routing_mode=routing_mode,
            )
            p_d = route_out["p_d"]
            top_idx = torch.argmax(p_d, dim=-1)
            selected_task_ids = torch.tensor(valid_task_ids, device=p_d.device, dtype=torch.long)[top_idx]
            expert_weight = torch.zeros(
                p_d.size(0),
                self.expert_num,
                device=p_d.device,
                dtype=p_d.dtype,
            )
            expert_weight.scatter_(1, selected_task_ids.unsqueeze(-1), 1.0)
            self._apply_expert_weight(expert_weight)

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
    """
    将向量映射到庞加莱球内。
    vector: [1, D] 均值/特征向量
    var_sum: float, 方差的和 (Expert用). 如果为None (测试样本用), 则默认方差为0
    scale: float, 控制方差对半径的压缩力度
    """
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
    """
    计算两个庞加莱球内点的距离
    u, v: [1, D]
    """
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

