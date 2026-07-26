# 作用：记录高斯统计量初始化、训练时在线更新 mean/var/count，以及推理时基于高斯相似度设置 expert_weight 的参考代码片段。
# 注意：该文件是从模型实现中摘出的片段，不是可直接独立运行的 Python 模块。
# 初始化部分
hidden_dim = 768
        self.image_mean = nn.ParameterList([
            nn.Parameter(torch.zeros(1, hidden_dim)) for _ in range(self.expert_num)
        ])

        self.image_var = nn.ParameterList([
            nn.Parameter(torch.zeros(1, hidden_dim)) for _ in range(self.expert_num)
        ])

        # count 是标量，用 float（而不是 bfloat16 更稳）
        self.image_count = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.expert_num)
        ])

        # ---- text mean/var/count ----
        self.text_mean = nn.ParameterList([
            nn.Parameter(torch.zeros(1, hidden_dim)) for _ in range(self.expert_num)
        ])

        self.text_var = nn.ParameterList([
            nn.Parameter(torch.zeros(1, hidden_dim)) for _ in range(self.expert_num)
        ])

        self.text_count = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.expert_num)
        ])


# 训练和测试部分
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

            for t in range(self.expert_num):

                # -------- image Gaussian similarity --------
                mean_i = self.image_mean[t]         # [1, D]
                var_i  = self.image_var[t] + 1e-6   # [1, D]

                diff = img - mean_i
                sim_i = - (diff.pow(2) / var_i).sum().item()
                image_sim.append(sim_i)

                # -------- text Gaussian similarity --------
                mean_t = self.text_mean[t]
                var_t  = self.text_var[t] + 1e-6

                diff = txt - mean_t
                sim_t = - (diff.pow(2) / var_t).sum().item()
                text_sim.append(sim_t)

            image_sim = np.array(image_sim)
            text_sim = np.array(text_sim)

            sim = (image_sim + text_sim) / 2.0
            sim_tensor = torch.tensor(sim, dtype=torch.float32)

            sim_softmax = F.softmax(sim_tensor / 0.1, dim=0)
            compute_expert_weight = sim_softmax.tolist()

            proj_names = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ]
            for proj_name in proj_names:
                if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    proj_layer = getattr(self.model.layers[-1].self_attn, proj_name)
                else:
                    proj_layer = getattr(self.model.layers[-1].mlp, proj_name)

                proj_layer.expert_weight = compute_expert_weight
