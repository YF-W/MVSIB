import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

class Loss(nn.Module):
    def __init__(self, batch_size, num_clusters, temperature_l, temperature_f,
                 R_max=1.0, margin=0.5, bound_weight=0.1,
                 global_minority_ratio=0.5, num_gmm_components=2, num_pseudo_samples=5,
                 mmd_weight=1.0, excl_weight=1.0, excl_sigma=2.0):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.temperature_l = temperature_l
        self.temperature_f = temperature_f
        self.similarity = nn.CosineSimilarity(dim=2)
        # 边界约束超参
        self.R_max = R_max
        self.margin = margin
        self.bound_weight = bound_weight

        # 全局少数判定超参
        self.global_minority_ratio = global_minority_ratio

        # GMM 伪样本生成超参
        self.num_gmm_components = num_gmm_components
        self.num_pseudo_samples = num_pseudo_samples

        # MMD 与排斥核超参
        self.mmd_weight = mmd_weight
        self.excl_weight = excl_weight
        self.excl_sigma = excl_sigma

        # 自适应权重相关：三项 [mmd, excl, bound] 的 EMA
        self.register_buffer('loss_ema', torch.ones(3))
        self.ema_momentum = 0.9
        self.weight_alpha = 1.0
        # 损失函数实例
        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss()

    def feature_loss(self, zi, z, w, y_pse):
        """
        zi: [N, D], z: [N, D]
        w:  [N, N] integer mask (0/1)
        y_pse: [N, N] float pseudo‐label mask (e.g. 1.0 for positives, 0.1 for FN, 0.0 for true negatives)
        """
        N = z.size(0)
        device = zi.device

        # —— 1) 计算距离矩阵 ——
        cross_view_distance = self.similarity(
            zi.unsqueeze(1),  # [N,1,D]
            z.unsqueeze(0)  # [1,N,D]
        ) / self.temperature_f  # → [N,N]
        inter_view_distance = self.similarity(
            zi.unsqueeze(1),
            zi.unsqueeze(0)
        ) / self.temperature_f  # → [N,N]

        # —— 2) 构造正样本 mask ——
        # 包含 w==1 的对，以及对角(自对)
        w_bool = w.to(device).bool()  # [N,N]
        eye = torch.eye(N, dtype=torch.bool, device=device)
        w_mask = w_bool | eye  # [N,N] 布尔

        # —— 3) 真实的“软”权重 y_pse ——
        y_pse = y_pse.to(device).float()  # [N,N]

        # —— 4) positive loss ——
        pos_mask = w_mask & (y_pse > 0)  # 只有 y_pse>0 才算正样本
        # 按 y_pse 乘以距离
        positive_term = (pos_mask.float() * y_pse * cross_view_distance).sum()
        positive_loss = -positive_term  # 保持原来 “-sum(...)”

        # —— 5) negative loss ——
        neg_mask = (~w_mask) & (y_pse == 0)
        SMALL_NUM = torch.log(torch.tensor(1e-45, device=device))

        neg_cross = (neg_mask.float() * cross_view_distance)
        neg_cross = neg_cross.masked_fill(neg_cross == 0, SMALL_NUM)

        neg_inter = (neg_mask.float() * inter_view_distance)
        neg_inter = neg_inter.masked_fill(neg_inter == 0, SMALL_NUM)

        # 拼到一起，再除一次 temperature（可根据实际 remove）
        neg_sim = torch.cat([neg_inter, neg_cross], dim=1) / self.temperature_f
        neg_loss = torch.logsumexp(neg_sim, dim=1).sum()

        # —— 6) 最终归一 ——
        return (positive_loss + neg_loss) / N

    def compute_mmd_loss(self, real, pseudo):
        n, m = real.size(0), pseudo.size(0)
        if n < 2 or m < 2:
            return torch.tensor(0., device=real.device)
        K_xx = torch.exp(-torch.cdist(real, real, p=2) ** 2 / (2 * self.excl_sigma ** 2))
        K_yy = torch.exp(-torch.cdist(pseudo, pseudo, p=2) ** 2 / (2 * self.excl_sigma ** 2))
        K_xy = torch.exp(-torch.cdist(real, pseudo, p=2) ** 2 / (2 * self.excl_sigma ** 2))
        sum_xx = (K_xx.sum() - K_xx.diag().sum()) / (n * (n - 1))
        sum_yy = (K_yy.sum() - K_yy.diag().sum()) / (m * (m - 1))
        sum_xy = 2 * K_xy.sum() / (n * m)
        return sum_xx + sum_yy - sum_xy

    def compute_exclusion_loss(self, pseudos):
        k = pseudos.size(0)
        if k < 2:
            return torch.tensor(0., device=pseudos.device)
        d = torch.cdist(pseudos, pseudos, p=2)
        mask = ~torch.eye(k, dtype=torch.bool, device=d.device)
        kr = torch.exp(-d[mask] ** 2 / (2 * self.excl_sigma ** 2))
        return kr.sum()

    def compute_cluster_loss(self,
                             q_centers,  # [L, D]
                             k_centers,  # [L, D]
                             psedo_labels_batch,  # [B]
                             features_batch=None,  # [B, D] or None
                             global_minority_mask=None,
                             return_mmd_excl=False):
        """
        如果 features_batch or global_minority_mask is None: 只做原始 InfoNCE（不带伪样本）。
        否则：使用全局+批次判定少数簇，并用 GMM 拟合生成 num_pseudo_samples 个伪样本，再平均得到 ψ_k。
        """
        total_mmd = 0.0
        total_excl = 0.0
        device = q_centers.device
        L, D = q_centers.shape

        # 1) 计算基础相似度矩阵 d_q
        d_q_raw = q_centers.mm(q_centers.T)
        with torch.no_grad():
            d_kdiag = (q_centers * k_centers).sum(dim=1) / self.temperature_l
        I_L = torch.eye(L, device=device, dtype=d_q_raw.dtype)
        d_q = (d_q_raw / self.temperature_l) * (1 - I_L) + torch.diag(d_kdiag)

        # 原始 InfoNCE 分支
        if features_batch is None or global_minority_mask is None:
            counts = torch.bincount(psedo_labels_batch, minlength=self.num_clusters).float()
            unique_labels = torch.unique(psedo_labels_batch)
            mask_unique = torch.zeros(self.num_clusters, device=device, dtype=torch.bool)
            mask_unique[unique_labels] = True
            zero_classes = torch.arange(self.num_clusters, device=device)[~mask_unique]

            losses = []
            eye = torch.eye(L, device=device, dtype=torch.bool)
            for k in range(L):
                if k in zero_classes:
                    continue
                pos = torch.exp(d_q[k, k])
                neg = (torch.exp(d_q[k, :]) * (~eye[k]).float()).sum()
                losses.append(-torch.log(pos / (pos + neg + 1e-8)))
            num_nonzero = L - zero_classes.numel()
            return torch.stack(losses).sum() / num_nonzero if num_nonzero > 0 else torch.tensor(0., device=device)

        # ——————— 走伪样本分支 ———————
        counts = torch.bincount(psedo_labels_batch, minlength=self.num_clusters).float().to(device)
        unique_labels = torch.unique(psedo_labels_batch).to(device)
        mask_unique = torch.zeros(self.num_clusters, device=device, dtype=torch.bool)
        mask_unique[unique_labels] = True
        zero_classes = torch.arange(self.num_clusters, device=device)[~mask_unique]
        batch_mask = counts > 0
        global_mask = global_minority_mask.to(device=device)
        batch_minority_mask = batch_mask & global_mask
        minority_indices = batch_minority_mask.nonzero(as_tuple=False).view(-1)

        # 伪样本生成（GMM）
        pseudos = torch.zeros((L, D), device=device, dtype=q_centers.dtype)
        mu_minority = {}
        for k_tensor in minority_indices:
            k = k_tensor.item()
            feats_k = features_batch[psedo_labels_batch == k]
            n_k = feats_k.size(0)
            if n_k <= 1:
                pseudos[k] = k_centers[k]
                mu_minority[k] = k_centers[k]
            else:
                mu_k = feats_k.mean(dim=0)
                mu_minority[k] = mu_k
                xc = feats_k - mu_k.unsqueeze(0)
                cov_k = xc.T @ xc / (n_k - 1 + 1e-6)
                cov_p = (self.num_pseudo_samples / n_k) * cov_k + 1e-6 * torch.eye(D, device=device)
                try:
                    mvn = torch.distributions.MultivariateNormal(mu_k, covariance_matrix=cov_p)
                    pseudos[k] = mvn.rsample()
                except:
                    pseudos[k] = k_centers[k]

        # 边界约束
        bound_loss = torch.tensor(0., device=device)
        if self.bound_weight > 0 and minority_indices.numel() > 0:
            for k in minority_indices.tolist():
                psi_k = pseudos[k]
                mu_k = mu_minority[k]
                d_self = torch.norm(psi_k - mu_k, p=2)
                dists = torch.norm(q_centers - psi_k.unsqueeze(0), dim=1)
                mask_self = torch.arange(L, device=device) == k
                d_other = dists.masked_fill(mask_self, float('inf')).min()
                bound_loss += F.relu(d_self - self.R_max) + F.relu(self.margin - d_other)

        # 簇级 InfoNCE + 伪样本
        if minority_indices.numel() > 0:
            q_exp = q_centers.unsqueeze(1).expand(L, L, D)
            psi_exp = pseudos.unsqueeze(0).expand(L, L, D)
            sim_pseudo_all = F.cosine_similarity(q_exp, psi_exp, dim=2) / self.temperature_l
        else:
            sim_pseudo_all = torch.zeros((L, L), device=device)
        eye_mask = torch.eye(L, device=device, dtype=torch.bool)
        losses = []
        for k in range(L):
            if k in zero_classes:
                continue
            pos_main = torch.exp(d_q[k, k])
            if batch_minority_mask[k]:
                pos_main = pos_main + torch.exp(sim_pseudo_all[k, k])
            dq_row = d_q[k]
            mask_neg = (~eye_mask[k]).float()
            neg_real = (torch.exp(dq_row) * mask_neg).sum()
            neg_pseudo = 0
            if minority_indices.numel() > 0:
                mask_minor = batch_minority_mask.float().to(device) * mask_neg
                neg_pseudo = (torch.exp(sim_pseudo_all[k]) * mask_minor).sum()
            denom = pos_main + neg_real + neg_pseudo + 1e-8
            losses.append(-torch.log(pos_main / denom))
        cluster_loss = torch.stack(losses).sum() / (
                    L - zero_classes.numel()) if L - zero_classes.numel() > 0 else torch.tensor(0., device=device)

        # 合并边界约束
        if self.bound_weight > 0:
            cluster_loss = cluster_loss + self.bound_weight * bound_loss
        if features_batch is not None and minority_indices.numel() > 1:
            # 批次中少数簇真实样本
            device = psedo_labels_batch.device
            mask_real = torch.zeros_like(psedo_labels_batch, dtype=torch.bool).to(device)
            for k in minority_indices:
                k = k.item()
                mask_real |= (psedo_labels_batch == k)

            real_feats = features_batch[mask_real]
            pseudo_feats = pseudos[minority_indices]
            total_mmd = self.compute_mmd_loss(real_feats, pseudo_feats)
            total_excl = self.compute_exclusion_loss(pseudo_feats)
            cluster_loss = cluster_loss + self.mmd_weight * total_mmd + self.excl_weight * total_excl

        if return_mmd_excl:
            return cluster_loss, total_mmd, total_excl
        else:
            return cluster_loss




    # —— 模块2：不确定度 MLP 回归损失 —— #
    def uncertainty_regression_loss(self, u_hat, u_true):

        return self.mse(u_hat, u_true.detach())


    # —— 模块4：加权 InfoNCE —— #
    def weighted_info_nce(self, reps, S, temperature):
        sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2) / temperature  # (N, N)

        # 使用 S 直接对所有样本对进行加权
        exp_sim = torch.exp(sim)  # (N, N)
        weighted_sim = exp_sim * S  # 每个样本对的加权相似度 (N, N)

        # 计算分子和分母
        num = torch.sum(weighted_sim, dim=1)  # 对每个样本的加权相似度求和
        den = torch.sum(exp_sim, dim=1)  # 对每个样本的相似度求和

        # 避免除零错误
        eps = 1e-8
        num = torch.clamp(num, min=eps)
        den = torch.clamp(den, min=eps)

        # 计算最终损失
        loss = -torch.mean(torch.log(num / den))  # 计算 InfoNCE 损失

        return loss

    def cross_view_weighted_loss(self,
                                 um,          # UncertaintyModule 实例
                                 zs_list,     # list of view representations
                                 common_z,    # consensus representation
                                 memberships, # list of Tensors[N×L]
                                 batch_psedo_label,
                                 temperature=None):
        """
        一体化接口，计算跨视图 & 共识空间的加权 InfoNCE 总和。
        """
        if temperature is None:
            temperature = self.temperature_f

        # 1) 计算一致性分数矩阵 S
        S = um.compute_consistency_scores(memberships, batch_psedo_label)

        # 2) 对每个视图与共识分别做加权 InfoNCE
        loss = 0.0
        for z in zs_list:
            loss += self.weighted_info_nce(z, S, temperature)
        loss += self.weighted_info_nce(common_z, S, temperature)
        return loss
