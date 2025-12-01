# 加入预热策略的模型

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict

# =============================================================================
# 这里是你原来 modelTest9.py 中的所有辅助模块，无需任何改动
# ----------------------- 通用基础模块 -------------------------
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,),
                 dilation=(1,), if_bias=False, relu=True, same_padding=True, bn=True):
        super().__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=p0, dilation=dilation, bias=True if if_bias else False)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return F.dropout(x, 0.2, training=self.training)

class LiquidNeuralLayer1D(nn.Module):
    """
    修复点：
    - 原实现每一步都从全零/初始 hidden 取 h_prev，未把 h_current 传给下一步 → 无时序依赖。
    - 现在在循环中滚动更新 h_prev，使隐状态在时间维上传播。
    - 接口与返回形状保持不变。
    """
    def __init__(self, input_dim: int, output_dim: int, tau: float = 0.1, gamma: float = 0.5,
                 activation=F.relu):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.activation = activation

        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.gate = nn.Linear(input_dim + output_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, Cin]
        return: [B, L, Cout]
        """
        B, L, _ = x.shape
        h_prev = torch.zeros(B, self.output_dim, device=x.device, dtype=x.dtype)
        outs = []

        for t in range(L):
            x_t = x[:, t]  # [B, Cin]

            # 门控：根据 (x_t, h_prev) 决定保留/更新比例
            gate_input = torch.cat([x_t, h_prev], dim=-1)   # [B, Cin+Cout]
            gate_score = torch.sigmoid(self.gate(gate_input))  # [B, Cout]

            # 候选状态
            h_tilde = self.activation(self.linear(x_t))     # [B, Cout]

            # 液化更新（带泄露/惯性）
            h_current = (1.0 - self.tau) * h_prev + self.tau * h_tilde
            h_current = gate_score * h_current + (1.0 - gate_score) * (self.gamma * h_prev)

            outs.append(h_current)
            h_prev = h_current                               # 关键：把当前状态传给下一步

        return torch.stack(outs, dim=1)  # [B, L, Cout]

class DualPathFeatureProcessor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.global_path = nn.Sequential(nn.Linear(input_dim, output_dim), LiquidNeuralLayer1D(input_dim=output_dim, output_dim=output_dim))
        self.local_path = nn.Sequential(Conv1d(input_dim, output_dim, kernel_size=(3,), same_padding=True), Conv1d(output_dim, output_dim, kernel_size=(3,), same_padding=True))
        self.gate = nn.Linear(output_dim * 2, output_dim)
    def forward(self, x):
        global_feat = self.global_path(x.transpose(1, 2)).transpose(1, 2)
        local_feat = self.local_path(x)
        gate_input = torch.cat([global_feat, local_feat], dim=1).transpose(1, 2)
        gate_score = torch.sigmoid(self.gate(gate_input)).transpose(1, 2)
        return gate_score * global_feat + (1 - gate_score) * local_feat
class LNNWrapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lnn = LiquidNeuralLayer1D(input_dim, output_dim)
    def forward(self, x):
        return self.lnn(x.transpose(1, 2)).transpose(1, 2)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveTempCrossAttention(nn.Module):
    """
    修复点：
    - 原实现 q,k,v 都来自 query → 退化为自注意力。
    - 现在改为：q=Wq(query), k=Wk(key), v=Wv(value)，真正引入对侧模态信息。
    - 接口与返回形状保持不变。
    """
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim 必须是 num_heads 的整数倍"
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # 分开三套投影，确保跨模态
        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=True)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=True)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=True)

        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=True)

        # 自适应温度的可学习偏置，保留你原思路
        self.temp_bias = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        query: [B, Nq, C]
        key:   [B, Nk, C]
        value: [B, Nk, C]
        return: [B, Nq, C]
        """
        B, Nq, C = query.shape
        _, Nk, Ck = key.shape
        assert C == self.feature_dim and Ck == self.feature_dim, "通道数不匹配"

        # 线性投影
        q = self.q_proj(query)   # [B, Nq, C]
        k = self.k_proj(key)     # [B, Nk, C]
        v = self.v_proj(value)   # [B, Nk, C]

        # 分头
        q = q.view(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, Nq, d]
        k = k.view(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, Nk, d]
        v = v.view(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, Nk, d]

        # 自适应温度（保留你的设计：基于 q,k 的相似度估计一个温度）
        # 注意 temp 的范围与尺度，乘以 sqrt(d) 以稳定梯度
        with torch.no_grad():
            # 用均值向量粗估整体相似度
            q_mean = q.mean(dim=2)       # [B,h,d]
            k_mean = k.mean(dim=2)       # [B,h,d]
            sim = (q_mean * k_mean).sum(dim=-1).mean()  # 标量
        temp = torch.sigmoid(sim + self.temp_bias) * math.sqrt(self.head_dim) + 1e-6

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / temp          # [B,h,Nq,Nk]
        attn_probs = F.softmax(attn_logits, dim=-1)                         # [B,h,Nq,Nk]
        attn_out = torch.matmul(attn_probs, v)                               # [B,h,Nq,d]

        # 合并各头并线性映射回 C
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, Nq, C)  # [B,Nq,C]
        return self.out_proj(attn_out)                                       # [B,Nq,C]

class CrossAttnLNN2D(nn.Module):
    def __init__(self, hidden_dim, tau=0.15, gamma=0.6, dropout=0.2):
        super().__init__()
        self.hidden_dim, self.tau, self.gamma = hidden_dim, nn.Parameter(torch.tensor(tau)), gamma
        self.dropout, self.fc1, self.gate, self.fc2 = nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim * 2), nn.Linear(hidden_dim * 3, hidden_dim), nn.Linear(hidden_dim * 3, hidden_dim)
    def forward(self, x, attn_out, hidden_state=None):
        B, C = x.shape
        if hidden_state is None: hidden_state = torch.zeros(B, C, device=x.device)
        x_proj = self.dropout(F.relu(self.fc1(attn_out)))
        gate_input = torch.cat([x, attn_out, hidden_state], dim=1)
        gate_score = torch.sigmoid(self.gate(gate_input))
        new_hidden = (1 - self.tau) * hidden_state + self.tau * x_proj[:, :C]
        gated_hidden = gate_score * new_hidden + (1 - gate_score) * (self.gamma * hidden_state)
        return self.fc2(torch.cat([x_proj, gated_hidden], dim=1)), gated_hidden
class FusionCrossModalBlock(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.attn, self.norm1 = AdaptiveTempCrossAttention(feature_dim, num_heads), nn.LayerNorm(feature_dim)
        self.protein_to_rna_gate, self.rna_to_protein_gate = nn.Linear(feature_dim * 2, feature_dim), nn.Linear(feature_dim * 2, feature_dim)
        self.lnn, self.norm2 = CrossAttnLNN2D(hidden_dim=feature_dim), nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(nn.Linear(feature_dim, feature_dim * 2), nn.ReLU(), nn.Linear(feature_dim * 2, feature_dim))
    def forward(self, query, key, value, is_protein=True, hidden_state=None):
        attn_out = self.attn(query, key, value)
        query_feat = query.squeeze(1) if query.dim() == 3 else query
        gate_input = torch.cat([query_feat, attn_out.squeeze(1)], dim=-1)
        gate_score = torch.sigmoid(self.rna_to_protein_gate(gate_input) if is_protein else self.protein_to_rna_gate(gate_input))
        gated_feat = gate_score * attn_out.squeeze(1) + (1 - gate_score) * query_feat
        x = self.norm1(gated_feat)
        lnn_out, new_hidden = self.lnn(x, attn_out.squeeze(1), hidden_state)
        x = self.norm2(x + self.ffn(lnn_out))
        return x.unsqueeze(1) if query.dim() == 3 else x, new_hidden
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=512, num_heads=8, ff_hidden=1024, dropout=0.4):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_hidden), nn.ReLU(), nn.Linear(ff_hidden, embed_dim))
        self.norm1, self.norm2, self.dropout = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim), nn.Dropout(dropout)
    def forward(self, x):
        x = self.input_fc(x).unsqueeze(1)
        att_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(att_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x.squeeze(1)
class InteractingLayer(nn.Module):
    def __init__(self, embedding_size, use_res=True, scaling=False):
        super().__init__()
        self.att_size, self.use_res = embedding_size // 2, use_res
        self.W_Query, self.W_Key, self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size)), nn.Parameter(torch.Tensor(embedding_size, embedding_size)), nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        if self.use_res: self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for w in [self.W_Query, self.W_Key, self.W_Value] + ([self.W_Res] if self.use_res else []): nn.init.normal_(w, mean=0.0, std=0.05)
    def forward(self, inputs):
        querys, keys, values = [torch.tensordot(inputs, w, dims=([-1], [0])) for w in [self.W_Query, self.W_Key, self.W_Value]]
        querys, keys, values = [torch.stack(torch.split(t, self.att_size, dim=2)) for t in [querys, keys, values]]
        inner_product = torch.einsum('nbik,nbjk->nbij', querys, keys)
        att_scores = F.softmax(inner_product, dim=-1)
        result = torch.cat(torch.split(torch.matmul(att_scores, values), 1), dim=-1).squeeze(0)
        if self.use_res: result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        return F.relu(result)

class DaLNPI_warmup(nn.Module):
    def __init__(self, feat_size, embedding_size, dnn_feature_columns,
                 att_layer_num=2, num_heads=4):
        super().__init__()
        self.sparse_cols = list(filter(lambda x: x[1] == 'sparse', dnn_feature_columns))
        self.dense_cols = list(filter(lambda x: x[1] == 'dense', dnn_feature_columns))
        self.feature_index = defaultdict(int)
        start = 0
        for feat in feat_size:
            self.feature_index[feat] = start
            start += 1

        self.embedding_dic = nn.ModuleDict({
            feat[0]: nn.Embedding(feat_size[feat[0]], embedding_size)
            for feat in self.sparse_cols
        })

        self.hmcn = nn.ModuleDict({
            "branch0": Conv1d(len(self.dense_cols), 32, kernel_size=(1,)),
            "branch1": nn.Sequential(
                Conv1d(len(self.dense_cols), 32, kernel_size=(1,)),
                LNNWrapper(32, 32)
            ),
            "branch2": nn.Sequential(
                Conv1d(len(self.dense_cols), 32, kernel_size=(1,)),
                Conv1d(32, 32, kernel_size=(5,)),
                Conv1d(32, 32, kernel_size=(5,))
            ),
            "branch3": DualPathFeatureProcessor(len(self.dense_cols), 32)
        })
        self.hmcn_residual = nn.Linear(len(self.dense_cols), 32 * 4)

        # 关键改动：移除复杂的交互模块，替换为简单的MLP预测头
        protein_feature_dim = 32 * 4
        rna_feature_dim = embedding_size * len(self.sparse_cols)
        self.warmup_predictor = nn.Sequential(
            nn.Linear(protein_feature_dim + rna_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, X):
        X = X.to(next(self.parameters()).device)


        sparse_embed = [
            self.embedding_dic[feat[0]](X[:, self.feature_index[feat[0]]].long()).reshape(X.shape[0], 1, -1)
            for feat in self.sparse_cols
        ]
        # sparse_input = torch.cat(sparse_embed, dim=1) # 这一步在warmup中非必需
        rna_input = torch.flatten(torch.cat(sparse_embed, dim=1), start_dim=1)


        dense_values = [X[:, self.feature_index[feat[0]]].reshape(-1, 1) for feat in self.dense_cols]
        dense_input = torch.cat(dense_values, dim=1).unsqueeze(2)
        x0 = self.hmcn["branch0"](dense_input)
        x1 = self.hmcn["branch1"](dense_input)
        x2 = self.hmcn["branch2"](dense_input)
        x3 = self.hmcn["branch3"](dense_input)
        protein_input = torch.cat([x0, x1, x2, x3], dim=1).squeeze(2)
        protein_input = protein_input + self.hmcn_residual(dense_input.squeeze(2))

        # 3. 关键改动：使用简单的MLP进行预测，不经过复杂的交互模块
        combined_features = torch.cat([protein_input, rna_input], dim=1)
        return torch.sigmoid(self.warmup_predictor(combined_features))


class DaLNPI(nn.Module):
    def __init__(self, feat_size, embedding_size, dnn_feature_columns,
                 att_layer_num=2, num_heads=4):
        super().__init__()
        # 特征分类与索引
        self.sparse_cols = list(filter(lambda x: x[1] == 'sparse', dnn_feature_columns))
        self.dense_cols = list(filter(lambda x: x[1] == 'dense', dnn_feature_columns))
        self.feature_index = defaultdict(int)
        start = 0
        for feat in feat_size:
            self.feature_index[feat] = start
            start += 1

        # 稀疏特征嵌入
        self.embedding_dic = nn.ModuleDict({
            feat[0]: nn.Embedding(feat_size[feat[0]], embedding_size)
            for feat in self.sparse_cols
        })

        # 多尺度特征提取（方案1 HMCN + 方案3双路径）
        self.hmcn = nn.ModuleDict({
            "branch0": Conv1d(len(self.dense_cols), 32, kernel_size=(1,)),
            "branch1": nn.Sequential(
                Conv1d(len(self.dense_cols), 32, kernel_size=(1,)),
                LNNWrapper(32, 32)
            ),
            "branch2": nn.Sequential(
                Conv1d(len(self.dense_cols), 32, kernel_size=(1,)),
                Conv1d(32, 32, kernel_size=(5,)),
                Conv1d(32, 32, kernel_size=(5,))
            ),
            "branch3": DualPathFeatureProcessor(len(self.dense_cols), 32)
        })
        self.hmcn_residual = nn.Linear(len(self.dense_cols), 32 * 4)

        # 跨模态交互
        self.protein_fc = nn.Linear(32 * 4, 512)
        self.rna_fc = nn.Linear(embedding_size * len(self.sparse_cols), 512)
        self.cross_blocks = nn.ModuleList([
            FusionCrossModalBlock(512, num_heads) for _ in range(att_layer_num)
        ])

        # 其他辅助模块
        self.feature_transformer = FeatureTransformer(
            input_dim=len(self.dense_cols) + embedding_size * len(self.sparse_cols),
            embed_dim=512
        )
        self.int_layers = nn.ModuleList([
            InteractingLayer(embedding_size) for _ in range(att_layer_num)
        ])

        # 输出层
        final_dim = 512 + embedding_size * len(self.sparse_cols) + 512 * 2
        self.dnn_linear = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, X):
        X = X.to(next(self.parameters()).device)

        # 1. 稀疏特征处理
        sparse_embed = [
            self.embedding_dic[feat[0]](X[:, self.feature_index[feat[0]]].long()).reshape(X.shape[0], 1, -1)
            for feat in self.sparse_cols
        ]
        sparse_input = torch.cat(sparse_embed, dim=1)
        rna_input = torch.flatten(sparse_input, start_dim=1)

        # 2. 密集特征处理
        dense_values = [X[:, self.feature_index[feat[0]]].reshape(-1, 1) for feat in self.dense_cols]
        dense_input = torch.cat(dense_values, dim=1).unsqueeze(2)
        x0 = self.hmcn["branch0"](dense_input)
        x1 = self.hmcn["branch1"](dense_input)
        x2 = self.hmcn["branch2"](dense_input)
        x3 = self.hmcn["branch3"](dense_input)
        protein_input = torch.cat([x0, x1, x2, x3], dim=1).squeeze(2)
        protein_input = protein_input + self.hmcn_residual(dense_input.squeeze(2))

        # 3. 跨模态交互
        pro = F.relu(self.protein_fc(protein_input)).unsqueeze(1)
        rna = F.relu(self.rna_fc(rna_input)).unsqueeze(1)
        pro_hidden, rna_hidden = None, None
        for block in self.cross_blocks:
            pro, pro_hidden = block(pro, rna, rna, is_protein=True, hidden_state=pro_hidden)
            rna, rna_hidden = block(rna, pro, pro, is_protein=False, hidden_state=rna_hidden)
        protein_feat, rna_feat = pro.squeeze(1), rna.squeeze(1)

        # 4. 特征融合与预测
        att_input = sparse_input
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)
        dnn_input = torch.cat((dense_input.squeeze(2), rna_input), dim=1)
        deep_out = self.feature_transformer(dnn_input)
        stack_out = torch.cat((att_output, deep_out, protein_feat, rna_feat), dim=-1)
        return torch.sigmoid(self.dnn_linear(stack_out))