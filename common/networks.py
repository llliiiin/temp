"""
这个文件用来存储不同的网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent.helpers import SinusoidalPosEmb



class MLP(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):  # n_actions是当前agent的动作维度大小
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # input layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # hidden layer
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # output layer

    def forward(self, obs):
        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out                  # n_actions维动作所对应的值

class MLP_Norm(nn.Module):
    """
    Enhanced actor network combining LayerNorm, dropout.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, dropout=0.1, num_layers=2):
        super().__init__()
        self.fc_in = nn.Linear(state_dim, hidden_dim)

        # Build a sequence of residual-attention blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict({
                "ln1": nn.LayerNorm(hidden_dim),
                # "attn": nn.MultiheadAttention(
                #     embed_dim=hidden_dim,
                #     num_heads=n_heads,
                #     batch_first=True,
                #     dropout=dropout
                # ),
                # "ln2": nn.LayerNorm(hidden_dim),
                "fc_mid": nn.Linear(hidden_dim, hidden_dim),
                "dropout": nn.Dropout(dropout),
            })
            self.blocks.append(block)

        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, state_dim) or (state_dim,)
        returns: logits Tensor of shape (batch_size, action_dim) or (action_dim,)
        """
        # Handle unbatched input
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        # Initial embedding
        h = F.relu(self.fc_in(x))  # (B, H)

        # Pass through each residual-attention block
        for block in self.blocks:
            # 1. Self-attention with residual
            h_norm = block["ln1"](h)
            # reshape for attention: (B, 1, H)
            # attn_out, _ = block["attn"](
            #     h_norm.unsqueeze(1),  # query
            #     h_norm.unsqueeze(1),  # key
            #     h_norm.unsqueeze(1)   # value
            # )
            # h = h + attn_out.squeeze(1)
            #
            # # 2. MLP with residual
            # h_norm2 = block["ln2"](h)
            h_mid = F.relu(block["fc_mid"](h_norm))
            h = h + block["dropout"](h_mid)

        # Final output logits
        logits = self.fc_out(h)  # (B, action_dim)
        if squeeze_output:
            return logits.squeeze(0)
        return logits


class En_MLP(nn.Module):
    """
    Enhanced actor network combining residual connections, LayerNorm, Multi-Head Self-Attention,
    dropout, and stacking of multiple such blocks.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, n_heads=4, dropout=0.1, num_layers=2):
        super().__init__()
        self.fc_in = nn.Linear(state_dim, hidden_dim)

        # Build a sequence of residual-attention blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict({
                "ln1": nn.LayerNorm(hidden_dim),
                "attn": nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=n_heads,
                    batch_first=True,
                    dropout=dropout
                ),
                "ln2": nn.LayerNorm(hidden_dim),
                "fc_mid": nn.Linear(hidden_dim, hidden_dim),
                "dropout": nn.Dropout(dropout),
            })
            self.blocks.append(block)

        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, state_dim) or (state_dim,)
        returns: logits Tensor of shape (batch_size, action_dim) or (action_dim,)
        """
        # Handle unbatched input
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        # Initial embedding
        h = F.relu(self.fc_in(x))  # (B, H)

        # Pass through each residual-attention block
        for block in self.blocks:
            # 1. Self-attention with residual
            h_norm = block["ln1"](h)
            # reshape for attention: (B, 1, H)
            attn_out, _ = block["attn"](
                h_norm.unsqueeze(1),  # query
                h_norm.unsqueeze(1),  # key
                h_norm.unsqueeze(1)   # value
            )
            h = h + attn_out.squeeze(1)

            # 2. MLP with residual
            h_norm2 = block["ln2"](h)
            h_mid = F.relu(block["fc_mid"](h_norm2))
            h = h + block["dropout"](h_mid)

        # Final output logits
        logits = self.fc_out(h)  # (B, action_dim)
        if squeeze_output:
            return logits.squeeze(0)
        return logits


class Feature_Attn_MLP(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, n_heads=4, dropout=0.1):
        """
        把每个特征维度当 token，做注意力。
        state_dim: 原始特征维度 D
        """
        super().__init__()
        # 1. 特征嵌入：把每个标量 xi → Hi 维向量
        self.embed = nn.Linear(1, hidden_dim)
        # 2. 多头注意力，针对 D 个 token
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=n_heads,
                                          batch_first=True,
                                          dropout=dropout)
        # 3. 输出头
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        # x: (B, D) 或 (D,)
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        B, D = x.shape
        # 1. 每个特征独立嵌入
        # 把 (B, D) → (B, D, 1) → (B*D,1) 再映射 → (B*D, H) → (B, D, H)
        e = x.unsqueeze(-1).view(-1, 1)              # (B*D, 1)
        e = F.relu(self.embed(e))                   # (B*D, H)
        e = e.view(B, D, -1)                        # (B, D, H)

        # 2. 跨特征注意力
        attn_out, _ = self.attn(e, e, e)            # (B, D, H)

        # 3. 池化（取平均）→ (B, H)
        pooled = attn_out.mean(dim=1)

        # 4. 输出
        logits = self.fc(pooled)                    # (B, action_dim)
        return logits.squeeze(0) if squeeze else logits



class En_Feature_Attn_MLP(nn.Module):
    """
    Enhanced Feature_Attn_MLP integrating:
      1. Residual connections
      2. LayerNorm before each sub-layer
      3. Multi-Head Self-Attention across feature tokens
      4. Two-phase Feed-Forward network with residual
      5. Dropout for regularization
      6. Stacking of multiple such blocks
    """
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        n_heads = 4,
        dropout = 0.1,
        num_layers = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # 1. Embed each scalar feature into hidden_dim
        self.embed = nn.Linear(1, hidden_dim)

        # 2. Build residual-attention + FFN blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict({
                # Attention sub-layer
                "ln_attn": nn.LayerNorm(hidden_dim),
                "attn": nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=n_heads,
                    batch_first=True,
                    dropout=dropout
                ),
                "dropout_attn": nn.Dropout(dropout),
                # Feed-forward sub-layer
                "ln_ffn": nn.LayerNorm(hidden_dim),
                "fc_ffn": nn.Linear(hidden_dim, hidden_dim),
                "dropout_ffn": nn.Dropout(dropout),
            })
            self.blocks.append(block)

        # 3. Output head
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, state_dim) or (state_dim,)
        returns: logits of shape (batch_size, action_dim) or (action_dim,)
        """
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        B, D = x.shape  # batch_size, feature_count

        # Embed features to tokens: (B, D, 1) -> (B*D, 1) -> (B*D, H) -> (B, D, H)
        e = x.unsqueeze(-1).view(-1, 1)
        e = F.relu(self.embed(e))
        h = e.view(B, D, self.hidden_dim)

        # Pass through each residual block
        for block in self.blocks:
            # 1) Multi-Head Self-Attention with residual
            h_norm = block["ln_attn"](h)
            attn_out, _ = block["attn"](h_norm, h_norm, h_norm)
            attn_out = block["dropout_attn"](attn_out)
            h = h + attn_out

            # 2) Feed-Forward with residual
            h_norm2 = block["ln_ffn"](h)
            ffn_out = F.relu(block["fc_ffn"](h_norm2))
            ffn_out = block["dropout_ffn"](ffn_out)
            h = h + ffn_out

        # Pool across feature tokens and output
        pooled = h.mean(dim=1)              # (B, H)
        out = self.ln_out(pooled)
        out = F.relu(out)
        logits = self.fc_out(out)           # (B, action_dim)

        return logits.squeeze(0) if squeeze else logits# 类似transformer的架构



class RNN(nn.Module):
    def __init__(self,
                 state_dim,     # 单步状态维度
                 hidden_dim,    # 隐藏层维度
                 action_dim,    # 动作空间维度
                 rnn_type='GRU',
                 num_layers=1,
                 dropout=0.1):
        super().__init__()
        self.rnn_type = rnn_type
        # self.norm = nn.LayerNorm(hidden_dim)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=False,      # 输入维度 [seq_len, batch, state_dim]
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=state_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=False,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, seq):
        """
        seq: [seq_len, state_dim]
        return: [action_dim]
        """
        seq = seq.unsqueeze(1)  # 添加 batch 维度 → [seq_len, 1, state_dim]
        rnn_out, hidden = self.rnn(seq)
        # rnn_out: 所有时间步的输出，形状 [seq_len, batch_size, hidden_dim]
        # hidden: 所有层最后时刻的隐藏状态，形状 [num_layers, batch_size, hidden_dim]

        # 最后一时间步的输出
        final_output = rnn_out[-1, 0, :]  # [hidden_dim]
        # final_output = F.relu(self.norm(final_output))
        action_logits = self.fc(final_output)  # [action_dim]

        return action_logits


class RNN_AddAtte(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, rnn_type='GRU', num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(state_dim, hidden_dim, num_layers,
                              batch_first=False,
                              dropout=dropout if num_layers > 1 else 0.0)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(state_dim, hidden_dim, num_layers,
                               batch_first=False,
                               dropout=dropout if num_layers > 1 else 0.0)
        else:
            raise ValueError("Unsupported RNN type")

        # 注意力模块
        self.attn_fc = nn.Linear(hidden_dim, 1)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # 输出层
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, seq):
        """
        seq: [seq_len, state_dim] without batch
        return: action logits
        """
        seq = seq.unsqueeze(1)  # [seq_len, 1, state_dim]
        rnn_out, _ = self.rnn(seq)  # [seq_len, 1, hidden_dim]

        attn_weights = self.attn_fc(rnn_out)           # [seq_len, 1, 1]
        attn_weights = F.softmax(attn_weights, dim=0)  # 时间维度 softmax
        attn_applied = torch.sum(attn_weights * rnn_out, dim=0)  # [1, hidden_dim]
        attn_applied = self.attn_dropout(attn_applied)            # Dropout
        attn_applied = self.attn_norm(attn_applied)               # LayerNorm
        final_hidden = attn_applied.squeeze(0)                    # [hidden_dim]

        action_logits = self.fc(final_hidden)  # [action_dim]
        return action_logits



# class RNN_Multihead_SelfAtte(nn.Module):
#     ''' RNN --> Attention'''
#     def __init__(self, state_dim, embed_dim, hidden_dim, action_dim,
#                  rnn_type='GRU', num_layers=1, num_heads=4, dropout=0.1):
#         super().__init__()
#         self.rnn_type = rnn_type.upper()
#         self.embed_dim = embed_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#
#         # 1. 状态嵌入层
#         self.state_encoder = StateEmbedding(state_dim, embed_dim)
#
#         # 2. RNN 模块（接收嵌入后的输入）
#         if self.rnn_type == 'GRU':
#             self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers,
#                               batch_first=False, dropout=dropout if num_layers > 1 else 0.0)
#         elif self.rnn_type == 'LSTM':
#             self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,
#                                batch_first=False, dropout=dropout if num_layers > 1 else 0.0)
#         else:
#             raise ValueError(f"Unsupported RNN type: {rnn_type}")
#
#         # 3. 多头注意力层（基于 RNN 输出）
#         self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
#                                           num_heads=num_heads,
#                                           dropout=dropout,
#                                           batch_first=False)
#
#         # 4. Learnable query 聚合向量（用于 attention pooling）
#         self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
#
#         # 5. 正则化与激活
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(dropout)
#
#         # 6. 输出层
#         self.fc = nn.Linear(hidden_dim, action_dim)
#
#     def forward(self, seq):
#         """
#         seq: [seq_len, state_dim] without batch
#         return: [action_dim]
#         """
#         seq = seq.unsqueeze(1)  # [seq_len, 1, state_dim]
#
#         # 状态嵌入
#         embedded_seq = self.state_encoder(seq.squeeze(1))  # [seq_len, embed_dim]
#         embedded_seq = embedded_seq.unsqueeze(1)  # ➜ [seq_len, 1, embed_dim]
#
#         # 初始化隐藏状态（可选）
#         if self.rnn_type == 'LSTM':
#             h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             c0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             rnn_out, _ = self.rnn(embedded_seq, (h0, c0))
#         else:
#             h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             rnn_out, _ = self.rnn(embedded_seq, h0)  # [seq_len, 1, hidden_dim]
#
#         # 多头注意力处理（Q = K = V = rnn_out）
#         attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)
#         attn_out = self.norm(attn_out + rnn_out)
#
#         # 使用 learnable query 聚合整个序列
#         pooled_out, _ = self.attn(self.query, attn_out, attn_out)
#         final_hidden = pooled_out.squeeze(0).squeeze(0)  # [hidden_dim]
#
#         # Dropout + 激活 + 输出
#         final_hidden = self.dropout(final_hidden)
#         final_hidden = F.relu(final_hidden)
#         action_logits = self.fc(final_hidden)  # [action_dim]
#
#         return action_logits


# class RNN_Multihead_SelfAtte(nn.Module):
#     """
#     方案 A：Embedding → Self-Attention → RNN → Query Pooling → FC
#     """
#     def __init__(
#         self,
#         state_dim: int,
#         embed_dim: int,
#         hidden_dim: int,
#         action_dim: int,
#         rnn_type: str = "GRU",
#         num_layers: int = 1,
#         num_heads: int = 4,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.rnn_type = rnn_type.upper()
#         self.embed_dim = embed_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#
#         # 1. 状态嵌入
#         self.state_encoder = StateEmbedding(state_dim, embed_dim)
#
#         # 2. **先做自注意力**（基于 embed_dim）
#         self.pre_attn = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=False,
#         )
#         self.pre_norm = nn.LayerNorm(embed_dim)
#
#         # 3. RNN 接收注意力输出
#         if self.rnn_type == "GRU":
#             self.rnn = nn.GRU(
#                 embed_dim,
#                 hidden_dim,
#                 num_layers,
#                 batch_first=False,
#                 dropout=dropout if num_layers > 1 else 0.0,
#             )
#         elif self.rnn_type == "LSTM":
#             self.rnn = nn.LSTM(
#                 embed_dim,
#                 hidden_dim,
#                 num_layers,
#                 batch_first=False,
#                 dropout=dropout if num_layers > 1 else 0.0,
#             )
#         else:
#             raise ValueError(f"Unsupported RNN type: {rnn_type}")
#
#         # 4. Learnable-query pooling（hidden_dim）
#         self.pool_attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=False,
#         )
#         self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
#
#         # 5. 其他层
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_dim, action_dim)
#
#     def forward(self, seq: torch.Tensor) -> torch.Tensor:
#         # 补 batch 维： [T, 1, state_dim]
#         seq = seq.unsqueeze(1)
#
#         # ① 状态嵌入  [T, 1, embed_dim]
#         embedded = self.state_encoder(seq.squeeze(1)).unsqueeze(1)
#
#         # ② 自注意力（K=Q=V）  [T, 1, embed_dim]
#         attn_out, _ = self.pre_attn(embedded, embedded, embedded)
#         attn_out = self.pre_norm(attn_out + embedded)  # 残差
#
#         # ③ RNN
#         if self.rnn_type == "LSTM":
#             h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             c0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             rnn_out, _ = self.rnn(attn_out, (h0, c0))  # [T, 1, hidden_dim]
#         else:
#             h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             rnn_out, _ = self.rnn(attn_out, h0)        # [T, 1, hidden_dim]
#
#         # ④ Learnable-query pooling
#         pooled_out, _ = self.pool_attn(self.query, rnn_out, rnn_out)
#         final_hidden = pooled_out.squeeze(0).squeeze(0)  # [hidden_dim]
#
#         # ⑤ Dropout + ReLU + FC
#         final_hidden = F.relu(self.dropout(final_hidden))
#         action_logits = self.fc(final_hidden)             # [action_dim]
#         return action_logits



# class RNN_Multihead_SelfAtte(nn.Module):
#     """
#     方案 A（改）：Embedding → Self-Attention → RNN → AvgPool → FC
#     """
#     def __init__(
#         self,
#         state_dim: int,
#         embed_dim: int,
#         hidden_dim: int,
#         action_dim: int,
#         rnn_type: str = "LSTM",
#         num_layers: int = 1,
#         num_heads: int = 4,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.rnn_type = rnn_type.upper()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#
#         # 1. 状态嵌入
#         self.state_encoder = StateEmbedding(state_dim, embed_dim)
#
#         # 2. 前置自注意力
#         self.pre_attn = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=False,
#         )
#         self.pre_norm = nn.LayerNorm(embed_dim)
#
#         # 3. RNN
#         rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}.get(self.rnn_type)
#         if rnn_cls is None:
#             raise ValueError(f"Unsupported RNN type: {rnn_type}")
#         self.rnn = rnn_cls(
#             embed_dim,
#             hidden_dim,
#             num_layers,
#             batch_first=False,
#             dropout=dropout if num_layers > 1 else 0.0,
#         )
#
#         # 4. Dropout + FC
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_dim, action_dim)
#
#     def forward(self, seq: torch.Tensor) -> torch.Tensor:
#         """
#         seq : [T, state_dim]（无 batch 维）
#         """
#         seq = seq.unsqueeze(1)                               # [T, 1, state_dim]
#
#         # ① 嵌入
#         emb = self.state_encoder(seq.squeeze(1)).unsqueeze(1) # [T, 1, E]
#
#         # ② 自注意力
#         attn_out, _ = self.pre_attn(emb, emb, emb)
#         attn_out = self.pre_norm(attn_out + emb)              # [T, 1, E]
#
#         # ③ RNN
#         if self.rnn_type == "LSTM":
#             h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             c0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             rnn_out, _ = self.rnn(attn_out, (h0, c0))         # [T, 1, H]
#         else:
#             h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             rnn_out, _ = self.rnn(attn_out, h0)               # [T, 1, H]
#
#         # ④ **平均池化**  —— 时间维 T 上取均值
#         #    结果形状 [1, hidden_dim] ➜ squeeze → [hidden_dim]
#         # avg_hidden = rnn_out.mean(dim=0).squeeze(0)
#         max_hidden = rnn_out.max(dim=0).values.squeeze(0)
#
#         # ⑤ Dropout + ReLU + 输出
#         logits = self.fc(F.relu(self.dropout(max_hidden)))    # [action_dim]
#         return logits


# class RNN_Multihead_SelfAtte(nn.Module):
#     """
#     Embedding → Self-Attention → FFN(+残差) → RNN → MaxPool → FC
#     -----------------------------------------------------------------
#     FFN = Linear → ReLU → Dropout → Linear  (维度: embed_dim → 4×embed_dim → embed_dim)
#     """
#     def __init__(
#         self,
#         state_dim: int,
#         embed_dim: int,
#         hidden_dim: int,
#         action_dim: int,
#         rnn_type: str = "LSTM",
#         num_layers: int = 1,
#         num_heads: int = 4,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.rnn_type = rnn_type.upper()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#
#         # 1) 状态嵌入
#         self.state_encoder = StateEmbedding(state_dim, embed_dim)
#
#         # 2) 自注意力
#         self.pre_attn = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=False,
#         )
#         self.pre_norm = nn.LayerNorm(embed_dim)
#
#         # 3) **前向残差 FFN**（Position-wise Feed-Forward）
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, 2 * embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(2 * embed_dim, embed_dim),
#             nn.Dropout(dropout),
#         )
#         self.ffn_norm = nn.LayerNorm(embed_dim)
#
#         # 4) RNN
#         rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}.get(self.rnn_type)
#         if rnn_cls is None:
#             raise ValueError(f"Unsupported RNN type: {rnn_type}")
#         self.rnn = rnn_cls(
#             embed_dim,
#             hidden_dim,
#             num_layers,
#             batch_first=False,
#             dropout=dropout if num_layers > 1 else 0.0,
#         )
#
#         # 5) 输出层
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_dim, action_dim)
#
#     # ------------------------------------------------------------------
#     def forward(self, seq: torch.Tensor) -> torch.Tensor:
#         seq = seq.unsqueeze(1)                                 # [T, 1, state_dim]
#
#         # ① 嵌入
#         emb = self.state_encoder(seq.squeeze(1)).unsqueeze(1)   # [T, 1, E]
#
#         # ② 自注意力
#         attn_out, _ = self.pre_attn(emb, emb, emb)
#         attn_out = self.pre_norm(attn_out + emb)                # 残差 + LN
#
#         # ③ **FFN + 残差 + LN**
#         ffn_out = self.ffn(attn_out)
#         attn_out = self.ffn_norm(ffn_out + attn_out)            # [T, 1, E]
#
#         # ④ RNN
#         if self.rnn_type == "LSTM":
#             h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             c0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             rnn_out, _ = self.rnn(attn_out, (h0, c0))           # [T, 1, H]
#         else:
#             h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             rnn_out, _ = self.rnn(attn_out, h0)                 # [T, 1, H]
#
#         # ⑤ Max Pooling (时间维)
#         max_hidden = rnn_out.max(dim=0).values.squeeze(0)       # [hidden_dim]
#
#         # ⑥ 输出
#         logits = self.fc(F.relu(self.dropout(max_hidden)))      # [action_dim]
#         return logits


class RNN_Multihead_SelfAtte(nn.Module):
    """
    Embedding
      ├──► Branch-A: RNN  (局部/顺序信息)
      └──► Branch-B: Self-Attention + AvgPool  (全局依赖)
          ▼
    Concat([feat_A, feat_B]) → ReLU → FC → action_logits
    """
    def __init__(
        self,
        state_dim: int,
        embed_dim: int,
        hidden_dim: int,
        action_dim: int,
        rnn_type: str = "GRU",
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 0. 状态嵌入
        self.state_encoder = StateEmbedding(state_dim, embed_dim)

        # 1A. RNN 分支
        rnn_cls = {"GRU": nn.GRU, "LSTM": nn.LSTM}.get(self.rnn_type)
        if rnn_cls is None:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 1B. 自注意力分支
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # 2. 融合 & 输出
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim + embed_dim, action_dim)  # 拼接后维度 = H + E

    # ------------------------------------------------------------------
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq : [T, state_dim]  —— 无 batch 维；如需批处理，可改为 [T, B, state_dim]
        """
        seq = seq.unsqueeze(1)                                   # [T, 1, state_dim]

        # (0) 嵌入
        emb = self.state_encoder(seq.squeeze(1)).unsqueeze(1)     # [T, 1, E]

        # ---------- Branch-A : RNN ----------
        if self.rnn_type == "LSTM":
            h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
            c0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
            rnn_out, _ = self.rnn(emb, (h0, c0))                  # [T, 1, H]
        else:
            h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
            rnn_out, _ = self.rnn(emb, h0)                        # [T, 1, H]

        # 取 **时间平均** 代表 RNN 特征（也可改 last step / max 等）
        rnn_feat = rnn_out.mean(dim=0).squeeze(0)                 # [H]

        # ---------- Branch-B : Self-Attention ----------
        attn_out, _ = self.attn(emb, emb, emb)                    # [T, 1, E]
        attn_out = self.attn_norm(attn_out + emb)                 # 残差 + LN

        # 时间平均池化
        attn_feat = attn_out.mean(dim=0).squeeze(0)               # [E]

        # ---------- 融合 ----------
        fused = torch.cat([rnn_feat, attn_feat], dim=-1)          # [H+E]
        logits = self.fc(F.relu(self.dropout(fused)))             # [action_dim]
        return logits



# class RNN_Multihead_SelfAtte(nn.Module):
#     """RNN + Multi‑Head Self‑Attention with Feed‑Forward & Mean Pooling.
#     -------------------------
#      **Attention Pooling → 普通平均池化**：去掉 learnable query，将
#        `attn_out` 在时间维做均值，得到全局表征。
#
#     其余结构（RNN、Self‑Attention、FFN、双 LayerNorm）保持不变。
#     """
#
#     def __init__(self,
#                  state_dim: int,
#                  embed_dim: int,
#                  hidden_dim: int,
#                  action_dim: int,
#                  rnn_type: str = 'GRU',
#                  num_layers: int = 1,
#                  num_heads: int = 4,
#                  dropout: float = 0.1):
#         super().__init__()
#         self.rnn_type = rnn_type.upper()
#         self.embed_dim = embed_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#
#         # 1. 状态嵌入层
#         self.state_encoder = StateEmbedding(state_dim, embed_dim)
#
#         # 2. RNN 模块
#         if self.rnn_type == 'GRU':
#             self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers,
#                               batch_first=False,
#                               dropout=dropout if num_layers > 1 else 0.0)
#         elif self.rnn_type == 'LSTM':
#             self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers,
#                                batch_first=False,
#                                dropout=dropout if num_layers > 1 else 0.0)
#         else:
#             raise ValueError(f"Unsupported RNN type: {rnn_type}")
#
#         # 3. Self‑Attention
#         self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
#                                           num_heads=num_heads,
#                                           dropout=dropout,
#                                           batch_first=False)
#
#         # 4. Position‑wise Feed‑Forward Network
#         self.ffn = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.SiLU(),          # GELU
#             nn.Linear(hidden_dim * 4, hidden_dim)
#         )
#         self.dropout = nn.Dropout(dropout)
#
#         # 5. 正则化层
#         self.norm1 = nn.LayerNorm(hidden_dim)  # after self‑attn
#         self.norm2 = nn.LayerNorm(hidden_dim)  # after pooling
#
#         # 6. 输出层
#         self.fc = nn.Linear(hidden_dim, action_dim)
#
#     def forward(self, seq: torch.Tensor) -> torch.Tensor:
#         """Forward pass.
#
#         Args:
#             seq: Tensor ``[seq_len, state_dim]`` (单条序列，无 batch dim)。
#         Returns:
#             action_logits: Tensor ``[action_dim]``.
#         """
#         # [seq_len, 1, state_dim]
#         seq = seq.unsqueeze(1)
#
#         # === State Embedding ===
#         embedded_seq = self.state_encoder(seq.squeeze(1))          # [seq_len, embed_dim]
#         embedded_seq = embedded_seq.unsqueeze(1)                   # [seq_len, 1, embed_dim]
#
#         # === RNN Encoding ===
#         h0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#         if self.rnn_type == 'LSTM':
#             c0 = torch.zeros(self.num_layers, 1, self.hidden_dim, device=seq.device)
#             rnn_out, _ = self.rnn(embedded_seq, (h0, c0))          # [seq_len, 1, hidden_dim]
#         else:
#             rnn_out, _ = self.rnn(embedded_seq, h0)                # [seq_len, 1, hidden_dim]
#
#         # === Self‑Attention ===
#         attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)
#         attn_out = self.norm1(attn_out + rnn_out)                  # Residual + LN
#
#         # === Feed‑Forward + Residual ===
#         ffn_out = self.ffn(attn_out)
#         attn_out = attn_out + self.dropout(ffn_out)
#
#         # === 普通平均池化（时间维度） ===
#         # attn_out: [seq_len, 1, hidden_dim] → mean over seq_len → [1, hidden_dim]
#         pooled_out = attn_out.mean(dim=0)                          # [1, hidden_dim]
#         final_hidden = pooled_out.squeeze(0)                       # [hidden_dim]
#         final_hidden = self.norm2(final_hidden)                    # second LayerNorm
#
#         # === Output Projection ===
#         final_hidden = self.dropout(F.relu(final_hidden))          # relu
#         action_logits = self.fc(final_hidden)                      # [action_dim]
#         return action_logits


















class MLP_diff(nn.Module):
    def __init__(
        self,
        state_dim,  # 状态维度
        action_dim,  # 动作维度
        hidden_dim,  # diffusion模型中用于学习噪声的MLP中隐藏层的神经元个数
        t_dim = 16,  # denoising step 位置编码的维度大小
    ):
        super(MLP_diff, self).__init__()
        _act = nn.ReLU  # 激活函数
        # self.state_mlp = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     _act(),
        #     nn.Linear(hidden_dim, state_dim)
        # )
        # 因为 denoising step 在reverse process中与位置（第几步）有关，所以利用一个正弦余弦位置编码器，获得 denoising step 的位置编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        # self.mid_layer = nn.Sequential(
        #     nn.Linear(state_dim + action_dim + t_dim, hidden_dim),
        #     _act(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     _act(),
        #     nn.Linear(hidden_dim, state_dim + action_dim + t_dim)
        # )
        self.fc1 = nn.Linear(state_dim + action_dim + t_dim, hidden_dim)  # 输入：环境状态，动作分布x，denoising step的位置编码
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出：噪声（维度和动作维度相同）

    def forward(self, x, time, state):
        # processed_state = self.state_mlp(state)
        # processed_state = state * 1000  # Hongyang乘1000是为了让后续x，t，processed_state处于一个量级
        t = self.time_mlp(time)
        obs = torch.cat([x, t, state], dim=1)  # 输入：环境状态，动作分布x，denoising step的位置编码
        # obs = self.mid_layer(obs)
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        q = self.fc3(obs)  # 输出：噪声（维度和动作维度相同），用q的意思是当到解噪的最后一步时，输出的就是不同动作的q值了
        return q


class StateEmbedding(nn.Module):
    """
      • 两层 MLP（hidden≈4×embed_dim）
      • 残差连接（若 state_dim ≠ embed_dim 自动线性投影）
      • LayerNorm 稳定训练
      • Dropout 防止过拟合
    """
    def __init__(
        self,
        state_dim: int,
        embed_dim: int,
        hidden_multiplier: int = 2,   # hidden_dim = hidden_multiplier * embed_dim
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_multiplier * embed_dim

        # 两层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        # 残差的尺寸对齐：若两维度不同就投影一下
        self.res_proj = (
            nn.Identity()
            if state_dim == embed_dim
            else nn.Linear(state_dim, embed_dim, bias=False)
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: 形状可为  [T, state_dim]  或  [T, B, state_dim]
        返回同批次形状的 [T, *, embed_dim]
        """
        out = self.mlp(x)                       # MLP
        residual = self.res_proj(x)             # 残差对齐
        out = self.norm(out + residual)         # 残差 + LN
        return self.dropout(out)                # 最终嵌入


class StateEmbedding2(nn.Module):
    """
    Level-2 版本：
      • 两层 MLP (+ 残差 + LayerNorm)
      • Learnable Positional Embedding
      • Feature Dropout
    """
    def __init__(
        self,
        state_dim: int,
        embed_dim: int,
        max_len: int = 20,          # 支持的最长序列
        hidden_multiplier: int = 2,
        dropout: float = 0.1,
        feature_drop: float = 0.1,   # Feature Dropout 概率
    ):
        super().__init__()
        hidden_dim = hidden_multiplier * embed_dim

        # ① 两层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        # ② 残差尺寸对齐
        self.res_proj = (
            nn.Identity()
            if state_dim == embed_dim
            else nn.Linear(state_dim, embed_dim, bias=False)
        )
        # ③ 可学习 Positional Embedding（[max_len, 1, embed_dim]）
        self.pos = nn.Parameter(torch.randn(max_len, 1, embed_dim) * 0.02)

        # ④ 正则化
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.feature_drop = FeatureDropout(feature_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [T, state_dim]  或  [T, B, state_dim]
        返回 : [T, *, embed_dim]
        """
        T = x.size(0)
        if T > self.pos.size(0):
            raise ValueError(f"Sequence length {T} > max_len {self.pos.size(0)}")

        # 两层 MLP
        out = self.mlp(x)
        residual = self.res_proj(x)
        out = self.norm(out + residual)          # 残差归一

        # 加位置编码（广播到批次维）
        pos = self.pos[:T]                       # [T, 1, E]
        out = out + pos if out.dim() == 3 else out + pos.squeeze(1)

        # Dropout（随机特征 & 元素）
        out = self.feature_drop(out)
        return self.drop(out)


class FeatureDropout(nn.Module):
    """
    对特征维度进行随机置零：
    概率 p 下，将同一特征列在所有时间步 / 批次同时置零。
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        # x: [T, B, C]  或 [T, C]
        mask_shape = (1, 1, x.size(-1)) if x.dim() == 3 else (1, x.size(-1))
        keep_prob = 1.0 - self.p
        mask = torch.rand(mask_shape, device=x.device).lt(keep_prob).float() / keep_prob
        return x * mask