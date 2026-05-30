import torch
import torch.nn as nn
from model import layers
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import math
from scipy.stats import pearsonr
from sklearn.decomposition import PCA



class STGCNChebGraphConv(nn.Module):
    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # D: Dropout
    def __init__(self, args, blocks, n_vertex):
        super(STGCNChebGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)

        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        self.text = args.text
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)
        self.use_film = args.film
        self.cross_att = args.cross_att
        if self.cross_att:
            self.CrossAttention = OptimizedCrossAttention(64, 768)
        if self.use_film:
            self.film = FiLM_NodeConditioned(768, 64, 1260)

        self.to(args.device)
    def forward(self, x):

        x = x.permute(0, 3, 1, 2)
        x = self.st_blocks(x) ##[b, 64, 7, 1260]

        # # 2. text features
        if self.text  ==1:
            text_emb = text_emb.to(x.device)
            if self.use_film:
                # x = rearrange(x, 'b t n d -> b (t n) d')  # (B, Lq, D)
                x = self.film(text_emb, x)
                # x = rearrange(num_seq, 'b (t n) d -> b t n d', t=T, n=N)
            elif self.cross_att:
                x = x.permute(0, 2, 3, 1)  ## b,t,n,d
                x = self.CrossAttention(x, text_emb)
                x = x.permute(0, 3, 1, 2)
            else :
                # insert traffic 时间长度 7
                text_h = self.text_proj(text_emb)  # (B, T, hidden)
                text_h = F.interpolate(text_h.permute(0, 2, 1), size=x.shape[2], mode='linear', align_corners=False)  # [B, 64, 7]
                text_h = text_h.unsqueeze(-1).expand(-1, -1, -1, 1260)  # [B, 64, 7, N]
                # print("shape of blocks, text_emb, text_h: ", x.shape,text_emb.shape, text_h.shape) ##256, 15, 64
                x = x + text_h  # Fusion (可以换成 concat 后接 Linear)
        B,D,T,N = x.shape
        # x_embedding = x.reshape(B,D,T*N).permute(0, 2, 1)
        if self.Ko > 1:   ## Ko=7
            x, x_embedding = self.output(x)
            # print(x_embedding.shape)    ##[64, 15, 1260, 128])
            x_embedding = x_embedding.reshape(-1, T * N, x_embedding.shape[3])

        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        x = x.permute(0, 2, 3, 1)   ## [256, 128, 1, 1260] ➡ [256, 1,  1260, F]
        return x, x_embedding.detach()

    def extract_features(self,x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)


class OptimizedCrossAttention(nn.Module):
    def __init__(self, g_dim, t_dim, d_model=128, n_heads=4, dropout=0.15, node_pos_dim_reduction=2):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.dk = d_model // n_heads

        self.W_q = nn.Linear(g_dim, d_model)
        self.W_k = nn.Linear(g_dim, d_model)
        self.W_v = nn.Linear(g_dim, d_model)

        # 位置编码 - 使用更高效的实现
        # self.node_map = nn.Parameter(torch.randn(1260, d_model) * 0.02)
        # reduced_dim = max(1, d_model // node_pos_dim_reduction)
        # self.register_buffer('graph_pos_emb', self._get_sinusoidal_encoding(15, d_model))
        # self.register_buffer('text_pos_emb', self._get_sinusoidal_encoding(15, d_model))

        # self.node_pos_emb = nn.Parameter(torch.zeros(1, 1, 1260, reduced_dim) * 0.02)
        # self.node_pos_proj = nn.Linear(reduced_dim, d_model, bias=False)
        # nn.init.normal_(self.node_pos_emb, std=0.02)


        # 可学习的温度参数 - 使用log空间以确保正值
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(self.dk ** 0.5)))

        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # 简化输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, g_dim)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(g_dim)
        self.norm2 = nn.LayerNorm(g_dim)

        self.modality_gate = nn.Sequential(
            nn.Linear(g_dim + d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def _get_sinusoidal_encoding(self, seq_len, device):
        """生成正弦位置编码"""
        position = torch.arange(seq_len, dtype=torch.float, device = device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) *
                             (-math.log(10000.0) / self.d_model))

        pos_enc = torch.zeros(1, seq_len, self.d_model, device=device)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def _efficient_attention(self, q, k, v, chunk_size=64):
        """
        高效的分块注意力计算
        减少内存峰值，避免大矩阵乘法
        """
        B, n_heads, T_g, N, dk = q.shape
        _, _, T_t, _, _ = k.shape

        temperature = torch.exp(self.log_temperature)

        outputs = []
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            chunk_q = q[:, :, :, i:end_idx, :]  # (B, n_heads, T_g, chunk, dk)

            # 计算注意力分数 - 使用scaled dot-product
            # (B, n_heads, T_g, chunk, dk) @ (B, n_heads, T_t, chunk, dk).T
            attn_score = torch.einsum('bhtnk,bhtmk->bhtnm', chunk_q, k) / temperature

            # 使用数值稳定的softmax
            attn_score = attn_score - attn_score.max(dim=-1, keepdim=True)[0].detach()
            attn_weight = F.softmax(attn_score, dim=-1)
            attn_weight = self.attn_dropout(attn_weight)

            # 计算输出
            chunk_out = torch.einsum('bhtnm,bhtmk->bhtnk', attn_weight, v)
            outputs.append(chunk_out)

        out = torch.cat(outputs, dim=3)  # (B, n_heads, T_g, N, dk)
        return out

    def rbf_kernel_torch(self, X, Y, gamma=None):
        """
        Compute RBF kernel between X and Y using PyTorch.
        Args:
            X: (N, d)
            Y: (M, d)
            gamma: 1 / (2 * sigma^2)
        Returns:
            (N, M) kernel matrix
        """
        XX = (X ** 2).sum(dim=1, keepdim=True)
        YY = (Y ** 2).sum(dim=1, keepdim=True)
        dist = XX + YY.T - 2 * X @ Y.T
        if gamma is None:
            # median heuristic
            median = torch.median(dist[dist > 0])
            gamma = 1.0 / (2 * (median + 1e-6))
        K = torch.exp(-gamma * dist)
        return K

    def approx_hsic_torch(self, X, Y, sample_size=500):
        """
        Approximate HSIC using RBF kernel with GPU acceleration.
        Args:
            X: (N, d1)
            Y: (N, d2)
        Returns:
            scalar HSIC value
        """
        N = X.shape[0]
        device = X.device

        # Nyström subsampling
        idx = torch.randperm(N, device=device)[:min(sample_size, N)]
        X_sub, Y_sub = X[idx], Y[idx]

        # Compute RBF kernels
        K = rbf_kernel_torch(X, X_sub)
        L = rbf_kernel_torch(Y, Y_sub)

        # Center kernels: H = I - 1/n
        n = K.shape[0]
        H = torch.eye(n, device=device) - torch.ones((n, n), device=device) / n
        KH = H @ K
        LH = H @ L

        hsic_value = torch.trace(KH.T @ LH) / (n ** 2)
        return hsic_value

    def filter_text_embeddings(self, text_emb, x, method="hsic", corr_threshold=None, top_k=64):
        """
        Args:
            text_emb (torch.Tensor): shape (B,T,768)
            y (torch.Tensor): shape (B,T,N,64)
        Returns:
            filtered_emb (torch.Tensor), selected_idx (np.ndarray)
        """
        B, T, D = text_emb.shape

        # Flatten for correlation
        text_tensor = text_emb.view(-1, D)  # (B*T, 768)
        x_tensor = x.mean(dim=2).view(-1, top_k)  # (B*T, 1)

        if method == "pearson":
            x_scalar = x.mean(dim=2).mean(dim=2).view(-1, 1) # (N,1)
            # standardize
            t_mean = text_tensor.mean(dim=0, keepdim=True)
            t_std = text_tensor.std(dim=0, unbiased=False, keepdim=True)
            text_std = (text_tensor - t_mean) / (t_std + 1e-8)  # (N, D_text)

            x_mean = x_scalar.mean(dim=0, keepdim=True)
            x_stddev = x_scalar.std(dim=0, unbiased=False, keepdim=True)
            x_std = (x_scalar - x_mean) / (x_stddev + 1e-8)  # (N,1)

            scores = torch.abs((text_std * x_std).mean(dim=0))
        else:
            hsic_scores = []
            for i in range(text_tensor.shape[1]):
                xi = text_tensor[:, i:i + 1]
                try:
                    score = approx_hsic_torch(xi, x_tensor, sample_size=sample_size)
                except Exception:
                    device = text_tensor.device
                    score = torch.tensor(0.0, device=device)
                hsic_scores.append(score)
            scores = torch.stack(hsic_scores)

        # Select dimensions
        if corr_threshold is not None:
            selected_idx = torch.nonzero(scores > corr_threshold, as_tuple=True)[0]
        else:
            _, selected_idx = torch.topk(scores, top_k)

        # Filter and reshape
        filtered_emb = text_tensor[:, selected_idx].view(B, T, -1)

        return filtered_emb, selected_idx.cpu().numpy()

    def forward(self, graph_x, text_x_original, attn_chunk_size=32):
        """
        Args:
            graph_x: (B, T_g, N, C_g)  图特征
            text_x:  (B, T_t, C_t)     文本特征
        Returns:
            out: (B, T_g, N, C_g)
        """
        B, T, N, C_g = graph_x.shape
        B, _, C_t = text_x_original.shape
        residual = graph_x

        graph_x = self.norm1(graph_x)
        text_x, kept_idx = self.filter_text_embeddings(
            text_emb=text_x_original,
            x=graph_x,
            top_k = C_g
        )

        q = self.W_q(text_x)  # (B, T_g, N, d_model)
        k = self.W_k(graph_x)  # (B, T_t, d_model)
        v = self.W_v(graph_x)  # (B, T_t, d_model)


        q = q.unsqueeze(2).expand(-1, -1, N, -1)

        q = q.view(B, T, N, self.n_heads, self.dk).permute(0, 3, 1, 2, 4)
        k = k.view(B, T, N, self.n_heads, self.dk).permute(0, 3, 1, 2, 4)
        v = v.view(B, T, N, self.n_heads, self.dk).permute(0, 3, 1, 2, 4)
        ## sometimes, Pearson Correlation Coefficient/mutual information are necessary to validate the contribution of text embeddings


        # Prefer using PyTorch's scaled_dot_product_attention if available (FlashAttention backed)
        # use_native = hasattr(F, 'scaled_dot_product_attention')  #for torch>2.0 use_native=True

        # native expects q,k,v shape (..., seq_len, head_dim) - we need to merge node dimension into seq dimension
        # We'll compute attention per node separately by reshaping: combine (T_g) as seq_q and (T_t) as seq_k
        # But F.scaled_dot_product_attention operates on last two dimensions for seq length, so we can loop over nodes if necessary
        # For performance, merge batch/head/node dims: (B*n_heads*N, T_g, dk) etc.
        BnhN = B * self.n_heads * N
        q_native = q.permute(0, 1, 3, 2, 4).contiguous().view(BnhN, T, self.dk)  # (B*n_heads*N, T_g, dk)
        k_native = k.permute(0, 1, 3, 2, 4).contiguous().view(BnhN, T, self.dk)  # (B*n_heads*N, T_t, dk)
        v_native = v.permute(0, 1, 3, 2, 4).contiguous().view(BnhN, T, self.dk)  # (B*n_heads*N, T_t, dk)

        # scaled_dot_product_attention supports dropout; requires attn_mask if needed (we don't use mask here)
        attn_out_native = F.scaled_dot_product_attention(q_native, k_native, v_native,
                                                         dropout_p=self.attn_dropout.p if self.training else 0.0,
                                                         is_causal=False)  # (BnhN, T_g, dk)
        # reshape back: (B, n_heads, T_g, N, dk)
        attn_out = attn_out_native.view(B, self.n_heads, N, T, self.dk).permute(0, 1, 3, 2, 4).contiguous()
        attn_out = attn_out.permute(0, 2, 3, 1, 4).contiguous().view(B, T, N, self.d_model)

        out = self.out_proj(attn_out)
        out = self.proj_dropout(out)

        with torch.no_grad():
            graph_summary = residual.mean(dim=2).mean(dim=1)
            text_summary = attn_out.mean(dim=1).mean(dim=1)

        gate_input = torch.cat([graph_summary, text_summary], dim=-1)  # (B, C_g + d_model)
        gate = torch.sigmoid(self.modality_gate(gate_input)).view(B, 1, 1, 1)  # (B,1,1,1)
        # print("Gate mean:", gate.mean().item(), "std:", gate.std().item()) ## if model try to block text
        out = self.norm2(residual + gate  * out)

        return out




class FiLM_NodeConditioned(nn.Module):
    def __init__(self, text_dim, emb_dim, num_nodes):
        super().__init__()
        self.text_net = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim * 2))
        self.node_gate = nn.Embedding(num_nodes, 2)   # (scale, shift)

    def forward(self, text_emb, nu_emb):
        # nu_emb: (B,D,L,N), text_emb: (B,L,text_dim)
        B,D,L,N = nu_emb.shape
        gamma_beta = self.text_net(text_emb)  # (B,L,2D)
        gamma, beta = gamma_beta.chunk(2, -1) # (B,L,D)
        #  gating (N,2) -> (1,N,2) -> (B,L,N,2)
        node_gates = self.node_gate.weight.unsqueeze(0).unsqueeze(0).expand(B,L,N,2)
        node_scale, node_shift = node_gates.unbind(-1)  # (B,L,N)
        # broadcasting (B,L,D) -> (B,L,N,D)
        gamma = gamma.unsqueeze(2).expand(B,L,N,D)
        beta  = beta.unsqueeze(2).expand(B,L,N,D)
        ## modulate the data
        gamma = gamma * (1 + node_scale.unsqueeze(-1))
        beta  = beta + node_shift.unsqueeze(-1)
        # reshape to (B,D,L,N)
        gamma, beta = gamma.permute(0,3,1,2), beta.permute(0,3,1,2)
        return nu_emb * (1 + gamma) + beta



class STGCNGraphConv(nn.Module):

    def __init__(self, args, blocks, n_vertex):
        super(STGCNGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.do = nn.Dropout(p=args.droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        return x

class STGCNWithText(nn.Module):
    def __init__(self, STGCNChebGraphConv, d_text=768, d_hidden=64):
        super().__init__()
        self.stgcn = STGCNChebGraphConv
        self.text_proj = nn.Linear(d_text, d_hidden)

    def forward(self, x_numeric, x_text):
        """
        x_numeric: (B, T, N, F_num)
        x_text: (B, T, d_text)
        """
        h_num = self.stgcn.extract_features(x_numeric)  # (B, T, N, d_hidden)

        # Project text embedding
        h_text = self.text_proj(x_text)  # (B, T, d_hidden)

        # Align: expand text to all nodes
        h_text_expanded = h_text.unsqueeze(2).expand(-1, -1, h_num.size(2), -1)
        # (B, T, N, d_hidden)

        # Fusion
        h_fused = h_num + h_text_expanded

        # Prediction
        y = self.stgcn.predict_from_features(h_fused)
        return y

