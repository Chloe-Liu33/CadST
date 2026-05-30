import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def mask_(seg_num, input_time):
  masknp = np.empty((seg_num*input_time, seg_num*input_time))
  for i in range(input_time):
    tmp = np.empty((seg_num, input_time*seg_num))
    tmp[:, :(i+1)*seg_num] = False
    tmp[:, (i+1)*seg_num:] = True
    masknp[i*seg_num:(i+1)*seg_num, :] = tmp
  return masknp.astype('bool')
#Our Traffic prediction Model
class GATModel(nn.Module):
    def __init__(self, args, seg_num, input_time,con_feature,input_feature,emb_feature,prediction_step,dropout_ratio):                                                                             # model parameters
        super(GATModel, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.seg_num = seg_num
        self.device = args.device
        self.input_time = args.n_his
        self.prediction_step = args.n_pred
        self.emb_feature = emb_feature
        self.node_cons = nn.Parameter(torch.FloatTensor(seg_num * input_time, con_feature))
        self.context_weight = nn.Parameter(torch.FloatTensor(con_feature, input_feature, emb_feature))
        self.context_bias = nn.Parameter(torch.FloatTensor(con_feature, emb_feature))
        self.bn1 = nn.BatchNorm1d(seg_num * input_time)
        self.bn2 = nn.BatchNorm1d(seg_num)
        self.FC = nn.Linear((input_time*emb_feature), int((emb_feature*input_time)/3))
        self.FC2 = nn.Linear(int((emb_feature*input_time)/3), prediction_step)
        self.AttFC = nn.Linear(emb_feature,emb_feature)
        self.ReLU = nn.ReLU()
        self.telayer = nn.TransformerEncoderLayer(d_model=emb_feature, nhead=2, batch_first=True)
        self.te = nn.TransformerEncoder(self.telayer, 1)
        nn.init.kaiming_normal_(self.node_cons)
        nn.init.kaiming_normal_(self.context_weight)
        nn.init.kaiming_normal_(self.context_bias)
        self.masking = torch.from_numpy(mask_(seg_num, input_time))
        self.cross_att = args.cross_att
        self.text = args.text
        if self.cross_att:
            self.CrossAttention = OptimizedCrossAttention(64, 768)

    def forward(self, x, text_emb):
        x = x.reshape(-1, self.input_time*self.seg_num,2)
        # model architecture
        do = torch.nn.Dropout(p=self.dropout_ratio)
        cwpl_weights = torch.einsum('ij,jkl->ikl', self.node_cons, self.context_weight)
        cwpl_bias = self.node_cons.matmul(self.context_bias)
        x = torch.einsum('bij,ijk->bik', x, cwpl_weights) + cwpl_bias
        x = self.bn1(x)
        if self.text == 1 and self.cross_att:
            text_emb = text_emb.to(x.device)  ##[64, 18900, 64]
            x = x.reshape(-1, self.input_time, self.seg_num, x.shape[2])  ## b,t,n,d=64
            x = self.CrossAttention(x, text_emb)
            x = x.reshape(-1, self.input_time * self.seg_num, x.shape[3])
        x = F.relu(x)
        output = self.te(x, mask=self.masking.to(self.device))
        output = torch.reshape(output, (-1, self.input_time, self.seg_num, self.emb_feature))
        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (-1, self.seg_num, self.input_time * self.emb_feature))

        x_emb = output.reshape(-1, self.seg_num * self.input_time, self.emb_feature)
        output = self.FC(output)
        output = self.bn2(output)
        output = F.relu(output)
        output = self.FC2(output)
        # print(output.shape)   ##[2, 1260, 30]
        output = output.reshape(-1, self.seg_num, 15, 2)
        output = torch.transpose(output, 1, 2)
        # output = torch.reshape(output, (-1, self.seg_num * self.prediction_step, 1))
        return output, x_emb

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

    def filter_text_embeddings(self, text_emb, x, method="hsic", corr_threshold=None, top_k=None):
        """
        Args:
            text_emb (torch.Tensor): shape (B,T,768)
            y (torch.Tensor): shape (B,T,N,64) (B,T,N,32)
        Returns:
            filtered_emb (torch.Tensor), selected_idx (np.ndarray)
        """
        B, T, D = text_emb.shape

        # Flatten for correlation
        text_tensor = text_emb.view(-1, D)  # (B*T, 768)
        x_tensor = x.mean(dim=2).view(-1, x.shape[3])  # (B*T, 1)

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
            top_k=graph_x.shape[3]
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



