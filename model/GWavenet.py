import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, args, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=15,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout

        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.blocks = blocks
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(args.device), requires_grad=True).to(args.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(args.device), requires_grad=True).to(args.device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(args.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(args.device)
                self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            # self.padding_time = ((kernel_size - 1) * new_dilation)

            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation, padding=(0, (kernel_size-1)*new_dilation)))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation, padding=(0, (kernel_size-1)*new_dilation )))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.cross_att = args.cross_att
        self.text = args.text
        if self.cross_att:
            self.CrossAttention = OptimizedCrossAttention(32, 768)

    def forward(self, input, text_emb):
        input = input.permute(0, 3, 2, 1)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)  ##[64, 32, 1260, 15]

        if self.text == 1 and self.cross_att:
            text_emb = text_emb.to(x.device)
            # print("shape before cross attention", x.shape)
            x = x.permute(0, 3, 2, 1)  ## b,t,n,d=32
            x = self.CrossAttention(x, text_emb)
            x = x.permute(0, 3, 2, 1)


        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x[...,:15]
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = x[...,:15]
            # print("s",x.shape, self.padding_time, residual.shape)

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                # skip = skip[:, :, :,  -s.size(3):]
                skip = skip[:, :, :, :15]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            # x = x + residual
            x = self.bn[i](x)

        x = F.relu(skip) #[64, 256, 1260, 15]
        x_emb = x.reshape(x.shape[0],x.shape[1],-1).permute(0, 2, 1)  #[64, 1260* 15, 256]
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x.permute(0, 3, 2, 1), x_emb

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

    def filter_text_embeddings(self, text_emb, x, method="hsic", corr_threshold=None, top_k=32):
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



