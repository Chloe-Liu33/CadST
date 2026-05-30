

import torch
import torch.nn as nn

import numpy as np
from torch.autograd import Variable
softmax = nn.Softmax(0)


## llm
def RFF(feature, w=None, sigma=None, num_f=None, device=None, dtype=None):
    if device is None:
        device = feature.device
    if dtype is None:
        dtype = feature.dtype
    if num_f is None:
        num_f = 1
    n, r, d = feature.shape
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:     # w is weight for the triangle function
        w = 1 / sigma * torch.randn(d, num_f, device=device, dtype=dtype)
        b = 2 * np.pi * torch.rand(num_f, device=device, dtype=dtype)


    mid = torch.matmul(feature, w) + b.view(1, 1, -1) ## 'int' object has no attribute 't'


    # ##what is the meaning of mid3??
    # mid3 = mid
    # mid3 -= mid3.min(dim=2, keepdim=True)[0].min(dim=1,keepdim=True)[0]      ###数据的归一化将从两个不同的维度进行
    # mid3 /= mid3.max(dim=2, keepdim=True)[0].max(dim=1,keepdim=True)[0]
    # # mid3 -= mid3.min(dim=3, keepdim=True)[0].min(dim=2,keepdim=True)[0]
    # # mid3 /= mid3.max(dim=3, keepdim=True)[0].max(dim=2,keepdim=True)[0]
    # if torch.isnan(mid3).any():
    #     print('Nan detected for mid')
    # # mid1 *= np.pi / 2.0     #n,r,1
    # mid3 *= np.pi / 2.0

    Z = torch.sqrt(torch.tensor(2.0 / num_f, device=device, dtype=dtype))
    # concatenate cos and sin instead of summing
    Z3 = Z * torch.cat([torch.cos(mid), torch.sin(mid)], dim=-1)

    # Z3 = Z * (torch.cos(mid).cuda() + torch.sin(mid).cuda())

    return Z3

def center_features(X):
    """
    X: (N, D)
    returns: X_centered (N, D)
    subtract mean across samples
    """
    mean = X.mean(dim=0, keepdim=True)
    return X - mean


# -------- covariance from features --------
def cov_matrix_from_features(X, unbiased=True):
    """
    X: (N, D) already centered or not
    returns: covariance matrix (D, D)
    """
    if X.dim() != 2:
        raise ValueError("Expect X shape (N, D)")
    N = X.size(0)
    Xc = X - X.mean(dim=0, keepdim=True)
    if unbiased:
        denom = (N - 1) if N > 1 else 1.0
    else:
        denom = N
    cov = (Xc.t() @ Xc) / denom  # (D, D)
    return cov

# -------- HSIC-like loss using RFF (single view, de-correlation of RFF dims) --------
def lossb_expect_rff(cfeaturec, num_f=1, device=None, dtype=None, scale_factor=1.0):
    """
    cfeaturec: (batch, r, d) torch tensor
    num_f: number of random features
    This loss will:
      - produce RFF features (batch*r, 2*num_f)
      - compute centered covariance across samples -> cov (D, D)
      - penalize off-diagonal squared sum: sum_{i!=j} cov_{ij}^2
    Returns: scalar loss (tensor), and raw_loss (before dividing by scale_factor)
    """
    if device is None:
        device = cfeaturec.device
    if dtype is None:
        dtype = cfeaturec.dtype

    # get rff
    Z3 = RFF(cfeaturec, num_f=num_f, device=device, dtype=dtype)  # (b, r, 2*num_f)

    # flatten samples: treat each (batch, r) row as independent sample
    N = Z3.shape[0] * Z3.shape[1]
    D = Z3.shape[-1]
    flat = Z3.reshape(N, D)  # (N, D)

    # center and cov
    cov = cov_matrix_from_features(flat, unbiased=True)  # (D, D)

    # zero diag
    diag_mask = torch.eye(D, device=device, dtype=dtype)
    off = cov * (1.0 - diag_mask)
    raw_loss = torch.sum(off ** 2)  # scalar

    loss = raw_loss / float(scale_factor)
    return loss

def cov_sample(x, w=None):   ##covariance
    # num = x.size(0)
    if w == None:
        # w1 = w[0:num,:]
        # w1 = w1.view(-1,1)
        # cov = torch.matmul((w1 * x).t(), x)
        # e = torch.sum(w1 * x, dim=0).view(-1, 1)
        # res = cov - torch.matmul(e, e.t())
        ##这里没有权重，计算整个矩阵的协方差
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        w = w[0:x.size(0), :]
        # w = w[0:x.size(0)*100,:]

        #x:bxNxd    cov:NxTxdxd
        # print('weight:',w.shape,'x:',x.shape)  ## bx1, bxNxTxd  * means the last two dimensions multiplicaiton
        x= x.permute(1,0,2)  ## Nxbxd
        cov = torch.matmul((w * x).permute(0,2,1), x)   ##  N,d,d,,第二个size

        e = w*x
        res = cov - torch.matmul(e.transpose(2,1),e )

    return res


# laplacian矩阵
def unnormalized_laplacian(adj_matrix):
    # 先求度矩阵
    R = np.sum(adj_matrix, axis=1)
    degreeMatrix = np.diag(R)
    return degreeMatrix - adj_matrix

# 随机漫步归一化邻接矩阵
def normalized_laplacian2(gamma, adj_matrix):
    # print('adjmatrix',adj_matrix+torch.float(torch.eye(adj_matrix.size(0))))
    # print('adjmatrix', adj_matrix)
    R = np.sum(adj_matrix, axis=1)
    # print("R",R)   #r 出现了0
    R = np.where(R==0,1000000,R)   ####R 应该取哪个值？？？
    R_frac = 1 / R
    D_frac = np.diag(R_frac)
    # D_sqrt = D_sqrt.fillna(0)   ## this is for data frame
    # D_sqrt[np.isnan(D_sqrt)]=0   ## this is for numpy
    I = np.eye(adj_matrix.shape[0])
    I = torch.tensor(I, device='cuda')
    return I + gamma*torch.tensor(np.matmul(D_frac, adj_matrix),device="cuda")
## 对称归一化邻接矩阵
def normalized_laplacian3(gamma, adj_matrix):
    # print('adjmatrix',adj_matrix+torch.float(torch.eye(adj_matrix.size(0))))
    # print('adjmatrix', adj_matrix)
    R = np.sum(adj_matrix, axis=1)
    # print("R",R)   #r 出现了0
    R = np.where(R==0,1000000,R)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    # D_sqrt = D_sqrt.fillna(0)   ## this is for data frame
    # D_sqrt[np.isnan(D_sqrt)]=0   ## this is for numpy
    I = np.eye(adj_matrix.shape[0])
    I = torch.tensor(I, device='cuda')
    # print(I+gamma*np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt))
    return I+gamma*torch.tensor(np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt),devide="cuda")

def cov_node(x, w=None):  ##covariance
    if w == None:
        # n = x.shape[0]
        # cov = torch.matmul(x.t(), x) / n
        # e = torch.mean(x, dim=0).view(-1, 1)
        # res = cov - torch.matmul(e, e.t())

        # N = x.shape[1]  ##64，244，64
        x1 = x.permute(0,2,1)
        cov = torch.matmul(x1, x)

        e = torch.mean(x1, dim=0)
        res = cov - torch.matmul(e, e.transpose(1,0))
        res = torch.mean(res, dim=0)
        ##这里没有权重，计算整个矩阵的协方差

    else:
        w = w.view(-1, 1)
        ##x:bxNxd

        cov = torch.matmul((w * x).permute(0, 2,1), x)
        e = w*x
        res = cov - torch.matmul(e.transpose(2,1), e)   ##cross-entropy calculation //n,64,64

    return res



##llm
def lossb_expect_old(cfeaturec, num_f, sum, weight=None):
    if num_f == 0:
        n = cfeaturec.size(0)
        r = cfeaturec.size(1)
        d = cfeaturec.size(2)
        cfeaturec3 = cfeaturec.view(n, r, d, 1)
    else:
        cfeaturec3  = RFF(cfeaturec,num_f=num_f, sum=sum)
        cfeaturec3 = cfeaturec3.cuda()

    loss = Variable(torch.FloatTensor([0]).cuda())

    for i in range(cfeaturec3.size()[-1]):    ##llm:this is only one i
        cfeaturec3 = cfeaturec3[:, :, :, i]


        cov3 = cov_node (cfeaturec3, weight)
        cov_matrix3= cov3 * cov3     ##b,d,d
        loss += torch.sum(cov_matrix3) - torch.trace(cov_matrix3)

    lambdap = 7e5 ## original 70
    loss = loss / lambdap

    return loss


def lossb_expect(cfeaturec, num_f, weight=None):
    if num_f > 0:
        cfeaturec_rff = RFF(cfeaturec, num_f=num_f)  # (n, r, d, num_f or 2*num_f)
    else:
        return Variable(torch.FloatTensor([0]).cuda())

    # 1. 将特征展平为 (batch*r*d, num_f)
    if sum:
        final_dim = cfeaturec_rff.size(-1)
    else:
        final_dim = cfeaturec_rff.size(-1)

    flattened_features = cfeaturec_rff.reshape(-1, final_dim)

    # 2. 计算协方差矩阵 (dim_f, dim_f)
    # 简化：使用无权重的标准协方差
    X_mean = flattened_features - flattened_features.mean(0, keepdim=True)
    cov_matrix = torch.matmul(X_mean.t(), X_mean) / (X_mean.size(0) - 1)

    # 3. 损失为非对角线元素的平方和 (Off-Diagonal Frobenius Norm Squared)
    identity = torch.eye(cov_matrix.size(0), device=cov_matrix.device)
    non_diag_cov = cov_matrix * (1 - identity)
    loss = torch.sum(non_diag_cov ** 2)
    print(loss)

    lambdap = 7e7
    return loss / lambdap


def global_local(pre_weight_sample1,pre_features, weight_sample, features, ratio,epoch,i):
    ## llm-global reweighting
    if epoch == 0 and i < 10:
        if features.size()[0] < pre_features.size()[0]:
            pre_features[:features.size()[0]] = (pre_features[:features.size()[0]] * i + features) /(i+1)
            pre_weight_sample1[:features.size()[0]] = (pre_weight_sample1[:features.size()[0]] * i + weight_sample[:features.size()[0]]) / (i+1)
        else:
            pre_features = (pre_features * i + features) / (i + 1)
            pre_weight_sample1 = (pre_weight_sample1 * i + weight_sample) / (i + 1)
    elif features.size()[0] < pre_features.size()[0]:
        pre_features[:features.size()[0]] = pre_features[:features.size()[0]] * ratio + features * (
                    1 - ratio)
        pre_weight_sample1[:features.size()[0]] = pre_weight_sample1[
                                                    :features.size()[0]] * ratio + weight_sample[:features.size()[0]] * (
                                                                1 - ratio)
    else:
        pre_features = pre_features * ratio + features * (1 - ratio)
        pre_weight_sample1 = pre_weight_sample1 * ratio + weight_sample * (1 - ratio)
    return pre_features,pre_weight_sample1
