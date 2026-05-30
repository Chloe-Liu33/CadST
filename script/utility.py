import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch
from sklearn.metrics import classification_report, f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

class MaskedStandardScaler:
    """
    带 mask 的 StandardScaler：
    - 仅使用 mask=1 的位置计算 mean/std
    - mask=0 的位置在 transform 时用有效值均值填充
    """

    def __init__(self):
        self.scaler = None
        self.C = None

    def fit(self, data, mask):
        """
        data: (N, T, V, C)
        mask: same shape as data OR (N, T, V, 1); 1=有效, 0=无效
        """
        N,T,V,C = data.shape
        self.C = C
        flat = data.reshape(-1, C)

        # 处理 mask
        flat_mask = mask.reshape(-1).astype(bool)
        # flat_mask = np.tile(flat_mask[:, None], (1, C))  # 扩展到每个特征

        flat_for_fit = flat.copy().astype(float)

        for c in range(C):
            # valid = flat_mask[:, c]
            valid = flat_mask
            # print(flat.shape, valid.shape, type(valid), valid.dtype)
            if valid.sum() == 0:
                mean_val = 0.0
            else:
                mean_val = flat[valid, c].mean()
            flat_for_fit[~valid, c] = mean_val

        # fit StandardScaler
        self.scaler = StandardScaler()
        self.scaler.fit(flat_for_fit)

    def transform(self, data, mask):
        N,T,V,C = data.shape
        flat = data.reshape(-1, C)

        flat_mask = mask.reshape(-1).astype(bool)
        # flat_mask = np.tile(flat_mask[:, None], (1, C))

        flat_for_transform = flat.copy().astype(float)
        for c in range(C):
            valid = flat_mask
            mean_val = flat_for_transform[valid, c].mean() if valid.sum()>0 else 0.0
            flat_for_transform[~valid, c] = mean_val

        flat_scaled = self.scaler.transform(flat_for_transform)
        return flat_scaled.reshape(N, T, V, C)

    def inverse_transform(self, data, mask):
        N,T,V,C = data.shape
        flat = data.reshape(-1, C)
        flat_inv = self.scaler.inverse_transform(flat)

        flat_mask = mask.reshape(-1)
        # flat_mask = np.tile(flat_mask[:, None], (1, C))

        for c in range(C):
            valid = flat_mask
            mean_val = flat_inv[valid, c].mean() if valid.sum()>0 else 0.0
            flat_inv[~valid, c] = mean_val

        return flat_inv.reshape(N, T, V, C)



def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    # If you encounter a NotImplementedError, please update your scipy version to 1.10.1 or later.
    eigval_max = norm(gso, 2)

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso

def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

def masked_loss(pred, target,mask, loss_type="mse"):
    mask = mask.float()  # 转成 0/1
    diff = pred - target

    if loss_type.lower() == "mae":
        loss = torch.abs(diff) * mask
    elif loss_type.lower() == "mse":
        loss = diff**2 * mask
    else:
        raise ValueError("loss_type must be 'mae' or 'mse'")
    return loss.sum() / mask.sum()


def evaluate_model(args, model, data_iter, F_num, loss_type="mse"):
    model.eval()
    l_sum1, l_sum2,l_sum3, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        # all_alpha = []
        for x, text, y in data_iter:
            x = x.float().to(args.device)
            if args.text == 0:
                y_pred = model(x)[0].cpu()
            else:
                y_pred = model(x,text)[0].cpu()
                # alpha, gamma, beta = model.film_alpha(x,text)
                # alpha = alpha.cpu()
                # gamma = gamma.cpu()
                # beta = beta.cpu()
                # all_alpha.append(alpha)

            if args.mask == 1:
                mask = y[:, :, :, 1] > 0.0
            else:
                mask = torch.ones_like(y[...,1],dtype=torch.bool)

            l1 = masked_loss(y_pred[:, :, :, 0], y[:, :, :, 0], mask, loss_type)
            l_sum1 += l1.item() * y.shape[0]
            if F_num ==2 or F_num == 3 :   ## congestion
                l2 = masked_loss(y_pred[:, :, :, 1], y[:, :, :, 1], mask, loss_type)
                l_sum2 += l2.item() * y.shape[0]

            # l2 = F.cross_entropy(congestion.reshape(-1,6), y[:,:,:,0].reshape(-1).long())
            n += y.shape[0]
        loss1 = l_sum1 / n

        # all_alpha = torch.cat(all_alpha, dim=0).numpy()  # (Total, L, N)
        # save_dir = os.path.join("..", "results")
        # os.makedirs(save_dir, exist_ok=True)
        # print("gamma mean/std:", gamma.mean().item(), gamma.std().item())
        # print("beta  mean/std:", beta.mean().item(), beta.std().item())
        # # ---- 1) 全局分布 ----
        # plt.hist(all_alpha.flatten(), bins=100)
        # plt.title("Global Alpha Distribution")
        # plt.savefig(os.path.join(save_dir, "alpha_hist.png"))
        # plt.close()
        #
        # # ---- 2) 节点平均 ----
        # top_k = 10
        # mean_alpha_node = all_alpha.mean(axis=(0, 1))  # (N,)
        # top_nodes = np.argsort(mean_alpha_node)[-top_k:]
        # plt.bar(range(top_k), mean_alpha_node[top_nodes])
        # plt.xticks(range(top_k), [f"node{n}" for n in top_nodes], rotation=45)
        # plt.title("Top Nodes by Mean Alpha")
        # plt.savefig(os.path.join(save_dir, "alpha_top_nodes.png"))
        # plt.close()
        #
        # # ---- 3) 时间平均 ----
        # mean_alpha_time = all_alpha.mean(axis=(0, 2))  # (L,)
        # plt.plot(mean_alpha_time)
        # plt.title("Mean Alpha over Time")
        # plt.xlabel("Time step")
        # plt.ylabel("Alpha")
        # plt.savefig(os.path.join(save_dir, "alpha_time.png"))
        # plt.close()

        if F_num == 1:
            return loss1, 0, 0
        if F_num ==2 or F_num == 3:
            loss2 = l_sum2 / n
            return loss1, loss2, 0


def evaluate_metric(args, model, data_iter, scaler, F_num):
    model.eval()

    with torch.no_grad():
        mae1 = mse1 = sum1 = count1 = 0.0
        mae1_20 = mse1_20 = sum1_20 = count1_20 = 0.0
        mae1_40 = mse1_40 = sum1_40 = count1_40 = 0.0
        if F_num == 2 or F_num == 3:
            mae2 = mse2 = sum2 = 0.0
            mae2_20 = mse2_20 = sum2_20  = 0.0
            mae2_40 = mse2_40 = sum2_40  = 0.0
        if F_num == 3:
            mae3 = mse3 = sum3 = count3 = 0.0
            mae3_20 = mse3_20 = sum3_20 = count3_20 = 0.0
            mae3_40 = mse3_40 = sum3_40 = count3_40 = 0.0


        for x, text, y in data_iter:
            x = x.float().to(args.device)
            B, T, N, D = y.shape  # [B, 15, 1260, 2]
            ### task1: congestion
            if args.mask == 1:
                mask1 = y[..., 1] > 0.0
            else:
                mask1 = torch.ones_like(y[..., 1], dtype=torch.bool)

            y = scaler.inverse_transform(y.cpu(),mask1).reshape(B, T, N, D)
            if args.text == 0:
                y_pred = scaler.inverse_transform(model(x)[0].cpu(),mask1).reshape(B, T, N, D)
            else:
                y_pred = scaler.inverse_transform(model(x, text)[0].cpu(),mask1).reshape(B, T, N, D)

            y = torch.tensor(y, dtype=torch.float32)
            y_pred = torch.tensor(y_pred, dtype=torch.float32)

            diff1 = torch.abs(y[..., 0] - y_pred[..., 0])
            mae1 += diff1[mask1].sum().item()
            mse1 += (diff1[mask1] ** 2).sum().item()
            sum1 += y[..., 0][mask1].sum().item()
            count1 += mask1.sum().item()

            mask1_20 = mask1[:, :5]
            mae1_20 += diff1[:, :5][mask1_20].sum().item()
            mse1_20 += (diff1[:, :5][mask1_20] ** 2).sum().item()
            sum1_20 += y[:, :5, :, 0][mask1_20].sum().item()
            count1_20 += mask1_20.sum().item()


            mask1_40 = mask1[:,:10]
            mae1_40 += diff1[:, :10][mask1_40].sum().item()
            mse1_40 += (diff1[:, :10][mask1_40] ** 2).sum().item()
            sum1_40 += y[:, :10, :, 0][mask1_40].sum().item()
            count1_40 += mask1_40.sum().item()

            # ### task2: speed
            if F_num == 2 or F_num == 3:
                diff2 = torch.abs(y[..., 1] - y_pred[..., 1])
                mae2 += diff2[mask1].sum().item()
                mse2 += (diff2[mask1] ** 2).sum().item()
                sum2 += y[..., 1][mask1].sum().item()

                # mask1_20 = mask1[:, :5]
                mae2_20 += diff2[:, :5][mask1_20].sum().item()
                mse2_20 += (diff2[:, :5][mask1_20] ** 2).sum().item()
                sum2_20 += y[:, :5, :, 1][mask1_20].sum().item()

                # mask1_40 = mask1[:,:10]
                mae2_40 += diff2[:, :10][mask1_40].sum().item()
                mse2_40 += (diff2[:, :10][mask1_40] ** 2).sum().item()
                sum2_40 += y[:, :10, :,1][mask1_40].sum().item()

            # ### task3: travel time = 3600 / speed
            if F_num == 3:
                # mask2 = y[..., 1] != 0.0
                speed_mask = y[..., 1] > 0.0
                # valid_mask_20 = valid_mask[:, :5]
                # valid_mask_40 = valid_mask[:, :10]
                #
                # speed_true = torch.clamp(y[..., 1], min=1e-3)
                # speed_pred = torch.clamp(y_pred[..., 1], min=1e-3)

                time_true = torch.zeros_like(y[..., 1])
                time_pred = torch.zeros_like(y_pred[..., 1])

                eps = 1.0  # km/h
                y_speed = torch.clamp(y[..., 1], min=eps)
                y_pred_speed = torch.clamp(y_pred[..., 1], min=eps)

                time_true[speed_mask] = 3600.0 / y_speed[speed_mask]
                time_pred[speed_mask] = 3600.0 / y_pred_speed[speed_mask]

                # print("min speed:", y[..., 1][speed_mask].min().item())
                # print("max speed:", y[..., 1][speed_mask].max().item())
                # print(time_true.mean(), time_pred.mean())
                diff3 = torch.abs(time_true - time_pred)

                # full
                mae3 += diff3[mask1].sum().item()
                mse3 += (diff3[mask1] ** 2).sum().item()
                sum3 += time_true[mask1].sum().item()

                # 20-horizon
                # valid_mask_20 = mask2[:, :5]
                mae3_20 += diff3[:, :5][mask1_20].sum().item()
                mse3_20 += (diff3[:, :5][mask1_20] ** 2).sum().item()
                sum3_20 += time_true[:, :5, :][mask1_20].sum().item()

                # 40-horizon
                # valid_mask_40 = mask2[:,:10]
                mae3_40 += diff3[:, :10][mask1_40].sum().item()
                mse3_40 += (diff3[:, :10][mask1_40] ** 2).sum().item()
                sum3_40 += time_true[:, :10, :][mask1_40].sum().item()
                # count3_40 += valid_mask_40.sum().item()

        # 汇总输出（每个 horizon 分开）
        def compute_metrics(sae, mse, total, count):
            if count == 0:
                return 0.0, 0.0, 0.0
            mae = sae / count
            rmse = (mse / count) ** 0.5
            wmape = sae / total
            return mae, rmse, wmape

        MAE1_20, RMSE1_20, WMAPE1_20 = compute_metrics(mae1_20, mse1_20, sum1_20, count1_20)
        MAE1_40, RMSE1_40, WMAPE1_40 = compute_metrics(mae1_40, mse1_40, sum1_40, count1_40)
        MAE1_60, RMSE1_60, WMAPE1_60 = compute_metrics(mae1, mse1, sum1, count1)
        if F_num == 1:
            return MAE1_20, RMSE1_20, WMAPE1_20, MAE1_40, RMSE1_40, WMAPE1_40, MAE1_60, RMSE1_60, WMAPE1_60

        if F_num == 2 :
            MAE2_20, RMSE2_20, WMAPE2_20 = compute_metrics(mae2_20, mse2_20, sum2_20, count1_20)
            MAE2_40, RMSE2_40, WMAPE2_40 = compute_metrics(mae2_40, mse2_40, sum2_40, count1_40)
            MAE2_60, RMSE2_60, WMAPE2_60 = compute_metrics(mae2, mse2, sum2, count1)
            return MAE1_20, RMSE1_20, WMAPE1_20, MAE1_40, RMSE1_40, WMAPE1_40, MAE1_60, RMSE1_60, WMAPE1_60,MAE2_20, RMSE2_20, WMAPE2_20, MAE2_40, RMSE2_40, WMAPE2_40, MAE2_60, RMSE2_60, WMAPE2_60

        if F_num == 3:
            MAE2_20, RMSE2_20, WMAPE2_20 = compute_metrics(mae2_20, mse2_20, sum2_20, count1_20)
            MAE2_40, RMSE2_40, WMAPE2_40 = compute_metrics(mae2_40, mse2_40, sum2_40, count1_40)
            MAE2_60, RMSE2_60, WMAPE2_60 = compute_metrics(mae2, mse2, sum2, count1)
            MAE3_20, RMSE3_20, WMAPE3_20 = compute_metrics(mae3_20, mse3_20, sum3_20, count1_20)
            MAE3_40, RMSE3_40, WMAPE3_40 = compute_metrics(mae3_40, mse3_40, sum3_40, count1_40)
            MAE3_60, RMSE3_60, WMAPE3_60 = compute_metrics(mae3, mse3, sum3, count1)

            return MAE1_20, RMSE1_20, WMAPE1_20, MAE1_40, RMSE1_40, WMAPE1_40, MAE1_60, RMSE1_60, WMAPE1_60, MAE2_20, RMSE2_20, WMAPE2_20, MAE2_40, RMSE2_40, WMAPE2_40, MAE2_60, RMSE2_60, WMAPE2_60, MAE3_20, RMSE3_20, WMAPE3_20, MAE3_40, RMSE3_40, WMAPE3_40, MAE3_60, RMSE3_60, WMAPE3_60






