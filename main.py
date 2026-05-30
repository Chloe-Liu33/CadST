import logging
import os
import gc
import argparse
import math
import random
import warnings
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy.sparse as sp
import logging
import time
import signal
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import dataloader, utility, earlystopping, opt
from model import models
from model.GWavenet import gwnet
from model.DCRNN import DCRNNModel
from model.GAT import GATModel
import os
# from sklearn.preprocessing import StandardScaler
from script.utility import MaskedStandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
from OOD import lossb_expect_rff
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_parameters():
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--model', type=str, default="STGCN",help="name of model")
    parser.add_argument('--mask', type=int, default=1)
    parser.add_argument('--mode', type=str, default="all", help="all or test")
    parser.add_argument('--text_data',type=str, default="road", help="different content of text data")
    parser.add_argument('--text', type=int, default=0)
    parser.add_argument('--film', type=bool, default=False)
    parser.add_argument('--OOD', type=bool, default=True)

    parser.add_argument('--cross_att', type=bool, default=False)
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=33, help='42 or 33')
    parser.add_argument('--dataset', type=str, default='BjTT', choices=['BjTT','metr-la', 'pems-bay', 'pemsd7-m'])
    parser.add_argument('--n_his', type=int, default=15)
    parser.add_argument('--n_pred', type=int, default=15, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=4)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', choices=['cheb_graph_conv', 'graph_conv'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=64, help='2 for GAT, and 64 for others')
    parser.add_argument('--epochs', type=int, default=200, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'nadamw', 'lion'], help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))
    set_env(args.seed)

    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache() # Clean cache
    else:
        device = torch.device('cpu')
        gc.collect() # Clean cache
    args.device = device
    return args

def debug_normalization_stats(data, mask=None, name="data"):
    # data shape: (N, T, V, C)
    # N,T,V,C = data.shape
    C = 2
    flat = data.reshape(-1, 2).astype(float)       # (N*T*V, C)
    print("=== stats for", name, "===")
    overall_mean = np.mean(flat, axis=0)
    overall_std  = np.std(flat, axis=0)
    print("overall mean:", overall_mean)
    print("overall std: ", overall_std)

    # zeros proportion per feature
    zeros = np.sum(flat == 0, axis=0)
    total = flat.shape[0]
    print("zero counts:", zeros, "/", total, "->", zeros/total)

    # per-node std (to detect std==0)
    per_node_std = np.std(data, axis=(0,1))  # shape (V, C)
    print("nodes with std==0 per feature:", np.sum(per_node_std==0, axis=0))

    if mask is not None:
        flat_mask = mask.reshape(-1).astype(bool)
        for c in range(C):
            if flat_mask.sum() == 0:
                print(f"feature {c}: no observed values!")
                continue
            observed_mean = np.mean(flat[flat_mask, c])
            observed_std  = np.std(flat[flat_mask, c])
            print(f"feature {c} observed mean/std (exclude masked):", observed_mean, observed_std)

def data_preparate(args,hidden_size):
    adj, n_vertex = dataloader.load_adj()
    args.adj = adj
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(args.device)

    dataset_path = '../BjTT/'
    # dataset_path = os.path.join(dataset_path, args.dataset)
    # data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 70: 15: 15
    # val_and_test_rate = 0.15

    # len_val = int(math.floor(data_col * val_and_test_rate))
    # len_test = int(math.floor(data_col * val_and_test_rate))
    # len_train = int(data_col - len_val - len_test)

    # train, val, test = dataloader.load_data(args.dataset, len_train, len_val)
    if "test" in args.mode:
        train_num_path = os.path.join(dataset_path, 'train_Jan.txt')
    else:
        train_num_path = os.path.join(dataset_path, 'train_622.txt')

    test_num_path = os.path.join(dataset_path, 'test_622.txt')
    val_num_path = os.path.join(dataset_path, 'val_622.txt')
    train_num,val_num, test_num = dataloader.load_data(args.mode, train_num_path,val_num_path, test_num_path, 2)


    ## embedding
    if args.text_data == "events":
        if "test" in args.mode:
            train_emb_path = os.path.join(dataset_path, 'train_BERT_events_Jan.txt')
        else:
            train_emb_path = os.path.join(dataset_path, 'train_BERT_events_622.txt')

        test_emb_path = os.path.join(dataset_path, 'test_BERT_events_622.txt')
        val_emb_path = os.path.join(dataset_path, 'val_BERT_events_622.txt')

    else:
        if "test" in args.mode:
            train_emb_path = os.path.join(dataset_path, 'train_BERT_Jan.txt')
        else:
            train_emb_path = os.path.join(dataset_path, 'train_BERT_622.txt')
        test_emb_path = os.path.join(dataset_path, 'test_BERT_622.txt')
        val_emb_path = os.path.join(dataset_path, 'val_BERT_622.txt')

    train_emb, val_emb, test_emb = dataloader.load_text_data(args.mode, train_emb_path, val_emb_path, test_emb_path, hidden_size)


    print("starting normalizing data")
    if args.mask == 1:
        mask_train = train_num.reshape(-1,2)[:,1] > 0.0
        mask_val = val_num.reshape(-1, 2)[:, 1] > 0.0
        mask_test = test_num.reshape(-1, 2)[:, 1] > 0.0
        # debug_normalization_stats(train_num.reshape(-1,2),mask )
    else:
        mask_train = torch.ones_like(train_num.reshape(-1,2)[:,1],dtype=torch.bool)
        mask_val = torch.ones_like(val_num.reshape(-1,2)[:,1],dtype=torch.bool)
        mask_test = torch.ones_like(test_num.reshape(-1,2)[:,1],dtype=torch.bool)
    scaler = MaskedStandardScaler()
    scaler.fit(train_num,mask_train)
    ## lots of 0 values in Feb should be mask
    train_num = scaler.transform(train_num,mask_train)
    train_num = train_num.reshape(-1, 30, 1260,2) #
    # val = zscore.transform(val)
    val_num = scaler.transform(val_num, mask_val)
    val_num = val_num.reshape(-1, 30, 1260, 2)  #
    test_num = scaler.transform(test_num, mask_test)
    test_num = test_num.reshape(-1, 30, 1260,2)

    # print("normalization data", train_num[...,0].mean(), train_num[...,1].mean())

    x_train_num, y_train = dataloader.data_transform(train_num, args.n_his, args.n_pred)
    x_val_num, y_val = dataloader.data_transform(val_num, args.n_his, args.n_pred)
    x_test_num, y_test = dataloader.data_transform(test_num, args.n_his, args.n_pred)

    x_train_emb, _ = dataloader.data_transform(train_emb, args.n_his, args.n_pred)
    x_val_emb, _ = dataloader.data_transform(val_emb, args.n_his, args.n_pred)
    x_test_emb, _ = dataloader.data_transform(test_emb, args.n_his, args.n_pred)

    train_data = utils.data.TensorDataset(x_train_num, x_train_emb, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_data = utils.data.TensorDataset(x_val_num,x_val_emb, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test_num, x_test_emb, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, scaler, train_iter, val_iter, test_iter

def prepare_model(args, n_vertex):
    # loss = nn.MSELoss()
    # loss = utility.masked_loss()
    es = earlystopping.EarlyStopping(delta=0.0, 
                                     patience=args.patience, 
                                     verbose=True, 
                                     path=args.model + args.dataset + ".pt")

    if args.model == "STGCN":
        Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

        # blocks: settings of channel size in st_conv_blocks and output layer,
        # using the bottleneck design in st_conv_blocks
        blocks = []
        blocks.append([2])
        for l in range(args.stblock_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([2])

        type_loss = "mse"
        if args.graph_conv_type == 'cheb_graph_conv':
            model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(args.device)

        else:
            model = models.STGCNGraphConv(args, blocks, n_vertex).to(args.device)
    elif args.model == "GWavenet":
        def asym_adj(adj):
            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1)).flatten()
            # print(rowsum)
            d_inv = np.power(rowsum.astype(float), -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            return d_mat.dot(adj).astype(np.float32).todense()
        type_loss = "mae"
        adj_mx = [asym_adj(args.adj), asym_adj(np.transpose(args.adj))]
        supports = [torch.tensor(i).to(args.device) for i in adj_mx]
        adjinit = None
        model = gwnet(args, n_vertex, 0.3, supports=supports, gcn_bool=True, addaptadj=True, aptinit=adjinit, in_dim=2, out_dim=2, residual_channels=32, dilation_channels=32, skip_channels=32 * 8, end_channels=32 * 16).to(args.device)

    elif args.model == "DCRNN":
        type_loss = "mae"
        logger = logging.getLogger("dcrnn")
        logger.setLevel(logging.DEBUG)
        args.learning_rate = 0.01
        model = DCRNNModel(
            adj_mx=args.adj,
            logger=logger,
            num_nodes=1260,
            rnn_units=64,  # RNN hidden size
            input_dim=2,  # 输入维度 (比如 speed, congestion)
            seq_len=15,  # encoder输入序列长度
            output_dim=2,  # 预测维度 (比如预测 speed)
            horizon=15,  # decoder预测步长
            num_rnn_layers=2,  #
            max_diffusion_step=2,  # 图卷积扩散步数
            filter_type="laplacian",
            cl_decay_steps=1000,
            use_curriculum_learning=True
        ).to(args.device)
    elif args.model == "GAT":
        type_loss = "mae"
        model = GATModel(args,1260, 15, 16,
                    2, 64,
                    15*2, 0.5).to(args.device)

    # all parameters
    # dummy_input = torch.zeros(16, 15, 1260, 2).to(args.device)  # seq_len, batch, num_nodes*input_dim
    # y = dummy_input
    # batches_seen = 0
    # model(dummy_input,labels = y, batches_seen = batches_seen)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # training parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    if args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "nadamw":
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, decoupled_weight_decay=True)
    elif args.opt == "lion":
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        raise ValueError(f'ERROR: The {args.opt} optimizer is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return es, model, optimizer, scheduler, type_loss

def train(args, model, optimizer, scheduler, es, train_iter,val_iter,F_num, type_loss):
    batches_seen = 0
    start_time = time.time()
    epoch_num = 0
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        l_sum1,l_sum2, l_sum3, n = 0.0, 0.0, 0.0,  0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()

        for x_t,text, y in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            x_t = x_t.float().to(args.device)
            y = y.float().to(args.device)
            if args.text == 0:
                if args.OOD:
                    y_pred, x_embedding = model(x_t)  # [batch_size, num_nodes]
                    loss_ood = lossb_expect_rff(cfeaturec=x_embedding, num_f=1,scale_factor=1e-4)

                elif args.model == "DCRNN":
                    y_pred = model(x_t,labels = y,batches_seen=batches_seen)
                    batches_seen += 1
                    y = y
                else:
                    y_pred = model(x_t)
            else:
                if args.OOD:
                    y_pred, x_embedding = model(x_t,text)  # [batch_size, num_nodes]
                    loss_ood = lossb_expect_rff(cfeaturec=x_embedding, num_f=1, scale_factor=1e-4)

                else:
                    y_pred = model(x_t,text)
            if args.mask == 1:
                mask = y[..., 1] > 0.0
            else:
                mask = torch.ones_like(y[...,1],dtype=torch.bool)
            # print(y.device,y_pred.device,mask.device)
            l1 = utility.masked_loss(y_pred[:,:,:,0], y[:,:,:,0],mask, loss_type=type_loss)
            l1_val = l1.item()
            l_sum1 += l1_val * y.shape[0]
            if F_num == 1:
                l = l1
                l_sum = l_sum1
            elif F_num == 2:
                l2 = utility.masked_loss(y_pred[:,:,:,1], y[:,:,:,1], mask, loss_type=type_loss)
                l2_val = l2.item()
                l_sum2 += l2_val * y.shape[0]
                l = l1 + l2
                l_sum = l_sum1 + l_sum2
            else:

                l2 = utility.masked_loss(y_pred[:, :, :, 1], y[:, :, :, 1], mask, loss_type=type_loss)
                l2_val = l2.item()
                l_sum2 += l2_val * y.shape[0]
                l = l1 * 0.5 + l2
                l_sum = l_sum1 * 0.5 + l_sum2

                mask_speed = y[..., 1] > 0.0

                time_true = torch.where(mask_speed, 3600.0 / y[..., 1], torch.zeros_like(y[..., 1]))
                time_pred = torch.where(mask_speed, 3600.0 / y_pred[..., 1], torch.zeros_like(y_pred[..., 1]))
                # time_true = time_pred = torch.zeros_like(y[..., 1], dtype=y_pred.dtype)
                # time_true[mask_speed] = 3600.0/ y[..., 1][mask_speed].to(time_pred.dtype)
                # time_pred[mask_speed] = 3600.0 / y_pred[..., 1][mask_speed].to(time_pred.dtype)
                l3 = utility.masked_loss(time_pred, time_true, mask, loss_type=type_loss)
                l3_val = l3.item()
                l_sum3 += l3_val * y.shape[0]

            if args.OOD:
                print(f"loss: {l:.6e}  loss_ood: {loss_ood:.6e}")
                l = l + loss_ood ## 0.2, 0.002
            l.backward()
            optimizer.step()
            n += y.shape[0]


        scheduler.step()
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |  GPU occupy: {:.2f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], gpu_mem_alloc))
        if F_num == 1:
            print('|Train loss = congestion level : {:.4}'. \
            format(l_sum / n))
        elif F_num == 2:
            print('|Train loss: {:.4f} |congestion : {:.4}| speed : {:.4}'. \
              format(l_sum / n, l_sum1 / n, l_sum2 / n))
        else:


            print('|Train loss: {:.4f} |congestion : {:.4}| speed : {:.4} | time : {:.4}'. \
                  format(l_sum / n, l_sum1 / n, l_sum2 / n, l_sum3 / n))

        # es(val_loss, model)
        epoch_num += 1
        train_loss = val(args, model, train_iter, F_num)
        val_loss = val(args, model, val_iter, F_num)
        # es(torch.tensor(l_sum/n),model)
        es(torch.tensor(val_loss), model)
        if es.early_stop and epoch > 50:
            print("Early stopping")
            torch.save(model.state_dict(), f"checkpoints/{args.model}/text{args.text}_cross_att{args.cross_att}_film{args.film}_mask{args.mask}_text{args.text_data}_{args.mode}.pth")
            break
        elif epoch == args.epochs -1:
            torch.save(model.state_dict(), f"checkpoints/{args.model}/text{args.text}_cross_att{args.cross_att}_film{args.film}_mask{args.mask}_text{args.text_data}_{args.mode}.pth")


        # elif epoch > 50 and epoch % 10 == 0:
        #     torch.save(model.state_dict(), f"checkpoints/{args.model}/text{args.text}_cross_att{args.cross_att}_film{args.film}_mask{args.mask}_text{args.text_data}_epoch{epoch}_{args.mode}.pth")
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/loss_{args.model}.png", dpi=300)
    plt.close()

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_epoch = total_time /  max(epoch_num, 1)
    print(f"Total training time: {total_time:.2f} s")
    print(f"Average time per epoch: {avg_time_per_epoch:.2f} s")
    # torch.save(model.state_dict(), f"checkpoints/{args.model}/text{args.text}_cross_att{args.cross_att}_film{args.film}_mask{args.mask}.pth")



@torch.no_grad()
def val(args, model, val_iter, F_num):
    # model.load_state_dict(torch.load(
    #     f"checkpoints/{args.model}/text{args.text}_cross_att{args.cross_att}_film{args.film}_mask{args.mask}_text{args.text_data}_{args.mode}.pth"))
    model.eval()
    test_loss1, test_loss2, test_loss3 = utility.evaluate_model(args, model, val_iter, F_num, loss_type=type_loss)
    loss = test_loss1 *0.5 + test_loss2
    # l_sum, n = 0.0, 0
    # for x,y in val_iter:
    #     y_pred = model(x,  F_num).view(len(x), -1)
    #     l = loss(y_pred, y, loss_type="mse")
    #     l_sum += l.item() * y.shape[0]
    #     n += y.shape[0]
    # return torch.tensor(l_sum / n)
    return loss

@torch.no_grad() 
def test(zscore, model, test_iter, args, F_num, type_loss):
    model.load_state_dict(torch.load(f"checkpoints/{args.model}/text{args.text}_cross_att{args.cross_att}_film{args.film}_mask{args.mask}_text{args.text_data}_{args.mode}.pth"))
    model.eval()
    test_loss1, test_loss2, test_loss3 = utility.evaluate_model(args, model, test_iter, F_num, loss_type=type_loss)
    start_time = time.time()
    if F_num == 1:
        MAE1_20, RMSE1_20, WMAPE1_20,  MAE1_40,RMSE1_40, WMAPE1_40,  MAE1_60, RMSE1_60, WMAPE1_60 = utility.evaluate_metric(args, model, test_iter, zscore, F_num)
        print(f'60 min | Test loss = congestion loss {test_loss1:.4f} ')
        print(f'20 min | MAE_congestion: {MAE1_20:.4f} | RMSE_congestion: {RMSE1_20:.4f} | WMAPE_congestion: {WMAPE1_20:.4f} ')
        print(f'40 min | MAE_congestion: {MAE1_40:.4f} | RMSE_congestion: {RMSE1_40:.4f} | WMAPE_congestion {WMAPE1_40:.4f} ')
        print(f'60 min | MAE_congestion: {MAE1_60:.4f} | RMSE_congestion: {RMSE1_60:.4f} | WMAPE_congestion: {WMAPE1_60:.4f} ')
    elif F_num == 2:
        print(f'60 min | Test loss {test_loss1 + test_loss2:.4f} | congestion: {test_loss1:.4f} | speed: {test_loss2:.4f}')
        MAE1_20, RMSE1_20, WMAPE1_20,  MAE1_40,RMSE1_40, WMAPE1_40, MAE1_60, RMSE1_60, WMAPE1_60,MAE2_20, RMSE2_20, WMAPE2_20,  MAE2_40,RMSE2_40, WMAPE2_40,  MAE2_60, RMSE2_60, WMAPE2_60 = utility.evaluate_metric(args,model, test_iter, zscore, F_num)
        print(f'20 min | MAE_congestion {MAE1_20:.4f} | RMSE_congestion {RMSE1_20:.4f} | WMAPE_congestion {WMAPE1_20:.4f}')
        print(f'20 min | MAE_speed: {MAE2_20:.4f} | RMSE_speed: {RMSE2_20:.4f} | WMAPE_speed: {WMAPE2_20:.4f} ')
        print(f'40 min | MAE_congestion {MAE1_40:.4f} | RMSE_congestion {RMSE1_40:.4f} | WMAPE_congestion {WMAPE1_40:.4f}')
        print(f'40 min | MAE_speed: {MAE2_40:.4f} | RMSE_speed: {RMSE2_40:.4f} | WMAPE_speed: {WMAPE2_40:.4f} ')
        print(f'60 min | MAE_congestion {MAE1_60:.4f} | RMSE_congestion {RMSE1_60:.4f} | WMAPE_congestion {WMAPE1_60:.4f}')
        print(f'60 min | MAE_speed: {MAE2_60:.4f} | RMSE_speed: {RMSE2_60:.4f} | WMAPE_speed: {WMAPE2_60:.4f} ')

    else:
        print(f'60 min | Test loss {test_loss1 + test_loss2 +test_loss3:.4f} | congestion: {test_loss1:.4f} | speed: {test_loss2:.4f} | time: {test_loss3:.4f}')


        MAE1_20, RMSE1_20, WMAPE1_20,  MAE1_40,RMSE1_40, WMAPE1_40, MAE1_60, RMSE1_60, WMAPE1_60,MAE2_20, RMSE2_20, WMAPE2_20,  MAE2_40,RMSE2_40, WMAPE2_40,  MAE2_60, RMSE2_60, WMAPE2_60, MAE3_20, RMSE3_20, WMAPE3_20, MAE3_40, RMSE3_40, WMAPE3_40, MAE3_60, RMSE3_60, WMAPE3_60 = utility.evaluate_metric(args, model, test_iter, zscore, F_num)


        print(f'20 min | MAE_congestion {MAE1_20:.4f} | RMSE_congestion {RMSE1_20:.4f} | WMAPE_congestion {WMAPE1_20:.4f}')
        print(f'20 min | MAE_speed: {MAE2_20:.4f} | RMSE_speed: {RMSE2_20:.4f} | WMAPE_speed: {WMAPE2_20:.4f} ')
        print(f'20 min | MAE_time: {MAE3_20:.4f} | RMSE_time: {RMSE3_20:.4f} | WMAPE_time: {WMAPE3_20:.4f} ')

        print(f'40 min | MAE_congestion {MAE1_40:.4f} | RMSE_congestion {RMSE1_40:.4f} | WMAPE_congestion {WMAPE1_40:.4f}')
        print(f'40 min | MAE_speed: {MAE2_40:.4f} | RMSE_speed: {RMSE2_40:.4f} | WMAPE_speed: {WMAPE2_40:.4f} ')
        print(f'40 min | MAE_time: {MAE3_40:.4f} | RMSE_time: {RMSE3_40:.4f} | WMAPE_time: {WMAPE3_40:.4f} ')

        print(f'60 min | | MAE_congestion {MAE1_60:.4f} | RMSE_congestion {RMSE1_60:.4f} | WMAPE_congestion {WMAPE1_60:.4f}')
        print(f'60 min | MAE_speed: {MAE2_60:.4f} | RMSE_speed: {RMSE2_60:.4f} | WMAPE_speed: {WMAPE2_60:.4f} ')
        print(f'60 min | MAE_time: {MAE3_60:.4f} | RMSE_time: {RMSE3_60:.4f} | WMAPE_time: {WMAPE3_60:.4f} ')
    end_time = time.time()
    total_time = end_time - start_time
    samples_num = 6422
    avg_time_per_batch = total_time / (samples_num/args.batch_size)
    avg_time_per_sample = total_time / samples_num
    throughput = args.batch_size / avg_time_per_batch

    print(f"Average inference time per batch: {avg_time_per_batch * 1000:.2f} ms")
    print(f"Average inference time per sample: {avg_time_per_sample * 1000:.4f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")
if __name__ == "__main__":
    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)
    F_num=3
    hidden_size = 768
    now = datetime.now()
    print("time for starting training:", now.strftime("%Y-%m-%d %H:%M:%S"))

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_parameters()
    print("Start preparing data")
    n_vertex, scaler, train_iter,val_iter, test_iter = data_preparate(args,hidden_size)
    print("Start preparing model")
    es, model, optimizer, scheduler, type_loss = prepare_model(args, n_vertex)
    print("Start training data")

    checkpoint_path = f"checkpoints/{args.model}/text{args.text}_cross_att{args.cross_att}_film{args.film}_mask{args.mask}.pth"
    def save_checkpoint(signum, frame):
        print(f"Received signal {signum}, saving checkpoint...")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        sys.exit(0)  # 停止训练
    signal.signal(signal.SIGTERM, save_checkpoint)
    signal.signal(signal.SIGINT, save_checkpoint)  # 支持 Ctrl+C 手动终止

    train(args, model, optimizer, scheduler, es, train_iter,val_iter, F_num, type_loss)
    print("Start testing data")
    test(scaler, model, test_iter, args, F_num, type_loss)
