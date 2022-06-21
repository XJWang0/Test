import torch
import torch.nn as nn
import numpy as np
from .attention import DEVICE
import tensorly as tl
from tensorly.decomposition import parafac


def set_seed(seed):
    try:
        import torch
        torch.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed, details are ",e)

    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)


def judge_done(rank_list):
    if len(rank_list) < 5:
        done = False
        return done

    same = 1
    for i in range(4):
        if rank_list[-1] == rank_list[-2-i]:
            same += 1
        else:
            done = False
            return done
    done = True
    return done

def change_cpcores(model, action, reward):
    rank0 = model.W_Q0.shape[1]
    rank = action + 1
    # input_size = model.W_Q.shape[0]
    if rank0 == rank or rank == 0:
        return rank0, reward

    Wq0 = model.W_Q0.cpu().detach().numpy()
    Wq1 = model.W_Q1.cpu().detach().numpy()
    Wq2 = model.W_Q2.cpu().detach().numpy()

    Wk0 = model.W_K0.cpu().detach().numpy()
    Wk1 = model.W_K1.cpu().detach().numpy()
    Wk2 = model.W_K2.cpu().detach().numpy()

    Wv0 = model.W_V0.cpu().detach().numpy()
    Wv1 = model.W_V1.cpu().detach().numpy()
    Wv2 = model.W_V2.cpu().detach().numpy()

    weight = np.array([1.]*rank0)
    wq = tl.cp_to_tensor((weight, [Wq0, Wq1, Wq2]))
    wk = tl.cp_to_tensor((weight, [Wk0, Wk1, Wk2]))
    wv = tl.cp_to_tensor((weight, [Wv0, Wv1, Wv2]))
    Wq = tl.tensor(wq)
    Wk = tl.tensor(wk)
    Wv = tl.tensor(wv)
    while True:
        try:
            weight1, WQ = parafac(Wq, rank)
            weight2, WK = parafac(Wk, rank)
            weight3, WV = parafac(Wv, rank)
            break
        except:
            reward += -0.005


            model.W_Q0 = nn.Parameter(torch.tensor(Wq0, dtype=torch.float).to(DEVICE))
            model.W_Q1 = nn.Parameter(torch.tensor(Wq1, dtype=torch.float).to(DEVICE))
            model.W_Q2 = nn.Parameter(torch.tensor(Wq2, dtype=torch.float).to(DEVICE))

            model.W_K0 = nn.Parameter(torch.tensor(Wk0, dtype=torch.float).to(DEVICE))
            model.W_K1 = nn.Parameter(torch.tensor(Wk1, dtype=torch.float).to(DEVICE))
            model.W_K2 = nn.Parameter(torch.tensor(Wk2, dtype=torch.float).to(DEVICE))

            model.W_V0 = nn.Parameter(torch.tensor(Wv0, dtype=torch.float).to(DEVICE))
            model.W_V1 = nn.Parameter(torch.tensor(Wv1, dtype=torch.float).to(DEVICE))
            model.W_V2 = nn.Parameter(torch.tensor(Wv2, dtype=torch.float).to(DEVICE))

            return rank0, reward

    model.W_Q0 = nn.Parameter(torch.tensor(WQ[0], dtype=torch.float).to(DEVICE))
    model.W_Q1 = nn.Parameter(torch.tensor(WQ[1], dtype=torch.float).to(DEVICE))
    model.W_Q2 = nn.Parameter(torch.tensor(WQ[2], dtype=torch.float).to(DEVICE))

    model.W_K0 = nn.Parameter(torch.tensor(WK[0], dtype=torch.float).to(DEVICE))
    model.W_K1 = nn.Parameter(torch.tensor(WK[1], dtype=torch.float).to(DEVICE))
    model.W_K2 = nn.Parameter(torch.tensor(WK[2], dtype=torch.float).to(DEVICE))

    model.W_V0 = nn.Parameter(torch.tensor(WV[0], dtype=torch.float).to(DEVICE))
    model.W_V1 = nn.Parameter(torch.tensor(WV[1], dtype=torch.float).to(DEVICE))
    model.W_V2 = nn.Parameter(torch.tensor(WV[2], dtype=torch.float).to(DEVICE))

    return rank, reward

def step(model, encoder, action, last_averAcc, min_index):
    reward = 0
    token = []
    real_ranks = []

    j = 0
    for block in model.model.module.TransformerEncoder.layers:
        if j == min_index:
            real_rank, reward = change_cpcores(block.attn, action, reward)
        else:
            real_rank, reward = change_cpcores(block.attn, -1, reward)
        token += ['MultiHeadAttention-0-5-2-%d' % real_rank]
        real_ranks.append(real_rank)
        j += 1

    next_info, _ = encoder(token)
    bestAcc, averAcc, Y_true, Y_pred, Score = model.train()
    next_state = next_info[0][min_index]
    # train_acc = train_accs[-1]
    # acc = accs[-1]
    if last_averAcc - averAcc <= 0.1:
        reward += averAcc / last_averAcc
    else:
        reward += - last_averAcc / averAcc

    return next_info, next_state, bestAcc, averAcc, real_ranks[min_index], reward, Y_true, Y_pred, Score

