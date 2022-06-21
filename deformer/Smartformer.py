import numpy as np
import sys
import time
import torch
import yaml

from deformer import CP_DEformer, CP_DEformer_
from deformer_dataset import DEformerDataset
from download_datasets import download_cifar10, download_mnist
from settings import *
from torch import nn, optim
from torch.utils.data import DataLoader

from cp_trans import AC, EncoderNet, DEVICE, judge_done
import tensorly as tl
from tensorly.decomposition import parafac

SEED = 2010
torch.manual_seed(SEED)
torch.set_printoptions(linewidth=160)
np.random.seed(SEED)


def init_datasets(opts):
    dataset = opts["train"]["dataset"]

    try:
        train_data = np.load(f"{dataset}_train.npy")
        test_data = np.load(f"{dataset}_test.npy")
    except FileNotFoundError:
        eval(f"download_{dataset}()")
        train_data = np.load(f"{dataset}_train.npy")
        test_data = np.load(f"{dataset}_test.npy")

    train_valid_idxs = np.arange(len(train_data))
    np.random.shuffle(train_valid_idxs)
    n_train = int(opts["train"]["train_prop"] * len(train_valid_idxs))
    train_idxs = train_valid_idxs[:n_train]
    valid_idxs = train_valid_idxs[n_train:]

    train_dataset = DEformerDataset(dataset, train_data[train_idxs], True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=opts["train"]["workers"],
    )
    valid_dataset = DEformerDataset(dataset, train_data[valid_idxs], False)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=2,
        num_workers=opts["train"]["workers"],
    )
    test_dataset = DEformerDataset(dataset, test_data, False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        num_workers=opts["train"]["workers"],
    )

    return (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    )


def init_model(opts, train_dataset):
    model_config = opts["model"]
    model_config["img_size"] = train_dataset.img_size
    model_config["pos_in_feats"] = 2 if opts["train"]["dataset"] == "mnist" else 3
    model_config["pixel_n"] = 1 if opts["train"]["dataset"] == "mnist" else 256
    model = CP_DEformer(**model_config)
    # model = CP_DEformer_(**model_config)
    return model


def get_preds_labels(tensors):
    preds = model(tensors)
    # labels = tensors["pixels"].flatten().to(device)
    labels = tensors["pixels"].to(device)
    if dataset == "mnist":
        # preds = preds.flatten()
        preds = preds.view(preds.size(0), preds.size(1))
    else:
        labels = labels.long()

    return (preds, labels)


def train_model():
    global score_mask, done_mask, Score, action_list, rank_list, state, next_state, next_info, min_index, action, reward, log, \
            last_train_loss, action_train_loss, action_step
    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    optimizer = optim.Adam(train_params, lr=opts["train"]["learning_rate"])
    if dataset == "mnist":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Continue training on a prematurely terminated model.
    try:
        model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))

        try:
            state_dict = torch.load(f"{JOB_DIR}/optimizer.pth")
            if opts["train"]["learning_rate"] == state_dict["param_groups"][0]["lr"]:
                optimizer.load_state_dict(state_dict)

        except ValueError:
            print("Old optimizer doesn't match.")

    except FileNotFoundError:
        pass

    best_train_loss = float("inf")
    best_valid_loss = float("inf")
    best_valid_loss = float("inf")
    test_loss_best_valid = float("inf")
    total_train_loss = None
    no_improvement = 0

    for epoch in range(50):
        print(f"\nepoch: {epoch}", flush=True)
        result.write('\nepoch:{}'.format(epoch))
        model.eval()
        total_valid_loss = 0.0

        with torch.no_grad():
            n_valid = 0
            for (valid_idx, valid_tensors) in enumerate(valid_loader):
                (preds, labels) = get_preds_labels(valid_tensors)

                loss = criterion(preds, labels)
                total_valid_loss += loss.item()
                n_valid += 1

            if dataset == "mnist":
                probs = 1 / (1 + (-preds).exp())
                preds = (probs > 0.5).int()

            else:
                probs = torch.softmax(preds, dim=1)
                (probs, preds) = probs.max(1)

            print(probs)
            print(preds)
            print(labels.int(), flush=True)
            result.write('probs:{}\n'.format(probs))
            result.write('preds:{}\n'.format(preds))
            result.write('labels:{}\n'.format(labels.int()))

            total_valid_loss /= n_valid

        if total_valid_loss < best_valid_loss:
            best_valid_loss = total_valid_loss
            no_improvement = 0
            torch.save(optimizer.state_dict(), f"{JOB_DIR}/optimizer.pth")
            torch.save(model.state_dict(), f"{JOB_DIR}/best_params.pth")

            test_loss_best_valid = 0.0
            with torch.no_grad():
                n_test = 0
                for (test_idx, test_tensors) in enumerate(test_loader):
                    (preds, labels) = get_preds_labels(test_tensors)
                    loss = criterion(preds, labels)
                    test_loss_best_valid += loss.item()
                    n_test += 1

            test_loss_best_valid /= n_test

        elif no_improvement < opts["train"]["patience"]:
            no_improvement += 1
            if no_improvement == opts["train"]["patience"]:
                print("Reducing learning rate.")
                for g in optimizer.param_groups:
                    g["lr"] *= 0.1

        print(f"total_train_loss: {total_train_loss}")
        print(f"best_train_loss: {best_train_loss}")
        print(f"total_valid_loss: {total_valid_loss}")
        print(f"best_valid_loss: {best_valid_loss}")
        print(f"test_loss_best_valid: {test_loss_best_valid}")

        result.write('total_train_loss: {}\n'.format(total_train_loss))
        result.write('best_train_loss: {}\n'.format(best_train_loss))
        result.write('total_valid_loss: {}\n'.format(total_valid_loss))
        result.write('best_valid_loss: {}\n'.format(best_valid_loss))
        result.write('test_loss_best_valid: {}\n'.format(test_loss_best_valid))


        model.train()
        total_train_loss = 0.0
        n_train = 0
        start_time = time.time()
        for (train_idx, train_tensors) in enumerate(train_loader):
            score_list = []
            if train_idx % 100 == 0:
                print(train_idx, flush=True)

            optimizer.zero_grad()
            (preds, labels) = get_preds_labels(train_tensors)
            loss = criterion(preds, labels)

            # compute layer importance
            if not score_mask.all() == done_mask.all():
                with torch.no_grad():
                    for layer in model.transformer.layers:
                        grad = torch.autograd.grad(loss, layer.attn.parameters(), retain_graph=True)
                        tuples = zip(grad, layer.attn.parameters())
                        importance = list(map(lambda p: (p[0] * p[1]).pow(2).sum(), tuples))
                        score_list.append(sum(importance))
                    score_list = torch.stack(score_list, dim=0).to(device)
                    Score += score_list

            total_train_loss += loss.item()
            action_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

            if train_idx % action_interval == 0 and train_idx > 1 and not score_mask.all() == done_mask.all():

                if action_step > 0:
                    ll = action_train_loss / action_interval
                    if ll <= 1.25 * last_train_loss:
                        reward += last_train_loss / ll
                    else:
                        reward += - ll / last_train_loss

                    ac.train_net(state, reward, next_state, action)
                    state = next_state
                    last_train_loss = ll
                    done = judge_done(action_list)
                    action_train_loss *= 0
                    if done:
                        log += 'The {}/{} Attn_layer R is {}, The action is {} \n'.format(min_index + 1, 6,
                                                                                          rank_list, action_list)
                        result.write('=' * 100 + '\n')
                        result.write('The {}/{} Attn_layer final R is {}, The action is {} \n'.format(min_index + 1,
                                                                                                 6,
                                                                                                 rank_list,
                                                                                                 action_list))
                        result.write('=' * 100 + '\n')
                        score_mask[min_index] = 1
                        score = Score.masked_fill(score_mask == 1, float('inf'))
                        min_index = torch.argmin(score).item()
                        state = next_info[0][min_index]
                        action_list = []
                        rank_list = []

                if not score_mask.all() == done_mask.all():
                    action_step += 1
                    Score /= action_interval

                    if action_step == 1:
                        min_index = torch.argmin(Score).item()
                        last_train_loss = action_train_loss / action_interval
                        state = arch_info[0][min_index]

                    action = ac.choose_action(state)
                    result.write('-' * 100 + '\n')
                    result.write('| Layer:{}, Action:{}, Rank:{}|\n'.format(min_index + 1, action, (action + 4) * 10))
                    result.write('-' * 100 + '\n')

                    print('-' * 100 + '\n')
                    print('| Layer:{}, Action:{}, Rank:{}|\n'.format(min_index + 1, action, (action + 4) * 10))
                    print('-' * 100 + '\n')

                    next_info, next_state, real_rank, reward = change(action, min_index)
                    action_list.append(action)
                    rank_list.append(real_rank)
                    Score *= 0.

        epoch_time = time.time() - start_time

        total_train_loss /= n_train
        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss

        torch.save(optimizer.state_dict(), f"{JOB_DIR}/final_optimizer.pth")
        torch.save(model.state_dict(), f"{JOB_DIR}/final_params.pth")

        print(f"epoch_time: {epoch_time:.2f}", flush=True)
        result.write('epoch_time:{:.2f}\n'.format(epoch_time))


def change_cpcores(model, action, reward):
    rank0 = model.W_Q0.shape[1]
    rank = (action + 4) * 10   # the smallest rank (40, 450)
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

def change(action, min_index):
    r = 0
    token = []
    real_ranks = []

    j = 0
    for layer in model.transformer.layers:
        if j == min_index:
            real_rank, r = change_cpcores(layer.attn, action, r)
        else:
            real_rank, r = change_cpcores(layer.attn, -4, r)
        token += ['MultiHeadAttention-0-8-64-%d' % real_rank]
        real_ranks.append(real_rank)
        j += 1

    next_info, _ = enc(token)
    next_state = next_info[0][min_index]

    return next_info, next_state, real_ranks[min_index], r


if __name__ == "__main__":
    JOB = sys.argv[1]
    # JOB = 'RL'
    EXPERIMENTS_DIR = './Deformer_experiments'
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    result = open('./Deformer_experiments/JOB/log.txt', 'w')

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    except IndexError:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))
    dataset = opts["train"]["dataset"]
    assert dataset in {"mnist", "cifar10"}

    # Initialize datasets.
    (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    ) = init_datasets(opts)

    # Initialize model.
    device = torch.device("cuda:0")
    model = init_model(opts, train_dataset).to(device)
    print(model)
    result.write('{}\n'.format(model))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")
    result.write('Parameters:{}\n'.format(n_params))

    action_interval = 5000
    action_step = 0
    token_list = ['MultiHeadAttention-0-8-64-450'] * 6
    enc = EncoderNet(512, 64)
    arch_info, (h, c) = enc(token_list)
    score_mask = torch.tensor([1.] * 6).cuda()
    done_mask = torch.tensor([1.] * 6).cuda()
    S = arch_info[0][0]
    ac = AC(S, 42)
    done = False

    Score = torch.tensor([0.] * 6, device=device)
    action_list = []
    rank_list = []
    state = 0
    next_state = 0
    next_info = 0
    min_index = 0
    action = -1
    reward = 0
    action_train_loss = 0
    last_train_loss = 0

    log = ''

    train_model()

    cp_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CP_Parameters: {cp_n_params}")
    result.write('CP_Parameters:{}\n'.format(cp_n_params))

    result.write('{}'.format(log))

