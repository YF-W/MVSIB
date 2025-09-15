import argparse
import random
from network import Network
from metric import *
from model import *
from loss import *
from logger import Logger
import datetime
import os
import torch
import numpy as np

Dataname = 'RGB-D'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname,
                    help='[CCV, RGB-D, Cora, Hdigit, prokaryotic]')
parser.add_argument('--save_model', default=False, help='Saving the model after training.')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5, type=float)
parser.add_argument("--temperature_l", default=0.5, type=float)
parser.add_argument("--learning_rate", default=0.0001, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--mse_epochs", default=200, type=int)
parser.add_argument("--con_epochs", default=100, type=int)
parser.add_argument("--feature_dim", default =256, type=int)
parser.add_argument("--large_datasets", default=False, type=lambda x: x.lower()=='true')
parser.add_argument("--k", default=5, type=int)
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx.')
# 以下是我们新增的动态策略超参
parser.add_argument('--warmup_epochs', default=20, type=int)
parser.add_argument('--lambda_u', default=0.1, type=float)
parser.add_argument('--lambda_hn_penalty',type=float,default=0.1)
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # — 数据集特定超参 —
    if args.dataset == "CCV":
        args.seed, args.k, alpha, beta = 10, 10, 0.0001, 0.001
        args.seed, args.k, alpha, beta = 10, 4, 0.01, 0.1
    elif args.dataset == "RGB-D":
        args.seed, args.k, alpha, beta = 5, 10, 0.01, 1
    elif args.dataset == "Cora":
        args.seed, args.k, alpha, beta = 10, 10, 0.01, 0.1
        args.con_epochs = 100
    elif args.dataset == "Hdigit":
        args.large_datasets = True
        args.seed, args.k, alpha, beta = 10, 5, 1, 0.1
    elif args.dataset == "prokaryotic":
        args.seed, args.k, alpha, beta = 10, 5, 0.01, 0.1

    print("==================================\nArgs:{}\n==================================".format(args))
    set_seed(args.seed)
    # — 准备数据和模型 —
    mv_data = MultiviewData(args.dataset, device)
    num_views = len(mv_data.data_views)
    num_samples = mv_data.labels.size
    num_clusters = int(np.unique(mv_data.labels).size)
    input_sizes = [mv_data.data_views[i].shape[1] for i in range(num_views)]

    network = Network(num_views, num_samples, num_clusters, device,
                      input_sizes, args.feature_dim).to(device)

    optimizer = torch.optim.Adam(
        list(network.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    mvc_loss = Loss(args.batch_size, num_clusters,
                    args.temperature_l, args.temperature_f).to(device)

    nowtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logger = Logger(f"{args.dataset}=={nowtime}")
    logger.info("Args: " + str(args))

    # — Warm-up 预训练 —
    epoch_list = []
    totalloss_list = []
    pre_train(network, mv_data, args.batch_size,
              args.mse_epochs, optimizer)

    best_acc = 0
    best_epoch = -1
    best_metrics = None
    acc_list, nmi_list, pur_list, ari_list, f1_list = [], [], [], [], []

    if not args.large_datasets:
        W = get_W(mv_data, k=args.k)
        mv_loader, _, _, _ = get_multiview_data(mv_data, args.batch_size)

        for epoch in range(1, args.con_epochs + 1):
            total_loss = contrastive_train(
                network, mv_data, mvc_loss,
                args.batch_size, epoch, W,
                alpha, beta,
                optimizer,
                args.warmup_epochs,

                args.lambda_u,
                args.temperature_f, args.lambda_hn_penalty
            )

            epoch_list.append(epoch)
            totalloss_list.append(total_loss)

            # 每轮评估
            acc, nmi, pur, ari, f_score = valid(network, mv_data, num_samples, num_clusters)
            logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
            print(f"[Epoch {epoch}] ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")

            acc_list.append(acc)
            nmi_list.append(nmi)
            pur_list.append(pur)
            ari_list.append(ari)
            f1_list.append(f_score)

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_metrics = (acc, nmi, pur, ari, f_score)

        avg_acc = sum(acc_list) / len(acc_list)
        avg_nmi = sum(nmi_list) / len(nmi_list)
        avg_pur = sum(pur_list) / len(pur_list)
        avg_ari = sum(ari_list) / len(ari_list)
        avg_f1 = sum(f1_list) / len(f1_list)
        logger.info(" Average over all epochs::")
        logger.info(f"ACC={avg_acc:.4f} NMI={avg_nmi:.4f} PUR={avg_pur:.4f} ARI={avg_ari:.4f} F1={avg_f1:.4f}")
        print("\n Average over all epochs:")
        print(f"AVG ACC = {avg_acc:.4f}  AVG NMI = {avg_nmi:.4f}  AVG PUR = {avg_pur:.4f}  "
              f"AVG ARI = {avg_ari:.4f}  AVG F1 = {avg_f1:.4f}")

        # 最后一轮评估
        print("\n Final Evaluation (Last Epoch):")
        acc, nmi, pur, ari, f_score = valid(network, mv_data, num_samples, num_clusters)
        logger.info("Final Evaluation (Last Epoch):")
        logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
        print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} F1 = {:.4f}'.format(
            acc, nmi, pur, ari, f_score))

        # 最优一轮
        if best_metrics:
            acc, nmi, pur, ari, f_score = best_metrics
            logger.info(f"Best Evaluation (Epoch {best_epoch}):")
            logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
            print(f"\n Best Evaluation (Epoch {best_epoch}):")
            print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} F1 = {:.4f}'.format(
                acc, nmi, pur, ari, f_score))


        # — 保存模型 —
        if args.save_model:
            torch.save({
                'network_state_dict': network.state_dict(),
            }, f'./models/{args.dataset}_complete_model.pth')
    else:
        best_acc = 0
        best_epoch = -1
        best_metrics = None

        for epoch in range(1, args.con_epochs + 1):
            total_loss = contrastive_largedatasetstrain(
                network, mv_data, mvc_loss,
                args.batch_size, epoch,
                args.k, alpha, beta, optimizer
            )


            acc, nmi, pur, ari, f_score = valid(network, mv_data, num_samples, num_clusters)
            logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
            print(f"[Epoch {epoch}] ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")

            acc_list.append(acc)
            nmi_list.append(nmi)
            pur_list.append(pur)
            ari_list.append(ari)
            f1_list.append(f_score)

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_metrics = (acc, nmi, pur, ari, f_score)

        avg_acc = sum(acc_list) / len(acc_list)
        avg_nmi = sum(nmi_list) / len(nmi_list)
        avg_pur = sum(pur_list) / len(pur_list)
        avg_ari = sum(ari_list) / len(ari_list)
        avg_f1 = sum(f1_list) / len(f1_list)
        logger.info(" Average over all epochs::")
        logger.info(f"ACC={avg_acc:.4f} NMI={avg_nmi:.4f} PUR={avg_pur:.4f} ARI={avg_ari:.4f} F1={avg_f1:.4f}")
        print("\n Average over all epochs:")
        print(f"AVG ACC = {avg_acc:.4f}  AVG NMI = {avg_nmi:.4f}  AVG PUR = {avg_pur:.4f}  "
              f"AVG ARI = {avg_ari:.4f}  AVG F1 = {avg_f1:.4f}")

        print("\n Final Evaluation (Last Epoch):")
        acc, nmi, pur, ari, f_score = valid(network, mv_data, num_samples, num_clusters)
        logger.info("Final Evaluation (Last Epoch):")
        logger.info(f"ACC={acc:.4f} NMI={nmi:.4f} PUR={pur:.4f} ARI={ari:.4f} F1={f_score:.4f}")
        print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} F1 = {:.4f}'.format(
            acc, nmi, pur, ari, f_score))

        if best_metrics:
            acc, nmi, pur, ari, f_score = best_metrics
            print(f"\n Best Evaluation (Epoch {best_epoch}):")
            print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} F1 = {:.4f}'.format(
                acc, nmi, pur,    ari, f_score))