#需要发布的代码
import warnings
from time import time
import logging
import os
import os.path as osp
import numpy as np

import torch
from torch import nn
from torch_geometric.nn import GINConv, GCNConv
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch.autograd import Variable

import random
from utils import stat_graph

import sys
# sys.path.append('/root/anaconda3/envs/gcl/lib/python3.9/site-packages')
sys.path.append('/home/jianwei/gcl/PyGCL-main')
from GCL.models import DualBranchContrast
import GCL.augmentors as A
from losses import SupConMixLoss_N,SupConMixLoss_G
from GCL.eval import get_split, SVMEvaluator


import argparse
import statistics
from ogb.graphproppred import PygGraphPropPredDataset


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d')

class LinearClassifierWithSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifierWithSoftmax, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        probs = F.softmax(x, dim=1)
        return probs


def prepare_dataset_x(dataset):
    if dataset[0].x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max( max_degree, degs[-1].max().item() )
            data.num_nodes = int( torch.max(data.edge_index) ) + 1

        if max_degree < 2000:

            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = F.one_hot(degs, num_classes=max_degree+1).to(torch.float)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = ( (degs - mean) / std ).view( -1, 1 )
    return dataset



def prepare_dataset_onehot_y(dataset):

    y_set = set()
    for data in dataset:
        y_set.add(int(data.y))
    num_classes = len(y_set)

    for data in dataset:
        data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]
    return dataset


def mixup_cross_entropy_loss(input, target, size_average=True):
    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


def make_gcn_conv(input_dim, out_dim):
    return GCNConv(input_dim, out_dim)

def hasNoNan(tensor:torch.FloatTensor):
    return tensor.isnan().sum() == 0
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gcn_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gcn_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential( 
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        x = x.float()
        z = x
        # zs = [z]
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_mean_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        z = F.normalize(z)
        g = F.normalize(g)
        
        return z, g


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential( 
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        x = x.float()
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        
        gs = [global_mean_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        z = F.normalize(z)
        g = F.normalize(g)
        
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)

        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2 = z1
        g2 = g1
        return z, g, z1, z2, g1, g2

def cutout(S_v, z1, z2, batch,saliency_ratio):
    sorted_indices = torch.argsort(S_v, descending=True)
    top_80_percent_idx = int(saliency_ratio * len(S_v)) 
    threshold = S_v[sorted_indices[top_80_percent_idx]]
    mask = S_v > threshold
    mask = mask.float().to(z1.device)
    
    result = mask * z1 + (1 - mask) * z2
    g3 = global_mean_pool(result, batch)
    g3 = F.normalize(g3)
    return g3, mask  

def mixup(g1, lambda_value,labels,seed=None):
    seed  = 22
    np.random.seed(seed)
    mapping = np.random.permutation(g1.shape[0])
    reordered_g1 = g1[mapping]
    reordered_labels = labels[mapping]
    g_prime = lambda_value * g1 + (1 - lambda_value) * reordered_g1
    labels_prime = lambda_value *labels +(1 - lambda_value) * reordered_labels
    
    return g_prime,labels_prime,mapping

def train(encoder_model, criterion,train_loader,optimizer,saliency):
    encoder_model.train()
    loss_all = 0
    graph_all = 0
    losses = []
    if saliency and epoch >= args.warmup:
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            #转变为浮点型
            data.x = data.x.float()

            # Hook for saliency
            data.x.requires_grad_()

            _, _, z1, z2, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
            g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
            labels = data.y.view(-1, num_classes)

            features1 = torch.stack([g1, g2], dim=1)
            loss1 = criterion(features1, labels)

            # Compute gradients without backward using torch.autograd.grad
            gradients = torch.autograd.grad(loss1, data.x, retain_graph=True, create_graph=True)[0]
            S_v = torch.norm(gradients, dim=1, keepdim=True)

            g3, mask = cutout(S_v, z1, z2, data.batch, args.saliency_ratio)
            features2 = torch.stack([g1, g2, g3], dim=1)
            loss = criterion(features2, labels)
            loss.backward()
    else:
        for data in train_loader:
            # print( "data.y", data.y )
            data = data.to(device)
            optimizer.zero_grad()
            #转变为浮点型
            data.x = data.x.float()

            _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
            
            labels = data.y.view(-1, num_classes)

            if isinstance(criterion, SupConMixLoss_G):
                g1_prime,labels_prime,mapping = mixup(g1, args.lambda_value, labels, seed=None)
                g1_prime, g2 = [encoder_model.encoder.project(g) for g in [g1_prime, g2]]
                features = torch.stack([g1_prime, g2], dim=1)
                loss = criterion(features, labels,labels_prime,mapping)

            elif isinstance(criterion, SupConMixLoss_N):
                if args.augsup:
                    g1_prime,labels_prime,mapping = mixup(g1, args.lambda_value, labels, seed=None)
                    g1_prime, g2 = [encoder_model.encoder.project(g) for g in [g1_prime, g2]]
                    features = torch.stack([g1_prime, g2], dim=1)
                    loss = criterion(features, labels)
                else:
                    g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
                    features = torch.stack([g1, g2], dim=1)
                    loss = criterion(features, labels)
            loss.backward()

    torch.nn.utils.clip_grad_norm_(list(encoder_model.parameters()), max_norm=5, norm_type=2)
    loss_all += loss.item() * data.num_graphs
    graph_all += data.num_graphs
    optimizer.step()
    # 记录损失值
    losses.append(loss.item())
    loss_avg = loss_all / graph_all
    return encoder_model, loss_avg
# def train(encoder_model, criterion,train_loader,optimizer):
#     encoder_model.train()
#     loss_all = 0
#     graph_all = 0
#     losses = []
#     for data in train_loader:
#         # print( "data.y", data.y )
#         data = data.to(device)
#         optimizer.zero_grad()

#         _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
#         g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        
#         labels = data.y.view(-1, num_classes)
    
#         features = torch.stack([g1, g2], dim=1)
        
#         loss = criterion(features, labels)

#         loss.backward()

#         torch.nn.utils.clip_grad_norm(encoder_model.parameters(), max_norm=5, norm_type=2)
#         loss_all += loss.item() * data.num_graphs
#         graph_all += data.num_graphs
#         optimizer.step()
#         # 记录损失值
#         losses.append(loss.item())

#         # 打印当前损失值
#         # print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item()}")
#     loss = loss_all / graph_all
#     return model, loss

@torch.no_grad()
def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        # num_nodes = data.batch.size(0)
        # data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)


    return x,y,result


def set_seed(seed):
    print(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="IMDB-BINARY")
    parser.add_argument('--model', type=str, default="GIN")
    parser.add_argument('--epoch', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--saliency', type=str, default="False")
    parser.add_argument('--lam_range', type=str, default="[0.005, 0.01]")
    parser.add_argument('--lambda_value', type=float, default=0.5)
    parser.add_argument('--saliency_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--log_screen', type=str, default="False")
    parser.add_argument('--criterion', type=str, default="SupConMixLoss")
    parser.add_argument('--temp', type=float, default="0.07") 
    parser.add_argument('--warmup', type=int, default="3") 
    parser.add_argument('--augsup', type=str, default="False")
    parser.add_argument('--num_layers', type=int, default="3") 
    


    args = parser.parse_args()

    
    data_path = args.data_path
    dataset_name = args.dataset
    seed = args.seed
    lam_range = eval(args.lam_range)
    log_screen = eval(args.log_screen)
    saliency = eval(args.saliency)
    augsup = eval(args.augsup)
    num_epochs = args.epoch

    num_hidden = args.num_hidden
    batch_size = args.batch_size
    learning_rate = args.lr
    lambda_value = args.lambda_value
    saliency_ratio = args.saliency_ratio
    model = args.model
    criterion = args.criterion
    temp = args.temp
    warmup = args.warmup
    num_layers = args.num_layers

    warnings.filterwarnings("ignore")

    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)


    logger.info('parser.prog: {}'.format(parser.prog))
    logger.info("args:{}".format(args))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"runing device: {device}")

    

    if dataset_name == "ogbg-molhiv":
        dataset = PygGraphPropPredDataset(name=dataset_name)
        split_idx = dataset.get_idx_split()
    else:
        path = osp.join(data_path, dataset_name)
        dataset = TUDataset(path, name=dataset_name)


    dataset = list(dataset)
    dataset = random.sample(dataset, len(dataset) // 2)

    for graph in dataset:
        graph.y = graph.y.view(-1)


    seeds = [1314,41314,51314]
    avg_test_mi_f1, avg_test_acc, avg_test_ma_f1 = [], [], []
    for seed in seeds:
        set_seed(seed)
        random.shuffle( dataset )

        train_nums = int(len(dataset) * 0.8)
        train_val_nums = int(len(dataset) * 0.9)
        
        avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(dataset[: train_nums])
        logger.info(f"avg num nodes of training graphs: { avg_num_nodes }")
        logger.info(f"avg num edges of training graphs: { avg_num_edges }")
        logger.info(f"avg density of training graphs: { avg_density }")
        logger.info(f"median num nodes of training graphs: { median_num_nodes }")
        logger.info(f"median num edges of training graphs: { median_num_edges }")
        logger.info(f"median density of training graphs: { median_density }")



        aug1 = A.Identity()
        aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                            A.NodeDropping(pn=0.1),
                            A.FeatureMasking(pf=0.1),
                            A.EdgeRemoving(pe=0.1)], 1)
        
        
        dataset = prepare_dataset_x( dataset )
        num_features = dataset[0].x.shape[1]
        num_classes = dataset[0].y.shape[0]

        if dataset_name == "ogbg-molhiv":
            train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            train_dataset = [dataset[i] for i in train_idx.tolist()]
            val_dataset = [dataset[i] for i in valid_idx.tolist()]
            test_dataset = [dataset[i] for i in test_idx.tolist()]

        else:
            train_dataset = dataset[:train_nums]
            random.shuffle(train_dataset)
            val_dataset = dataset[train_nums:train_val_nums]
            train_val_dataset = dataset[:train_val_nums]
            test_dataset = dataset[train_val_nums:]
        


        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        if model == "GIN":
            gconv = GIN(input_dim=num_features, hidden_dim=32, num_layers=3).to(device)
        elif model == "GCN":
            gconv = GCN(input_dim=num_features, hidden_dim=32, num_layers=2).to(device)
        else:
            logger.info(f"No model."  )


        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
        if criterion == "SupConMixLoss_G":
            criterion = SupConMixLoss_G(temperature=temp).to(device)
        elif criterion == "SupConMixLoss_N":
            criterion = SupConMixLoss_N(temperature=temp).to(device)

        optimizer = torch.optim.Adam(encoder_model.parameters(), lr=learning_rate)

        present_acc = best_val_acc = best_val_mi_f1 = best_val_ma_f1 = 0


        for epoch in range(1, num_epochs):

            model, train_loss = train(encoder_model, criterion, train_loader, optimizer,saliency)
            train_acc = 0
            _,_,val_result = test(encoder_model, dataloader)
            x,y,test_result = test(encoder_model, dataloader)
            val_acc = val_result["accuracy"]
            test_acc = test_result["accuracy"]
            test_mi_f1 = test_result["micro_f1"]
            test_ma_f1 = test_result["macro_f1"]


            logger.info('Epoch: {:03d}, Train Loss: {:.6f}, Val acc: {:.6f}, micro_f1: {:.6f}, macro_f1: {:.6f}, accuracy: {:.6f}'.format(
                epoch, train_loss, val_result["accuracy"],test_result["micro_f1"], test_result["macro_f1"], test_result["accuracy"]))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                present_acc = test_acc
                logger.info('Epoch: {:03d},  val acc: {:.6f}, test acc: {:.6f}'.format(
                epoch,  best_val_acc, present_acc))
                best_test_mi_f1 = test_mi_f1 
                best_test_ma_f1 = test_ma_f1

                
        avg_test_acc.append(present_acc)
        avg_test_mi_f1.append(best_test_mi_f1)
        avg_test_ma_f1.append(best_test_ma_f1)

    
    acc_CI =  statistics.stdev(avg_test_acc)
    mi_f1_CI =  statistics.stdev(avg_test_mi_f1)
    ma_f1_CI =  statistics.stdev(avg_test_ma_f1)

    
    avg_acc = statistics.mean(avg_test_acc)
    avg_mi_f1 = statistics.mean(avg_test_mi_f1)
    avg_ma_f1 = statistics.mean(avg_test_ma_f1)



    avg_log = 'Test Acc: {:.4f} +- {:.4f}, micro_f1: {:.4f} +- {:.4f}, macro_F1: {:.4f} +- {:.4f}'
    avg_log = avg_log.format(avg_acc ,acc_CI ,avg_mi_f1, mi_f1_CI, avg_ma_f1, ma_f1_CI)
    print(avg_log)

    

            

