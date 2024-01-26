import torch
import tqdm
import datetime
import pytz
import os
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataset import *
from utils import *
from fair_models.dgp import DGP
from torch.optim.lr_scheduler import ExponentialLR, StepLR


def train(model, optimizer_bb, optimizer_env, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    torch.autograd.set_detect_anomaly(True)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        output = model(fields)
        if len(output)==5:
            y, reg, y_g, y_main, y_id  = output
            model.zero_grad()
            loss = criterion(y, target.float())
            loss += criterion(y_g[0],target.float())+criterion(y_main,target.float())-0.001*criterion(y_g[1],target.float())
            loss += reg
            loss.backward()
            optimizer_bb.step()
        else:
            print("Output length error!!!")
            print(len(output))
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
  


def test(model, data_loader, device):
    model.eval()
    targets, predicts, loss = list(), list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            loss.extend(torch.nn.functional.binary_cross_entropy(y, target.float(), reduction='none').tolist())
    return roc_auc_score(targets, predicts), np.array(loss).mean()


def main(dataset_name,
         dataset_path,
         fair_name,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):
    device = torch.device(device)
    train_dataset = get_dataset(dataset_name, dataset_path, 'train')
    valid_dataset = get_dataset(dataset_name,dataset_path, 'valid')
    test_dataset = get_dataset(dataset_name, dataset_path,'test')
    gender0 = get_dataset(dataset_name, dataset_path, 'gender0')
    gender1 = get_dataset(dataset_name, dataset_path, 'gender1')
    field_dims = np.maximum(train_dataset.field_dims, valid_dataset.field_dims, test_dataset.field_dims)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    gender0_loader = DataLoader(gender0, batch_size=batch_size, num_workers=8)
    gender1_loader = DataLoader(gender1, batch_size=batch_size, num_workers=8)
    model = get_fair(fair_name, model_name, field_dims).to(device)
    criterion = torch.nn.BCELoss()
    optimizer_bb = torch.optim.Adam([params for name,params in model.named_parameters() if 'env' not in name], lr=learning_rate, weight_decay=weight_decay)
    optimizer_env=None
    timezone = pytz.timezone('Etc/GMT-0')
    current_time = datetime.datetime.now(timezone).strftime("%Y%m%d_%H%M%S_%f")[:20]
    print(current_time)
    directory = os.path.join(save_dir, f"{dataset_name}_{fair_name}_{model_name}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    chkpt_file = os.path.join(directory, f"{current_time}.pt")
    if os.path.exists(chkpt_file):
        print("WARNING: REPEATED CHECKPOINT FILE NAME!!!")
    early_stopper = EarlyStopper(num_trials=4, save_path=chkpt_file)
    scheduler_bb = StepLR(optimizer_bb,step_size=8,gamma=0.8)
    for epoch_i in range(epoch):
        train(model, optimizer_bb, optimizer_env,train_data_loader, criterion, device)
        scheduler_bb.step()
        if optimizer_env:
            scheduler_env.step()
        auc, loss = test(model, valid_data_loader, device)        
        print('epoch:', epoch_i, 'validation: auc:', auc, 'logloss:',loss)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    model = torch.load(chkpt_file)
    auc, loss = test(model, test_data_loader, device)
    print(f'test auc: {auc}, test logloss: {loss}')
    auc0, _ = test(model, gender0_loader, device)
    auc1, _ = test(model, gender1_loader, device)
    print(f'Auc Gender 0: {auc0}, Gender1: {auc1}')
    print(f'auc difference:{auc1-auc0}')
    return auc, loss, auc1, auc0, auc1-auc0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ml1m')
    parser.add_argument('--model_name', default='mlp')
    parser.add_argument('--fair_name', default='dgp')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.dataset_path ='./ml1m/'
    args.batch_size = 4096   
    args.learning_rate=0.002
    args.rnum=10
    user_idx = [3,4,5,6,7]
    def get_fair(fair_name, model_name, field_dims):
        if fair_name == 'dgp':
            print(user_idx)
            return DGP(model_name,field_dims,user_idx)
        else:
            raise ValueError('unknown model name: ' + fair_name)
    set_random_seed(42)
    res = []
    for i in range(args.rnum):
        r  = main(args.dataset_name,
            args.dataset_path,
            args.fair_name,
            args.model_name,
            args.epoch,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.device,
            '/chkpt')
        res.append(r)
    print(res)
    print(f'Averaged Results for {args.model_name}, {args.fair_name}: {np.round(np.mean(res,0),5)}')
    print(f'Results std for {args.model_name}, {args.fair_name}: {np.round(np.std(res,0),5)}')
    print(f'Results ptp of {args.model_name}: {np.round(np.ptp(res,0),5)}')
