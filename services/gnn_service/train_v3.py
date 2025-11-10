"""
GNN Training V3: full pipeline (BIM synth, warmup, advanced scheduler)
- Поддержка синтетики с time/degradation/cascade
- Warmup + ReduceLROnPlateau
- Улучшенные early stopping и метрики
"""
import json
import logging
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
import random
from config_v3 import model_config_v3, training_config_v3
from model_v3 import create_enhanced_model_v3

# Датасет
class GNNTabularDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, faults, features, time_col='time'):
        df = pd.read_csv(csv_path)
        if time_col not in df.columns:
            df[time_col] = np.arange(len(df))
        self.X = df[features].values.astype(np.float32)
        self.y = df[faults].values.astype(np.float32)
        self.time = df[time_col].values.astype(np.int64)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.X[idx]),
            'y': torch.tensor(self.y[idx]),
            'time': self.time[idx]
        }

# Подготовка данных
def prepare_loaders():
    # Маппинг фичей и меток
    with open('services/gnn_service/data/equipment_metadata.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    feats = sum(meta['column_mapping'].values(), [])
    faults = list(meta['fault_columns'].values())
    df = pd.read_csv(training_config_v3.data_path)
    N = len(df)
    idx = np.arange(N)
    random.shuffle(idx)
    n_train = int(N*training_config_v3.train_split)
    n_val = int(N*training_config_v3.val_split)
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    ds = GNNTabularDataset(training_config_v3.data_path, faults, feats)
    train = torch.utils.data.Subset(ds, train_idx)
    val = torch.utils.data.Subset(ds, val_idx)
    test = torch.utils.data.Subset(ds, test_idx)
    return (
        DataLoader(train, batch_size=training_config_v3.batch_size, shuffle=True),
        DataLoader(val, batch_size=training_config_v3.batch_size, shuffle=False),
        DataLoader(test, batch_size=training_config_v3.batch_size, shuffle=False)
    )

# Warmup scheduler
def get_scheduler(optimizer):
    def lr_lambda(epoch):
        if epoch < training_config_v3.warmup_epochs:
            return (epoch+1)/training_config_v3.warmup_epochs
        return 1.0
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor=training_config_v3.lr_scheduler_factor,
        patience=training_config_v3.lr_scheduler_patience,
        min_lr=training_config_v3.lr_scheduler_min_lr
    )
    return warmup, plateau

# Label smoothing
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing):
        super().__init__()
        self.smooth = smoothing
    def forward(self, preds, targets):
        targets_sm = targets*(1-self.smooth) + self.smooth*0.5
        return nn.functional.binary_cross_entropy_with_logits(preds, targets_sm)

# Тренировочный цикл
def train():
    train_loader, val_loader, test_loader = prepare_loaders()
    model = create_enhanced_model_v3(training_config_v3.device)
    criterion = LabelSmoothingBCE(training_config_v3.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=training_config_v3.learning_rate, weight_decay=training_config_v3.weight_decay)
    warmup, plateau = get_scheduler(optimizer)
    history = {'val_f1': [], 'train_f1': [], 'train_loss': [], 'val_loss': [], 'lr': []}
    best_f1, best_epoch = 0, 0
    patience = training_config_v3.patience
    stop_counter = 0
    model.train()
    for epoch in range(training_config_v3.epochs):
        total, lsum, y_true, y_pred = 0, 0, [], []
        for batch in train_loader:
            optimizer.zero_grad()
            X, y = batch['x'].to(training_config_v3.device), batch['y'].to(training_config_v3.device)
            out = model(X)
            loss = criterion(out, y)
            if training_config_v3.gradient_clip:
                nn.utils.clip_grad_norm_(model.parameters(), training_config_v3.gradient_clip)
            loss.backward()
            optimizer.step()
            lsum += loss.item() * X.size(0)
            y_true.append(y.cpu().numpy())
            y_pred.append((torch.sigmoid(out) > 0.5).cpu().numpy())
            total += X.size(0)
        # Warmup (плавный старт)
        if epoch < training_config_v3.warmup_epochs:
            warmup.step()
        else:
            val_f1, val_loss = evaluate(model, val_loader, criterion)
            plateau.step(val_f1)
        train_loss = lsum / total
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        train_f1 = f1_score(y_true, y_pred, average='macro')
        val_f1, val_loss = evaluate(model, val_loader, criterion)
        lr = optimizer.param_groups[0]['lr']
        history['lr'].append(lr)
        history['val_f1'].append(val_f1)
        history['train_f1'].append(train_f1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} train_F1={train_f1:.3f} val_F1={val_f1:.3f} lr={lr:.2e}")
        # Early stopping
        if val_f1 > best_f1 + 1e-3:
            best_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), training_config_v3.model_save_path)
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter >= patience:
            break
    print(f"[✓] Training finished — Best val_F1={best_f1:.3f} at epoch {best_epoch+1}")
    json.dump(history, open(training_config_v3.history_save_path, 'w'), indent=2)
    # Тест
    model.load_state_dict(torch.load(training_config_v3.model_save_path))
    test_f1, test_loss = evaluate(model, test_loader, criterion, verbose=True)
    print(f"[TEST] macro F1={test_f1:.3f}")

def evaluate(model, loader, criterion, verbose=False):
    model.eval()
    lsum, total = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            X, y = batch['x'].to(training_config_v3.device), batch['y'].to(training_config_v3.device)
            out = model(X)
            loss = criterion(out, y)
            lsum += loss.item() * X.size(0)
            y_true.append(y.cpu().numpy())
            y_pred.append((torch.sigmoid(out) > 0.5).cpu().numpy())
            total += X.size(0)
    avg_loss = lsum / total
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    if verbose:
        print('[Evaluation] loss=%.4f, F1=%.3f' % (avg_loss, f1))
    model.train()
    return f1, avg_loss

if __name__ == '__main__':
    train()