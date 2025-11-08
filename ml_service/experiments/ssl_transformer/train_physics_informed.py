"""Train Component Models with Physics-Informed Thresholds"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from pathlib import Path

from models.component_models import CylinderModel, PumpModel
from config.equipment_schema import EquipmentConfig, create_cat336_config


class PhysicsInformedLoss(nn.Module):
    def __init__(self, equipment_config, component_type):
        super().__init__()
        self.config = equipment_config
        self.component_type = component_type
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets, sensor_data):
        ml_loss = self.bce_loss(predictions, targets.float())
        physics_penalty = self._calculate_physics_penalty(sensor_data, predictions)
        return ml_loss + 0.1 * physics_penalty
    
    def _calculate_physics_penalty(self, sensor_data, predictions):
        penalty = 0.0
        if self.component_type == "cylinder":
            pressure_diff = sensor_data[:, 4] if sensor_data.shape[1] > 4 else sensor_data[:, 0]
            ml_fault_prob = torch.sigmoid(predictions).squeeze()
            disagreement = (pressure_diff > 40) * (1 - ml_fault_prob)
            penalty = disagreement.mean()
        return penalty


def normalize_features(features, equipment_config, component_type):
    normalized = features.copy()
    
    if component_type == "cylinder":
        cyl = equipment_config.boom_cylinder
        max_pressure = cyl.max_pressure
        
        if "pressure_extend" in normalized.columns:
            normalized["pressure_extend"] /= max_pressure
        if "pressure_retract" in normalized.columns:
            normalized["pressure_retract"] /= max_pressure
        if "position" in normalized.columns:
            normalized["position"] = normalized["position"] / (cyl.stroke / 1000)
        if "velocity" in normalized.columns:
            normalized["velocity"] = normalized["velocity"] / cyl.velocity_max
        if "pressure_diff" in normalized.columns:
            normalized["pressure_diff"] = normalized["pressure_diff"] / (2 * max_pressure)
    
    elif component_type == "pump":
        pump = equipment_config.pump
        
        if "pressure_outlet" in normalized.columns:
            normalized["pressure_outlet"] /= pump.max_pressure
        if "speed_rpm" in normalized.columns:
            normalized["speed_rpm"] /= pump.nominal_rpm
        if "temperature" in normalized.columns:
            normalized["temperature"] /= pump.temp_threshold.max_value
    
    return normalized


def train_component_model(
    model_class,
    data_path,
    model_name,
    feature_cols,
    equipment_config,
    component_type,
    epochs=50,
    batch_size=64,
    lr=0.0001
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} (Physics-Informed)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Equipment: {equipment_config.equipment_id}")
    
    df = pd.read_csv(data_path)
    
    X = normalize_features(df[feature_cols], equipment_config, component_type)
    X = X.values.astype("float32")
    y = df["fault_any"].values.astype("int64")
    X_raw = df[feature_cols].values.astype("float32")
    
    X_train, X_temp, y_train, y_temp, X_raw_train, X_raw_temp = train_test_split(
        X, y, X_raw, train_size=0.7, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test, X_raw_val, X_raw_test = train_test_split(
        X_temp, y_temp, X_raw_temp, train_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Fault rate: {y_train.mean()*100:.2f}%")
    
    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(X_raw_train)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val),
        torch.tensor(X_raw_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = model_class(hidden_dim=64, num_layers=2, dropout=0.1).to(device)
    criterion = PhysicsInformedLoss(equipment_config, component_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float("inf")
    best_model_state = None
    
    print(f"\nStarting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch, X_raw_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            X_raw_batch = X_raw_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).view(-1)
            loss = criterion(outputs, y_batch, X_raw_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch, X_raw_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                X_raw_batch = X_raw_batch.to(device)
                
                outputs = model(X_batch).view(-1)
                loss = criterion(outputs, y_batch, X_raw_batch)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Val Acc: {val_acc*100:.2f}% | "
                  f"Time: {elapsed:.1f}s")
    
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_preds, test_labels = [], []
    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test),
        torch.tensor(X_raw_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for X_batch, y_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).view(-1)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(y_batch.numpy())
    
    acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="binary"
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTS - {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "equipment_config": equipment_config.to_dict(),
        "component_type": component_type,
        "feature_cols": feature_cols,
        "metrics": {"accuracy": acc, "f1": f1}
    }, save_dir / f"{model_name.lower()}_physics.pt")
    
    print(f"\n✅ Model saved!")
    
    return model, acc, f1


if __name__ == "__main__":
    print("🚀 Physics-Informed Training Pipeline")
    print(f"Device: {'CUDA ✅' if torch.cuda.is_available() else 'CPU'}")
    
    config = create_cat336_config()
    
    # Train Cylinder (5 features)
    cyl_model, cyl_acc, cyl_f1 = train_component_model(
        CylinderModel,
        "data/component_cylinder_clean.csv",
        "CylinderModel",
        ["pressure_extend", "pressure_retract", "position", "velocity", "pressure_diff"],
        config,
        "cylinder",
        epochs=50,
        batch_size=128
    )
    
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Cylinder - Acc: {cyl_acc*100:.2f}%, F1: {cyl_f1*100:.2f}%")
