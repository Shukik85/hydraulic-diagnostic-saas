# services/gnn_service/train_dynamic.py
"""
Training loop для UniversalDynamicGNN.
"""
import torch
import torch.optim as optim
from tqdm import tqdm
from schemas import EquipmentMetadata
from model_dynamic_gnn import create_model
from data_loader_dynamic import create_dynamic_dataloaders

csv_path = "data/bim_comprehensive.csv"
metadata_path = "data/equipment_metadata.json"
model_save_path = "models/universal_dynamic_best.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# You can parameterize below
EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# Load metadata
import json
with open(metadata_path) as f:
    metadata = EquipmentMetadata(**json.load(f))

# Create dataloaders
train_loader, val_loader, test_loader = create_dynamic_dataloaders(
    csv_path, metadata_path, batch_size=BATCH_SIZE, sequence_length=5)

# Create model
model = create_model(metadata, device=device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        feats = {k:v.to(device) for k,v in batch["component_features"].items()}
        # TODO: get y_true (health) from batch or create supervised label
        # y_true = batch["target"]
        y_true = torch.zeros(BATCH_SIZE, len(feats), device=device)  # dummy, replace
        optimizer.zero_grad()
        health, degradation, _ = model(feats)
        loss = loss_fn(health, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss/len(train_loader):.4f}")
# Save model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
