""" 
A regression analysis program that loads data from Excel 
and predicts values using PyTorch deep learning with MPS/CUDA support.
"""
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Device Configuration
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device.type.upper()}")

# Argument parser configuration
parser = argparse.ArgumentParser(description='PyTorch Regression from Excel Data')
parser.add_argument('--file', type=str, default='sample_data.xlsx', help='Path to the excel file')
parser.add_argument('--sheet', type=str, default='Sheet1', help='Sheet name')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch', type=int, default=32, help='Batch size')
parser.add_argument('--save_model', type=str, default='nn_model.pth', help='Path to save the trained model')
args = parser.parse_args()

# Load and preprocessing
try:
    with pd.ExcelFile(args.file) as xls:
        sheet_name = args.sheet if args.sheet in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet_name).dropna()
        print(f"Loaded: {args.file} [{sheet_name}] | Shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)
input_dim = X.shape[1]

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_x, scaler_y = StandardScaler(), StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# To Tensors
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train_scaled).to(device), torch.tensor(y_train_scaled).to(device)),
    batch_size=args.batch, shuffle=True
)
X_test_t = torch.tensor(X_test_scaled).to(device)
y_test_t = torch.tensor(y_test_scaled).to(device)

# Model Definition
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), # Increased capacity
            nn.BatchNorm1d(128),       # Stable training
            nn.ReLU(),
            nn.Dropout(0.1),           # Prevent overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

model = RegressionModel(input_dim=input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

history = {'train_loss': [], 'val_loss': []}

# Training Loop
print("Starting training...")
for epoch in range(args.epochs):
    model.train()
    train_loss_accum = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / len(train_loader)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_t)
        v_loss = criterion(val_outputs, y_test_t)
        scheduler.step(v_loss)
        val_loss_val = v_loss.item()

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss_val)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_val:.4f}')

# Save the trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,
    'scaler_x': scaler_x,
    'scaler_y': scaler_y
}, args.save_model)
print(f"\nModel and scalers saved to: {args.save_model}")

# Final Evaluation
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t).cpu().numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    r2 = r2_score(y_test, y_pred)

print(f"\nFinal Evaluation:\nR2 Score: {r2:.4f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Learning Curve
ax1.plot(history['train_loss'], label='Train Loss', color='blue', lw=1.5)
ax1.plot(history['val_loss'], label='Validation Loss', color='orange', lw=1.5)
ax1.set_title('Model Loss Progression', fontsize=12)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MSE Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Actual vs Predicted
ax2.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', color='teal')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_title(f'NN Regression: Actual vs Predicted\n(R² = {r2:.4f})', fontsize=12)
ax2.set_xlabel('Actual Values')
ax2.set_ylabel('Predicted Values')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
