import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Plot da curva de perda
import matplotlib.pyplot as plt
import numpy as np

# Configurações
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(os.getcwd(), 'datasets')
files = sorted(glob(os.path.join(data_dir, 'dataset_coarse_SNR*.mat')))
assert files, "Nenhum dataset encontrado."

# Parâmetros
num_classes = 12
batch_size = 128
num_epochs = 100
val_freq = 200
patience = 8
lr = 5e-4
weight_decay = 1e-4

# Carregar e empilhar datasets
X_all, Y_all = [], []
for file in files:
    with h5py.File(file, 'r') as f:
        T = np.array(f['Tcoarse']).transpose(3, 2, 1, 0)  # [N, 3, 10, 10] → [10, 10, 3, N]
        Y = np.array(f['Ylabel']).T  # [12, N]
        X_all.append(T)
        Y_all.append(Y)
X_all = np.concatenate(X_all, axis=3)  # → [10, 10, 3, total]
Y_all = np.concatenate(Y_all, axis=1)  # → [12, total]

# Separar treino/validação
X = X_all.transpose(3, 2, 0, 1)  # → [N, 3, 10, 10]
Y = Y_all.T                     # → [N, 12]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
Y_val   = torch.tensor(Y_val, dtype=torch.float32)

print(f"Total={len(X)}  Treino={len(X_train)}  Validação={len(X_val)}")

# Dataset/DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
val_dataset   = torch.utils.data.TensorDataset(X_val, Y_val)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

# CNN definida
class CoarseCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.2),

            nn.Flatten(),
            nn.Linear(32*10*10, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes),
            nn.Sigmoid()  # multiclasse independente
        )

    def forward(self, x):
        return self.model(x)

model = CoarseCNN(num_classes).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Loop de treino
best_val_loss = float('inf')
no_improve = 0
global_step = 0

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        if global_step % val_freq == 0:
            model.eval()
            val_loss = 0
            preds_all, targets_all = [], []
            with torch.no_grad():
                for xvb, yvb in val_loader:
                    xvb, yvb = xvb.to(device), yvb.to(device)
                    pred_v = model(xvb)
                    loss_v = criterion(pred_v, yvb)
                    val_loss += loss_v.item()
                    preds_all.append(pred_v.cpu().numpy())
                    targets_all.append(yvb.cpu().numpy())

            val_loss /= len(val_loader)
            preds_all = np.concatenate(preds_all)
            targets_all = np.concatenate(targets_all)

            pred_bin = preds_all >= 0.5
            acc  = np.mean((pred_bin == targets_all).astype(float))
            prec = precision_score(targets_all, pred_bin, average='macro', zero_division=0)
            rec  = recall_score(targets_all, pred_bin, average='macro', zero_division=0)
            f1   = f1_score(targets_all, pred_bin, average='macro', zero_division=0)

            print(f"Ep {epoch+1} | Step {global_step} | Train {loss.item():.4f} | Val {val_loss:.4f} | Acc {acc:.3f} | Prec {prec:.3f} | Rec {rec:.3f} | F1 {f1:.3f}")

            train_losses.append(loss.item())
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(model.state_dict(), "coarseDOA_net.pth")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping na época {epoch+1}.")
                    break
    if no_improve >= patience:
        break

print(f"Modelo salvo (ValLoss={best_val_loss:.4f})")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=1.8)
plt.plot(val_losses, label='Validation Loss', linewidth=1.8)
plt.xlabel('Validação periódica (x200 iterações)')
plt.ylabel('Loss (BCE)')
plt.title('Curva de Perda - coarseDOA')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()

