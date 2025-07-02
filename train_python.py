import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

# ----------------------- Configurações -----------------------
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
num_epochs = 100
val_freq = 200
patience = 5
learning_rate = 1e-3
threshold = 0.5
data_dir = './datasets'
input_size = (10, 10, 3)
num_classes = 12

# --------------------- Carregar os dados ---------------------
T_all, Y_all = [], []

for file in sorted(glob(os.path.join(data_dir, 'dataset_coarse_SNR*.mat'))):
    data = scipy.io.loadmat(file)
    T_all.append(data['Tcoarse'])  # (10,10,3,N)
    Y_all.append(data['Ylabel'])   # (12,N)

T_all = np.concatenate(T_all, axis=3).astype(np.float32)
Y_all = np.concatenate(Y_all, axis=1).astype(np.float32)

# Transpor para (N, 3, 10, 10)
X = torch.tensor(T_all).permute(3, 2, 0, 1)
Y = torch.tensor(Y_all).T

# Shuffle e split
N = X.shape[0]
idx = torch.randperm(N)
N_train = int(0.9 * N)

X_train, Y_train = X[idx[:N_train]], Y[idx[:N_train]]
X_val, Y_val     = X[idx[N_train:]], Y[idx[N_train:]]

print(f"Total={N}  (treino={N_train}  validação={N - N_train})")

# --------------------- Modelo CNN ---------------------
class DOACNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),

            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 2048),
            nn.ReLU(),

            nn.Linear(2048, 512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = DOACNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --------------------- Avaliação ---------------------
def evaluate_metrics(y_true, y_pred, threshold=0.5):
    y_bin = (y_pred >= threshold).float()
    y_true = y_true > 0.5

    TP = (y_bin * y_true).sum(dim=0)
    FP = (y_bin * (1 - y_true)).sum(dim=0)
    FN = ((1 - y_bin) * y_true).sum(dim=0)
    TN = ((1 - y_bin) * (1 - y_true)).sum(dim=0)

    prec = (TP / (TP + FP + 1e-8)).mean().item()
    rec  = (TP / (TP + FN + 1e-8)).mean().item()
    f1   = (2 * prec * rec) / (prec + rec + 1e-8)
    acc  = ((TP + TN) / (TP + TN + FP + FN + 1e-8)).mean().item()

    return acc, prec, rec, f1

def val_loss(model, X, Y):
    model.eval()
    with torch.no_grad():
        losses, preds = [], []
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].to(device)
            yb = Y[i:i+batch_size].to(device)
            yp = model(xb)
            loss = criterion(yp, yb)
            losses.append(loss.item())
            preds.append(yp.cpu())

        Y_pred = torch.cat(preds, dim=0)
        L = np.mean(losses)
        acc, prec, rec, f1 = evaluate_metrics(Y.cpu(), Y_pred, threshold)
        return L, Y_pred, acc, prec, rec, f1

# --------------------- Treinamento ---------------------
best_val_loss = float('inf')
best_model = None
pat_count = 0
train_losses, val_losses, iters = [], [], []

total_it = 0
for epoch in range(num_epochs):
    model.train()
    perm = torch.randperm(N_train)
    for i in range(0, N_train, batch_size):
        total_it += 1
        idx = perm[i:i+batch_size]
        xb = X_train[idx].to(device)
        yb = Y_train[idx].to(device)

        optimizer.zero_grad()
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        loss.backward()
        optimizer.step()

        if total_it % val_freq == 0:
            val_loss_val, Ypred_val, acc, prec, rec, f1 = val_loss(model, X_val, Y_val)
            print(f"Ep {epoch+1:2d} | It {total_it:5d} | Train {loss.item():.4f} | Val {val_loss_val:.4f} | Acc {acc:.3f} | Prec {prec:.3f} | Rec {rec:.3f} | F1 {f1:.3f}")
            
            train_losses.append(loss.item())
            val_losses.append(val_loss_val)
            iters.append(total_it)

            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                best_model = model.state_dict()
                pat_count = 0
            else:
                pat_count += 1
                if pat_count >= patience:
                    print(f"Early stopping na época {epoch+1}")
                    break
    if pat_count >= patience:
        break

# --------------------- Salvar modelo ---------------------
torch.save(best_model, 'coarseDOA_net.pt')
print(f'Modelo salvo (ValLoss={best_val_loss:.4f})')

# --------------------- Plotar perda ---------------------
plt.figure()
plt.plot(iters, train_losses, '-b', label='train')
plt.plot(iters, val_losses, '-r', label='val')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.title('Loss × iteration')
plt.grid(True)
plt.show()
