import torch
import torch.nn as nn
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from glob import glob

# -------------------------
# Configuração
# -------------------------
num_classes = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ângulos adaptados: -60° a +60°, espaçados uniformemente
angles_deg = np.linspace(-60, 60, num_classes, endpoint=False)
sector_angles = np.deg2rad(angles_deg)

# Procurar arquivo .mat automaticamente
mat_files = sorted(glob(os.path.join('datasets', 'dataset_coarse_SNR*.mat')))
assert mat_files, "Nenhum arquivo .mat encontrado em 'datasets/'"
mat_path = mat_files[0]
print(f"Usando arquivo: {mat_path}")

# -------------------------
# Arquitetura do modelo
# -------------------------
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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# -------------------------
# Carregar modelo
# -------------------------
model = CoarseCNN(num_classes).to(device)
model.load_state_dict(torch.load('coarseDOA_net.pth', map_location=device))
model.eval()

# -------------------------
# Carregar amostras do .mat
# -------------------------
with h5py.File(mat_path, 'r') as f:
    T = np.array(f['Tcoarse']).transpose(3, 2, 1, 0)
    Y = np.array(f['Ylabel']).T

X = T[..., :6].transpose(3, 2, 0, 1)
X = torch.tensor(X, dtype=torch.float32).to(device)

# -------------------------
# Predição
# -------------------------
with torch.no_grad():
    Ypred = model(X)

Ybin = (Ypred >= 0.5).int().cpu().numpy()

# -------------------------
# Mostrar resultados
# -------------------------
print("Setores ativados por amostra:")
for i, bin_vec in enumerate(Ybin):
    setores = np.where(bin_vec == 1)[0]
    print(f"Amostra {i}: setores = {setores.tolist()}")

# -------------------------
# Visualização polar adaptada (–60° a +60°)
# -------------------------
fig, axes = plt.subplots(2, 3, subplot_kw={'projection': 'polar'}, figsize=(12, 6))
axes = axes.flatten()

for i, bin_vec in enumerate(Ybin):
    ax = axes[i]
    theta = sector_angles
    r = bin_vec.tolist() + [bin_vec[0]]         # fecha o ciclo
    t = list(theta) + [theta[0]]                # idem

    ax.plot(t, r, drawstyle='steps-post', linewidth=2)
    ax.fill(t, r, alpha=0.4)
    ax.set_title(f"Amostra {i}", va='bottom')
    ax.set_rticks([0, 1])
    ax.set_yticklabels([])
    ax.set_thetamin(-90)   # limitar ângulo polar visualizado
    ax.set_thetamax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(True)

plt.suptitle("Setores Ativados (–60° a +60°) – Predição coarseDOA", fontsize=14)
plt.tight_layout()
plt.show()
