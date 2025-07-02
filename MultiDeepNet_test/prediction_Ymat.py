import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matlab.engine # type: ignore

# --------------------------
# 1. Executar script do MATLAB
# --------------------------
# Definir parâmetros
snapshots = 10
AoA       = matlab.double([55, 23.22])
dist      = matlab.double([9, 9])
snr_dB    = 15
K         = 2

print("[INFO] Executando script MATLAB para gerar Y...")
eng = matlab.engine.start_matlab()
eng.addpath(r'MultiDeepNet_Datasets', nargout=0)
eng.gera_Y(float(snapshots), AoA, dist, float(snr_dB), float(K), nargout=0)
eng.quit()
print("[INFO] Arquivo entrada_Y.mat gerado com sucesso.")

# --------------------------
# 2. Carregar Y do MATLAB
# --------------------------
print("[INFO] Carregando matriz Y...")
data = scipy.io.loadmat('entrada_Y.mat')
Y = data['Y']  # [M × snapshots]

# --------------------------
# 3. Construir R e tensor T
# --------------------------
snapshots = Y.shape[1]
R = Y @ Y.conj().T / snapshots  # [M×M] matriz de covariância

T = np.zeros((R.shape[0], R.shape[1], 3), dtype=np.float32)
T[..., 0] = np.real(R)
T[..., 1] = np.imag(R)
T[..., 2] = np.angle(R)

Xtest = T.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, M, M]
Xtest = torch.tensor(Xtest, dtype=torch.float32)

# --------------------------
# 4. Arquitetura da rede
# --------------------------
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
            nn.Linear(32 * 10 * 10, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 12),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# --------------------------
# 5. Carregar modelo treinado
# --------------------------
print("[INFO] Carregando modelo coarseDOA_net.pth...")
model = CoarseCNN(12)

model_path = r'MultiDeepNet/coarseDOA_net.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
#model.load_state_dict(torch.load('coarseDOA_net.pth', map_location='cpu'))
model.eval()

# --------------------------
# 6. Inferência
# --------------------------
with torch.no_grad():
    probs = model(Xtest).squeeze().numpy()

# --------------------------
# 7. Resultado numérico
# --------------------------
setores = np.arange(-55, 56, 10)
print("\n[INFO] Probabilidades por setor:")
for i, p in enumerate(probs):
    print(f"Setor {setores[i]:+d}°: {p:.3f}")

detected = setores[probs > 0.5]
print(f"\n[INFO] Setores detectados (prob > 0.5): {detected.tolist()}")

# --------------------------
# 8. Plot gráfico de barras
# --------------------------
plt.figure(figsize=(10, 4))
plt.bar(setores, probs, width=9, color='royalblue', edgecolor='k')
plt.axhline(0.5, color='r', linestyle='--', label='Threshold 0.5')
plt.xticks(setores)
plt.xlabel('Ângulo (graus)')
plt.ylabel('Probabilidade')
plt.title('Saída coarseDOA por setor (–60° a 60°)')
plt.grid(True, axis='y', linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('predicao_coarseDOA.png', dpi=150)
plt.show()
