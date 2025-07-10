import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from python_signals import signals  # Certifique-se que python_signals.py está no mesmo diretório

# --------------------------
# 1. Gerar sinal Y com função Python
# --------------------------
print("[INFO] Gerando sinal Y...")

M         = 10
lambda_   = 3.8e-3
delta     = 0.5 * lambda_
AoA       = [55, 23.22]             # ângulos de chegada em graus
dist      = np.array([9, 9])        # distâncias convertidas para array NumPy
K         = 2
snapshots = 10
SNRdB     = 15

Y, Z = signals(M, snapshots, delta, lambda_, AoA, K, dist, SNRdB)
print("[INFO] Matriz Y gerada com sucesso.")

# --------------------------
# 2. Construir matriz de covariância e tensor T
# --------------------------
R = Y @ Y.conj().T / snapshots

T = np.zeros((M, M, 3), dtype=np.float32)
T[..., 0] = np.real(R)
T[..., 1] = np.imag(R)
T[..., 2] = np.angle(R)

Xtest = T.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, M, M]
Xtest = torch.tensor(Xtest, dtype=torch.float32)

# --------------------------
# 3. Definir arquitetura da rede coarseDOA
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
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# --------------------------
# 4. Carregar pesos da rede treinada
# --------------------------
print("[INFO] Carregando modelo coarseDOA_net.pth...")
model = CoarseCNN(num_classes=12)
model.load_state_dict(torch.load('MultiDeepNet/coarseDOA_net.pth', map_location='cpu'))
model.eval()

# --------------------------
# 5. Inferência
# --------------------------
with torch.no_grad():
    probs = model(Xtest).squeeze().numpy()

# --------------------------
# 6. Resultado numérico
# --------------------------
setores = np.arange(-55, 56, 10)
print("\n[INFO] Probabilidades por setor:")
for i, p in enumerate(probs):
    print(f"Setor {setores[i]:+d}°: {p:.3f}")

detected = setores[probs > 0.5]
print(f"\n[INFO] Setores detectados (prob > 0.5): {detected.tolist()}")

# --------------------------
# 7. Plotagem dos resultados
# --------------------------
label_user = ', '.join([f'{ang:.2f}° ({d:.1f} m)' for ang, d in zip(AoA, dist)])

plt.figure(figsize=(10, 3))
plt.bar(setores, probs, width=9, color='royalblue', edgecolor='k', label='Saída da rede')
plt.axhline(0.5, color='r', linestyle='--', label='Threshold 0.5')

for ang in AoA:
    plt.axvline(x=ang, color='g', linestyle='-.', linewidth=1.5)

plt.legend(title=f'Usuários: {label_user}')
plt.xticks(setores)
plt.xlabel('Ângulo (graus)')
plt.ylabel('Probabilidade')
plt.title('Saída coarseDOA por setor (–60° a 60°)')
plt.grid(True, axis='y', linestyle=':')
plt.tight_layout()
plt.savefig('predicao_coarseDOA.png', dpi=150)
plt.show()
