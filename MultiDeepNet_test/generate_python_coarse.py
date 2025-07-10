import numpy as np
import torch
from joblib import Parallel, delayed
from python_signals import signals
from tqdm import tqdm
import os

# ----------------------------
# 1. Parâmetros
# ----------------------------
c         = 3e8
fc        = 78.737692e9
lambda_   = c / fc
delta     = lambda_ / 2

M         = 10
snapshots = 10
SNRdB_list = np.arange(-10, 17, 2)  # de -10 até 15
nSamples   = 100_000
maxSources = 3

edgesCoarse = np.linspace(-60, 60, 13)  # 12 setores
nCoarse     = len(edgesCoarse) - 1      # 12

save_dir = "datasets_pt"
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# 2. Função para uma única amostra
# ----------------------------
def generate_sample(snr_dB):
    K    = np.random.randint(1, maxSources + 1)
    AoA  = np.random.rand(K) * 120 - 60
    dist = np.random.rand(K) * 9 + 1

    Y, _ = signals(M, snapshots, delta, lambda_, AoA, K, dist, snr_dB)
    Rs   = Y @ Y.conj().T / snapshots

    T = np.zeros((M, M, 3), dtype=np.float32)
    T[..., 0] = np.real(Rs)
    T[..., 1] = np.imag(Rs)
    T[..., 2] = np.angle(Rs)

    bins = np.clip(np.floor((AoA + 60) / 10).astype(int), 0, nCoarse - 1)
    label = np.zeros(nCoarse, dtype=np.float32)
    label[np.unique(bins)] = 1.0

    return T, label

# ----------------------------
# 3. Loop principal por SNR
# ----------------------------
for snr_dB in SNRdB_list:
    print(f"[INFO] Gerando dados para SNR = {snr_dB} dB ...")

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(generate_sample)(snr_dB) for _ in tqdm(range(nSamples))
    )

    Tcoarse = torch.tensor(np.stack([r[0] for r in results], axis=0))  # [nSamples, M, M, 3]
    Ylabel  = torch.tensor(np.stack([r[1] for r in results], axis=0))  # [nSamples, 12]

    # Salvar como dicionário PyTorch
    filename = os.path.join(save_dir, f"coarse_SNR{snr_dB:+03d}.pt")
    torch.save({'Tcoarse': Tcoarse, 'Ylabel': Ylabel}, filename)
    print(f"[OK] Salvo: {filename} ({nSamples} amostras)")
