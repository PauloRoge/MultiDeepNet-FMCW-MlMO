# run_coarse_predict.py  (substitua o antigo)
import sys, h5py, torch, numpy as np
from torch import nn

# ---------------- CNN ----------------
class CoarseCNN(nn.Module):
    def __init__(self, C=12):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,128,3,padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.3), nn.Dropout(0.2),
            nn.Conv2d(128,64,3,padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.3), nn.Dropout(0.2),
            nn.Conv2d(64,32,3,padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.3), nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(32*10*10,1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,C), nn.Sigmoid()
        )
    def forward(self,x): return self.model(x)
# -------------------------------------

if __name__ == '__main__':
    matfile = sys.argv[1]
    with h5py.File(matfile, 'r') as f:
        T = f['T'][:]                 # (3,10,10)  ou  (10,10,3)

    # --- normaliza shape para (1,3,10,10) ---
    if T.shape[0] == 3:               # canais já na 1ª dimensão
        x = torch.from_numpy(T).unsqueeze(0)          # (1,3,10,10)
    elif T.shape[-1] == 3:            # canais na última
        x = torch.from_numpy(T).permute(2,0,1).unsqueeze(0)
    else:
        raise ValueError(f'Formato inesperado: {T.shape}')

    x = x.float()                     # garante float32

    # --- inferência ---
    dev   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoarseCNN().to(dev)
    model.load_state_dict(torch.load('MultiDeepNet/coarseDOA_net.pth',
                                     map_location=dev))
    model.eval()
    with torch.no_grad():
        y = model(x.to(dev)).cpu().numpy()

    print(int(np.argmax(y, axis=1)[0]))   # devolve 0–11
