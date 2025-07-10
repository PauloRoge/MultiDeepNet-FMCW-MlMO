# coarse_predict.py
import numpy as np, torch
from torch import nn

# --- CNN -----------------------------------------------------------------
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
# -------------------------------------------------------------------------

# carrega a rede uma única vez (lazy-load)
_model = None
def _load():
    global _model
    if _model is None:
        _model = CoarseCNN()
        _model.load_state_dict(torch.load('MultiDeepNet/coarseDOA_net.pth',
                                          map_location='cpu'))
        _model.eval()
    return _model

# ========= API visível pelo MATLAB ======================================
def predict_batch(tnp):
    """
    Recebe tensor numpy em uma destas formas
        (10,10,3,N)  ou  (N,10,10,3)
    Retorna vetor numpy int64 de tamanho N com as classes 0-11.
    """
    arr = np.asarray(tnp)           # converte PyProxy → numpy
    if arr.ndim != 4:
        raise ValueError(f'Esperado dim=4, recebi {arr.shape}')
    # normaliza para (N,3,10,10)
    if arr.shape[-1] == 3:
        arr = arr.transpose(3,2,0,1)
    elif arr.shape[1] == 3:
        pass
    else:
        raise ValueError(f'Formato inesperado {arr.shape}')
    x = torch.from_numpy(arr.astype('float32'))
    with torch.no_grad():
        y = _load()(x).argmax(1).cpu().numpy().astype('int64')
    return y
