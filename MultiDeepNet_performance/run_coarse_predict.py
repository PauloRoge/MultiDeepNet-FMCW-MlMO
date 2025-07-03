# run_coarse_predict.py
import h5py, torch, numpy as np
from torch import nn

class CoarseCNN(nn.Module):
    # ... mesma definição anterior ...

def predict(mat_path: str) -> int:
    with h5py.File(mat_path,'r') as f:
        T = f['T'][:].astype('float32')
    x = torch.from_numpy(T).permute(2,0,1).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = CoarseCNN().to(device)
    model.load_state_dict(torch.load('coarseDOA_net.pth', map_location=device))
    model.eval()
    with torch.no_grad():
        y = model(x.to(device)).cpu().numpy()
    return int(np.argmax(y,axis=1)[0])

if __name__=='__main__':
    import sys
    print(predict(sys.argv[1]))
