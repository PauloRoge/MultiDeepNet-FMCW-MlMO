import numpy as np

def responsearray(M, delta, lambda_, theta):
    gamma = 2 * np.pi * delta / lambda_
    a = np.exp(-1j * gamma * np.arange(M).reshape(-1, 1) * np.sin(np.radians(theta)))
    return a

def signals(M, snapshots, delta, lambda_, AoA, numSources, d, SNRdB):
    # Converte listas para arrays NumPy, se necessário
    AoA = np.asarray(AoA, dtype=np.float64)
    d   = np.asarray(d, dtype=np.float64)

    PL = (lambda_ / (4 * np.pi))**2 / (d**2)
    beta = np.sqrt(PL)

    H = np.zeros((M, numSources), dtype=complex)
    for s in range(numSources):
        a = responsearray(M, delta, lambda_, AoA[s])
        H[:, s] = beta[s] * a[:, 0]

    X = (np.random.randn(numSources, snapshots) + 1j * np.random.randn(numSources, snapshots)) / np.sqrt(2)
    Y_sig = H @ X
    P_signal = np.mean(np.abs(Y_sig)**2)
    noiseVar = P_signal / (10**(SNRdB / 10))
    Z = np.sqrt(noiseVar) * (np.random.randn(M, snapshots) + 1j * np.random.randn(M, snapshots)) / np.sqrt(2)

    Y = Y_sig + Z
    return Y, Z
