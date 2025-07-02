clear; clc; rng('default');

%% 1. Parâmetros da simulação
M         = 10;                  % Número de antenas receptoras (ULA)
frequency = 78.737692e9;         % Frequência central f_c = 78.737692 GHz
lambda    = 3e8 / frequency;     % Comprimento de onda (λ = c/f)
delta     = lambda / 2;          % Espaçamento entre elementos: λ/2
snapshots = 10;                  % Número de snapshots
AoA       = [-15 45];                  % Ângulo real de chegada (graus)
dist      = [9 9];                   % Distância real do usuário (mesma unidade de r)
snr_dB    = 15;                  % SNR em dB
K         = 2;                   % Número de fontes

%% 2. Geração do sinal
X = signals(M, snapshots, delta, lambda, AoA, K, dist, snr_dB);
Y=X;
save('entrada_Y.mat', 'Y');