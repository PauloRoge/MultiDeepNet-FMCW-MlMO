clear; clc; rng('default');

%% 1. Parâmetros da simulação
M           = 10;                   % Número de antenas receptoras (ULA)
frequency   = 78.737692e9;          % Frequência central f_c = 78.737692 GHz
lambda      = 3e8 / frequency;      % Comprimento de onda (λ = c/f)
delta       = lambda / 2;           % Espaçamento entre elementos: λ/2
snapshots   = 1;                   % N = número de snapshots
% AoA = 10;
% dist = 5;
AoA = 25.32;                       % angulos de chegada dos alvos (graus)
dist = 10;                       % distancia relativa (m)
snr_dB = 20;
K = 1;

%% 2. Geração do sinal
X = signals(M, snapshots, delta, lambda, AoA, K, dist, snr_dB);  % [M×N]

%% 3. Construção do tensor de entrada
R = (X * X') / snapshots;

T = zeros(M, M, 3, 'single');
T(:,:,1) = real(R);
T(:,:,2) = imag(R);
T(:,:,3) = angle(R);

Xtest = reshape(T, [M M 3 1]);        % [10×10×3×1]
dlX   = dlarray(Xtest, 'SSCB');       % formato esperado pela rede

%% 4. Carregar e inferir com a rede treinada
load('coarseDOA_net10dB.mat', 'bestNet');
YPred = predict(bestNet, dlX);
probs = extractdata(YPred);  % vetor [1×12]

%% 5. Visualização com blocos retangulares e legendas inclinadas [θ₁, θ₂]
centros = -55:10:55;
intervalos_str = arrayfun(@(c) sprintf('[%d,%d]', c-5, c+5), centros, 'UniformOutput', false);

figure('Units','normalized','Position',[0.25 0.4 0.5 0.2]);
hold on;

% Normaliza os valores para o colormap
cmap = parula(256);
probs_norm = (probs - min(probs)) / (max(probs) - min(probs));
probs_idx = max(1, round(probs_norm * 255) + 1);

% Desenha cada bloco e adiciona legenda
for i = 1:12
    x = [i-1 i i i-1];
    y = [0 0 1 1];
    fill(x, y, cmap(probs_idx(i),:), 'EdgeColor', 'k');
    
    % Legenda inclinada sob o bloco
    text(i - 0.5, -0.15, intervalos_str{i}, ...
        'Rotation', 45, ...
        'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 8);
end

% Ajustes visuais
xlim([0 12]);
ylim([-0.3 1]);                    % espaço para as legendas
axis off;

%title(sprintf('Predição coarse DOA - %d usuário(s)  |  SNR = %d dB', K, snr_dB));
colorbar('Ticks', [0 0.5 1], 'TickLabels', {'0', '0.5', '1'});
colormap(parula);

%% 6. Impressão dos resultados
fprintf('\n[INFO] Ângulos verdadeiros:\n');
disp(AoA);

fprintf('[INFO] Vetor de probabilidades predito:\n');
disp(probs);

fprintf('[INFO] Classes detectadas (prob > 0.5):\n');
disp(centros(probs > 0.5));
