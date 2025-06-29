%% Script completo para pseudoespectro MUSIC com linhas verticais em cada AoA real

close all; clear; clc

% Parâmetros do sistema
architecture = 10;            % número de antenas no array (pode ser vetor, ex: [8 10 12])
frequency   = 78.737692e9;    % 78 GHz
lambda      = 3e8 / frequency; % comprimento de onda
delta       = lambda/2;        % espaçamento entre antenas (½λ)
snapshots   = 10;              % número de instantâneos
SNRdB       = 20;              % relação sinal-ruído em dB
AoA         = [-56 48 -10 12]; % ângulos de chegada (graus)
d           = [10 4 6 2];      % distâncias relativas (m)
source      = length(AoA);     % número de fontes
theta       = -60:1:60;        % grid de ângulos para pseudoespectro

% Cria figura
figure;
hold on;

% Cores para cada curva
colors = {[0.3,0.3,0.3], 'b', 'r'};

% Loop sobre arquiteturas
for ii = 1:length(architecture)
    M = architecture(ii);
    
    % Gera sinais recebidos (função externa)
    Y = signals(M, snapshots, delta, lambda, AoA, source, d, SNRdB);
    
    % Chama MUSIC (retorna pseudoespectro e autovalores normalizados)
    [Pmusic, eigenvalues] = music(Y, M, theta, snapshots, delta, lambda, source);
    
    % Plota pseudoespectro em dB
    plot(theta, 10*log10(Pmusic), ...
         'Color', colors{mod(ii-1, length(colors))+1}, ...
         'DisplayName', ['M = ' num2str(M)]);
end

% --- Traça linhas verticais tracejadas cinza em cada AoA real ---
if exist('xline','builtin')
    % MATLAB R2018b ou superior
    for k = 1:length(AoA)
        xline(AoA(k), '--', ...
              'Color', [0.5 0.5 0.5], ...
              'LineWidth', 1, ...
              'HandleVisibility', 'off');
    end
else
    % Compatibilidade com versões anteriores
    yl = ylim;
    for k = 1:length(AoA)
        line([AoA(k) AoA(k)], yl, ...
             'LineStyle', '--', ...
             'Color', [0.5 0.5 0.5], ...
             'LineWidth', 1, ...
             'HandleVisibility', 'off');
    end
end
% -------------------------------------------------------------------

% Adiciona informação de AoA e distância na legenda
legend_entry = sprintf('AoA: %s (°) \nDistance: %s (m)', strjoin(string(AoA), ', '), strjoin(string(d), ', '));
plot(nan, nan, 'Color', 'none', 'DisplayName', legend_entry);

% Ajustes finais de figura
xlabel('Angle (degrees)');
ylabel('Pseudo Spectrum (dB)');
grid on;
legend show;
hold off;


%% Função MUSIC
function [musicpseudospectrum, eigenvalues] = music(Y, M, theta, snapshots, delta, lambda, users)
    % Estima matriz de covariância
    R = (Y * Y') / snapshots;

    % Decomposição em autovalores/vetores
    [V, D] = eig(R);
    eigvals = diag(D);
    [eigvals_sorted, idx] = sort(eigvals, 'descend');
    eigvals_norm = eigvals_sorted / sum(eigvals_sorted);
    
    % Espaço de ruído
    V = V(:, idx);
    Vn = V(:, users+1:end);

    % Calcula pseudoespectro MUSIC
    Pmusic = zeros(size(theta));
    for ii = 1:length(theta)
        a = responsearray(M, delta, lambda, theta(ii));
        Pmusic(ii) = 1 / (a' * (Vn * Vn') * a);
    end

    % Normaliza e retorna
    Pmusic = abs(Pmusic) / max(abs(Pmusic));
    musicpseudospectrum = Pmusic;
    eigenvalues = eigvals_norm;
end

%% Função de vetor de resposta (steering vector)
function a = responsearray(M, delta, lambda, theta)
    gamma = 2*pi * delta / lambda;
    a = exp(-1j * gamma * (0:M-1)' * sind(theta));
end
