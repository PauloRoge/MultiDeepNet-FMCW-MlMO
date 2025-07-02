clear; clc;
addpath(genpath('./'));  % ajuste o caminho se necessário

%% PARÂMETROS FIXOS
frequency = 78.737692e9;
lambda    = 3e8 / frequency;
delta     = lambda/2;
AoA       = 48;      % ângulo real (graus)
K         = 1;       % número de alvos
d         = 8;       % distância relativa (m)
SNRdB     = 10;      % SNR (dB)
M         = 10;      % número de antenas
theta     = -60:1:60;
snap_vec  = 10:50:1000;  % varredura de snapshots
MCS       = 200;         % repetições Monte Carlo

RMSE_music = zeros(1,numel(snap_vec));
RMSE_net   = zeros(1,numel(snap_vec));

%% CARREGA A REDE E CRIA O CONSTANTE PARA O PARFOR
netData  = load('coarseDOA_net10dB.mat','bestNet');
netConst = parallel.pool.Constant(netData.bestNet);

%% LOOP PRINCIPAL
parfor idx = 1:numel(snap_vec)
    snapshots = snap_vec(idx);
    err_music = 0;
    err_net   = 0;

    % cada worker acessa netConst.Value sem re-serializar a cada iteração
    bestNet = netConst.Value;

    for mc = 1:MCS
        % Gera sinal
        X = signals(M, snapshots, delta, lambda, AoA, K, d, SNRdB);

        % --- MUSIC clássico ---
        [Pmusic, ~] = music(X, M, theta, snapshots, delta, lambda, K);
        [~, ip]     = max(Pmusic);
        theta_est   = theta(ip);
        err_music   = err_music + (theta_est - AoA)^2;

        % --- Rede coarseDOA ---
        R = (X*X')/snapshots;
        T_local = zeros(M,M,3,'single');
        T_local(:,:,1) = real(R);
        T_local(:,:,2) = imag(R);
        T_local(:,:,3) = angle(R);
        dlX = dlarray(reshape(T_local,[M M 3 1]), 'SSCB');

        YPred = predict(bestNet, dlX);
        probs = extractdata(YPred);
        [~, im] = max(probs);
        theta_net = -60 + 10*(im-1) + 5;  % centro do setor
        err_net   = err_net + (theta_net - AoA)^2;
    end

    RMSE_music(idx) = sqrt(err_music / MCS);
    RMSE_net(idx)   = sqrt(err_net   / MCS);
end

%% PLOT
figure;
semilogy(snap_vec, RMSE_music, 'b-',  'LineWidth',1.5); hold on;
semilogy(snap_vec, RMSE_net,   'r--','LineWidth',1.5);
xlabel('Número de Snapshots');
ylabel('RMSE (graus)');
legend('MUSIC clássico','Rede coarseDOA','Location','northeast');
grid on;
title(sprintf('RMSE vs Snapshots | SNR = %d dB | AoA = %d°', SNRdB, AoA));


%% FUNÇÃO MUSIC
function [Pmusic, eigenvalues_norm] = music(Y, M, theta, snapshots, delta, lambda, users)
    R = (Y*Y')/snapshots;
    [V,D] = eig(R);
    vals  = diag(D);
    [~, ord] = sort(vals,'descend');
    Vn = V(:, ord(users+1:end));

    Pmusic = zeros(size(theta));
    for k = 1:numel(theta)
        a = responsearray(M, delta, lambda, theta(k));
        Pmusic(k) = 1/(a'*(Vn*Vn')*a);
    end
    Pmusic = abs(Pmusic)/max(abs(Pmusic));
    eigenvalues_norm = vals(ord)/sum(vals);
end

%% FUNÇÃO responsearray
function a = responsearray(M, delta, lambda, theta)
    gamma = 2*pi*delta/lambda;
    a = exp(-1j*gamma*(0:M-1).' * sind(theta));
end
