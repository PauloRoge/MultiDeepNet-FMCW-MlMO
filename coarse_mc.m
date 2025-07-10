% coarseDOA_MC.m  –  versão pyenv (Nível-3)
rng(0,'twister');

% ------- parâmetros de cenário ----------
M=10; snapshots=10;
fc=78.737692e9; lambda=3e8/fc; delta=lambda/2;
SNRdB=-10:5:15; N_MC=10; nUsersMax=3;
batchSize = 200;                     % tamanho do lote para Python
Tbuf  = zeros(10,10,3,batchSize,'single');
label = zeros(batchSize,1,'uint8');

% ------- ativa pyenv uma única vez -------
pyExe='C:\Users\JR\AppData\Local\Programs\Python\Python311\python.exe';
if isempty(pyenv)
    pyenv('Version',pyExe, 'ExecutionMode','OutOfProcess');  % mantém isolado
end
pyCP = py.importlib.import_module('coarse_predict');         % módulo Python

% ------- pool paralelo opcional ----------
if isempty(gcp('nocreate')); parpool('local',2); end         % poucos workers

accuracySNR = zeros(size(SNRdB));

for si = 1:numel(SNRdB)
    snr = SNRdB(si);
    acc = 0; tot = 0;

    parfor (k = 1:N_MC, 2)          % no máx. 2 workers → 2 instâncias PyTorch
        % ------ gera amostra -------------------------
        nUsers = randi([1,nUsersMax]);
        AoA    = -60 + 120*rand(1,nUsers);
        d      = 10*rand(1,nUsers);
        [Y,~]  = signals(M,snapshots,delta,lambda,AoA,nUsers,d,snr);
        Tlocal = single(extract_tensor(Y));   % 10×10×3
        gt     = uint8(min(floor((AoA(1)+60)/10),11));

        % ------ cada worker usa buffer próprio -------
        [accPart,totPart] = workerPredict(Tlocal,gt,batchSize,pyCP);
        acc = acc + accPart;
        tot = tot + totPart;
    end

    accuracySNR(si) = acc/tot;
    fprintf('SNR %+3d dB  |  Acc = %.4f\n', snr, accuracySNR(si));
end

figure; plot(SNRdB,accuracySNR,'-o','LineWidth',1.8);
xlabel('SNR (dB)'), ylabel('Acurácia'), grid on
title('CoarseDOAnet – Acurácia vs. SNR');

% ================= worker helper ====================
function [hits,total] = workerPredict(T,gt,batchSize,pyCP)
persistent buf lab idx
if isempty(idx); idx=0; buf=[]; lab=[]; end

idx = idx+1;
if idx==1
    buf = zeros(10,10,3,batchSize,'single');
    lab = zeros(batchSize,1,'uint8');
end
buf(:,:,:,idx) = T;
lab(idx)       = gt;

hits  = 0; total = 0;
if idx==batchSize
    preds = int32(pyCP.predict_batch(buf));
    hits  = sum(preds.'==double(lab));
    total = batchSize;
    idx   = 0;                      % esvazia
end
end

% extract_tensor.m
function T = extract_tensor(Y)
    % Calcula a matriz de covariância e extrai os canais: real, imag, angle
    R = (Y * Y') / size(Y,2);   % M x M
    T(:,:,1) = real(R);         % canal 1
    T(:,:,2) = imag(R);         % canal 2
    T(:,:,3) = angle(R);        % canal 3
end
