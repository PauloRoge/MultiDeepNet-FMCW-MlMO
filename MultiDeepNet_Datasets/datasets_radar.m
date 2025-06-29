% generate_coarse_dataset_multi.m
clear; clc; rng(1234);

%% Parâmetros de geração
M           = 10;               % número de antenas
frequency   = 3e9;              % 3 GHz
lambda      = 3e8/frequency;    % comprimento de onda
delta       = lambda/2;         % espaçamento inter-antenas
snapshots   = 10;               % número de snapshots
SNRdB_list  = -20:2:20;         % níveis de SNR para treinamento
nSamples    = 5e4;              % amostras por nível de SNR
maxSources  = 6;                % máximo de alvos simultâneos

%% Configuração coarse-DOA
edgesCoarse = linspace(-60,60,13);   % define 12 intervalos de 10°
nCoarse     = numel(edgesCoarse)-1;  % =12

for iSNR = 1:numel(SNRdB_list)
    snr_dB = SNRdB_list(iSNR);
    
    % pré-aloca (type single para economizar memória)
    Tcoarse = zeros(M, M, 3, nSamples, 'single');
    Ylabel  = false(nCoarse, nSamples);
    
    for idx = 1:nSamples
        % 1) sorteia número de alvos, ângulos e distâncias
        K     = randi([1 maxSources]);
        AoA   = (rand(K,1)*120 - 60);    % em graus, unif em [−60,60]
        dist  = rand(K,1)*9 + 1;         % em [1,10] m
        
        % 2) gera sinais com SNR desejada
        [Y, ~] = signals(M, snapshots, delta, ...
            lambda, AoA, K, dist, snr_dB);
        
        % 3) monta tensor 3-canal a partir da matriz de covariância
        Rs = (Y * Y') / snapshots;
        Tcoarse(:,:,1,idx) = real(Rs);
        Tcoarse(:,:,2,idx) = imag(Rs);
        Tcoarse(:,:,3,idx) = angle(Rs);
        
        % 4) multilabel one-hot para os K alvos nos 12 intervalos
        label = false(nCoarse,1);
        for k = 1:K
            bin = find(AoA(k) >= edgesCoarse, 1, 'last');
            if bin>=1 && bin<=nCoarse
                label(bin) = true;
            end
        end
        Ylabel(:,idx) = label;
    end
    
    % 5) salva em .mat (usa -v7.3 para grandes volumes)
    fname = sprintf('dataset_coarse_SNR%+03d.mat', snr_dB);
    save(fname, 'Tcoarse', 'Ylabel', '-v7.3');
    fprintf('Salvo %s (nSamples=%d)\n', fname, nSamples);
end
