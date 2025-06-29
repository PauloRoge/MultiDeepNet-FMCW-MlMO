%% generate_coarse_dataset_multi_parfor_fixed.m
clear; clc; rng(0);

%% Parâmetros conforme artigo Multi-DeepNet
M           = 10;                   % Número de antenas receptoras (ULA)
frequency   = 78.737692e9;          % Frequência central f_c = 78.737692 GHz
lambda      = 3e8 / frequency;      % Comprimento de onda (λ = c/f)
delta       = lambda / 2;           % Espaçamento entre elementos: λ/2
snapshots   = 10;                   % N = número de snapshots

SNRdB_list  = -20:2:20;             % Intervalo de SNRs usado no artigo

nSamples    = 5e5;                  % Número de amostras por SNR (artigo usa 500 mil)
maxSources  = 6;                    % K ~ Uniforme de 1 a 6 alvos

%% Configuração coarse-DOA
edgesCoarse = linspace(-60, 60, 13); % 12 bins de 10° para estimação coarse
nCoarse     = numel(edgesCoarse)-1; % nCoarse = 12


% Inicializa pool paralelo
if isempty(gcp('nocreate')), parpool; end

for iSNR = 1:numel(SNRdB_list)
    snr_dB = SNRdB_list(iSNR);

    % Pré-aloca tudo
    Tcoarse = zeros(M, M, 3, nSamples, 'single');
    Ylabel  = false(nCoarse, nSamples);

    parfor idx = 1:nSamples
        % 1) sorteia K alvos
        K    = randi([1 maxSources]);
        AoA  = rand(K,1)*120 - 60;
        dist = rand(K,1)*9 + 1;

        % 2) gera sinais
        [Y,~] = signals(M, snapshots, delta, lambda, AoA, K, dist, snr_dB);

        % 3) covariância e tensor 3-canal em variável local
        Rs   = (Y * Y') / snapshots;
        chan = single( cat(3, real(Rs), imag(Rs), angle(Rs)) );

        % 4) atribuição única por slice
        Tcoarse(:,:,:,idx) = chan;

        % 5) multilabel one-hot
        bins = floor((AoA + 60)/10) + 1;
        bins = min(max(bins,1),nCoarse);
        lbl  = false(nCoarse,1);
        lbl(unique(bins)) = true;
        Ylabel(:,idx) = lbl;
    end

    % Salva em disco
    fname = sprintf('dataset_coarse_SNR%+03d.mat', snr_dB);
    save(fname, 'Tcoarse', 'Ylabel', '-v7.3');
    fprintf('Salvo %s (%d amostras)\n', fname, nSamples);
end
