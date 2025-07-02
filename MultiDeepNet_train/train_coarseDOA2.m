% =======================================================================
%  train_coarseDOA_singleUser.m
%  CNN LEVE (12 classes de 10°) – otimizada p/ notebook Ryzen-7-5700U (8 GB RAM)
% -----------------------------------------------------------------------
%  • Entrada  : tensores 10×10×3 (Re, Im, Ângulo da matriz de covariância)
%  • Saída    : vetor 1×12 de probabilidades – setor com argmax é o DOA
%  • Treino   : Binary Cross-Entropy (BCE) + Adam + Early-Stopping
%  • Versões  : – MATLAB R2020b ↑  → usa binaryCrossEntropyLayer (multi-rótulo)
%               – MATLAB < R2020b → cai automaticamente para Softmax + classificationLayer
% =======================================================================
clear; clc; rng default; tic

%% ----------------------------------------------------------------------
% 1) Carregar e empilhar datasets (garanta que existam *.mat em /datasets)
% -----------------------------------------------------------------------
dataDir = fullfile(pwd,'datasets');
files   = dir(fullfile(dataDir,'dataset_coarse_SNR*.mat'));
assert(~isempty(files),'Nenhum dataset encontrado em "%s".',dataDir);

T_all = [];  Y_all = [];
for k = 1:numel(files)
    D  = load(fullfile(files(k).folder,files(k).name),'Tcoarse','Ylabel');
    %  Tcoarse → 10×10×3×N   |   Ylabel → 12×N (one-hot)
    T_all = cat(4,T_all, D.Tcoarse);
    Y_all = cat(2,Y_all, D.Ylabel);
end
T_all = single(T_all);   Y_all = single(Y_all);

%% ----------------------------------------------------------------------
% 2) Divisão treino/validação (90 % / 10 %)
% ----------------------------------------------------------------------
N      = size(Y_all,2);
perm   = randperm(N);
Ntr    = floor(0.9*N);
trIdx  = perm(1:Ntr);
vaIdx  = perm(Ntr+1:end);

XTrain = T_all(:,:,:,trIdx);   YTrain = Y_all(:,trIdx).';   % (Ntr × 12)
XVal   = T_all(:,:,:,vaIdx);   YVal   = Y_all(:,vaIdx).';   % (Nval × 12)
clear T_all Y_all

%% ----------------------------------------------------------------------
% 3) Definição da CNN leve
% ----------------------------------------------------------------------
layersCore = [
    imageInputLayer([10 10 3],'Name','input','Normalization','none')

    convolution2dLayer(3,64,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    leakyReluLayer(0.3,'Name','lrelu1')

    convolution2dLayer(3,32,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    leakyReluLayer(0.3,'Name','lrelu2')

    convolution2dLayer(3,16,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    leakyReluLayer(0.3,'Name','lrelu3')

    fullyConnectedLayer(256,'Name','fc1')
    reluLayer('Name','relu1')
    dropoutLayer(0.5,'Name','drop1')

    fullyConnectedLayer(64,'Name','fc2')
    reluLayer('Name','relu2')
    dropoutLayer(0.5,'Name','drop2')

    fullyConnectedLayer(12,'Name','fc_out')
];

% -----------------------------------------------------------------------
% 4) Escolha automática da camada de perda
% -----------------------------------------------------------------------
useBCE = exist('binaryCrossEntropyLayer','file') == 2;  % disponível R2020b+

if useBCE
    % ---- Versão multi-rótulo com BCE ----
    layers = [
        layersCore
        sigmoidLayer('Name','sigmoid')
        binaryCrossEntropyLayer('Name','bce')  % fecha o grafo
    ];
    YTrainTable = YTrain;   % continua como matriz one-hot
    YValTable   = YVal;
else
    % ---- Fallback: Softmax + classificationLayer (single-label) ----
    warning('binaryCrossEntropyLayer não encontrado – usando Softmax + classificationLayer.');
    layers = [
        layersCore
        softmaxLayer('Name','softmax')
        classificationLayer('Name','out')      % fecha o grafo
    ];
    % Converte one-hot → categorical [1-12]
    [~,YtrIdx] = max(YTrain,[],2);
    [~,YvaIdx] = max(YVal,  [],2);
    YTrainTable = categorical(YtrIdx);
    YValTable   = categorical(YvaIdx);
end

%% ----------------------------------------------------------------------
% 5) Opções de treinamento
% ----------------------------------------------------------------------
opts = trainingOptions('adam', ...
    'InitialLearnRate',      1e-3, ...
    'L2Regularization',      1e-4, ...
    'MaxEpochs',             100, ...
    'MiniBatchSize',         64, ...
    'Shuffle',               'every-epoch', ...
    'ValidationData',        {XVal, YValTable}, ...
    'ValidationFrequency',   100, ...
    'ValidationPatience',    5, ...
    'OutputNetwork',         'best-validation', ...
    'ExecutionEnvironment',  'cpu', ...
    'Verbose',               false, ...
    'Plots',                 'training-progress');

%% ----------------------------------------------------------------------
% 6) Treinamento
% ----------------------------------------------------------------------
fprintf('Treinando (%d treino | %d validação)...\n',size(XTrain,4),size(XVal,4));
[net,info] = trainNetwork(XTrain, YTrainTable, layers, opts);

%% ----------------------------------------------------------------------
% 7) Avaliação simples na validação
% ----------------------------------------------------------------------
if useBCE
    YPred  = predict(net,XVal,'MiniBatchSize',64);
    [~,id] = max(YPred,[],2);
    [~,gt] = max(YVal, [],2);
else
    YPred  = classify(net,XVal,'MiniBatchSize',64);
    id     = double(YPred);      % 1-12
    gt     = double(YValTable);  % 1-12
end
accVal = mean(id==gt);
fprintf('Acurácia top-1 (validação) = %.2f %%\n',100*accVal);

%% ----------------------------------------------------------------------
% 8) Salvar a melhor rede
% ----------------------------------------------------------------------
bestValLoss = min(info.ValidationLoss);
save coarseDOA_net_singleUser.mat net bestValLoss
fprintf('Rede salva em coarseDOA_net_singleUser.mat  (ValLoss = %.4f)\n',bestValLoss);

toc
