% train_coarseDOA_stage1.m
% CNN “grossa” de 12 classes (setores de 10°) – primeiro estágio do Multi-DeepNet
% -------------------------------------------------------------------------
clear; clc; rng default;                           % inicialização reprodutível
tic;                                               % mede tempo de execução

%% 1) Carregar e empilhar todos os datasets em 'datasets/'
dataDir   = fullfile(pwd,'datasets');
files     = dir(fullfile(dataDir,'dataset_coarse_SNR*.mat'));

assert(~isempty(files), ...
    "Nenhum dataset encontrado em 'datasets/' – verifique caminho.");

T_all = []; Y_all = [];                            % tensores acumuladores
for k = 1:numel(files)
    D        = load(fullfile(files(k).folder, files(k).name), ...
                    'Tcoarse','Ylabel');
    T_all    = cat(4, T_all, D.Tcoarse);           % (10×10×3×N)  – entrada
    Y_all    = cat(2, Y_all, D.Ylabel);            % (12×N)       – rótulo
end
T_all = single(T_all);   Y_all = single(Y_all);

%% 2) Dividir treino / validação (90 % / 10 %)
%      ─ NÃO ESPECIFICADO NO ARTIGO (o texto só fala em treino)
Ntot   = size(Y_all, 2);
perm   = randperm(Ntot);
Ntrain = floor(0.9 * Ntot);                       % 90 % para treino

trainIdx = perm(1:Ntrain);
valIdx   = perm(Ntrain+1:end);

XTrain = T_all(:,:,:,trainIdx);
XVal   = T_all(:,:,:,valIdx);

% Converter rótulos one-hot para classe inteira (índice)
[~, yClassTrain] = max(Y_all(:,trainIdx), [], 1);
[~, yClassVal]   = max(Y_all(:,valIdx),   [], 1);

YTrain = categorical(yClassTrain);
YVal   = categorical(yClassVal);

clear T_all Y_all;                                   % libera RAM

fprintf("Total=%d  (treino=%d  validação=%d)\n", ...
        Ntot, Ntrain, numel(valIdx));

%% 3) Arquitetura da CNN (idêntica à Tabela 1 do artigo)
inputSize  = size(XTrain,1:3);      % [10 10 3]
numClasses = 12;                    % 12 setores de 10°

layers = [
    imageInputLayer(inputSize, 'Normalization','none')

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer(0.3)

    fullyConnectedLayer(256)
    dropoutLayer(0.3)
    reluLayer

    fullyConnectedLayer(numClasses)
    softmaxLayer                                 % ← modificado
    classificationLayer                          % ← modificado
];

%% 4) Hiper-parâmetros de treino
%      β1, β2 e taxa de aprendizagem são iguais aos do artigo.
%      As demais escolhas são locais do script.
learnRate     = 1e-4;                 % η
miniBatchSize = 128;                  %% ─ NÃO ESPECIFICADO NO ARTIGO
numEpochs     = 100;                  % conforme artigo
valFreq       = 200;                  %% ─ NÃO ESPECIFICADO NO ARTIGO
patience      = 10;                   %% ─ NÃO ESPECIFICADO NO ARTIGO

options = trainingOptions('adam', ...
    'InitialLearnRate', learnRate, ...
    'MaxEpochs', numEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal,YVal}, ...
    'ValidationFrequency', valFreq, ...
    'ValidationPatience', patience, ...
    'Verbose', true, ...
    'Plots','training-progress');

%% 5) Treinamento da rede
net = trainNetwork(XTrain, YTrain, layers, options);

%% 6) Salvar a melhor rede encontrada
save coarseDOA_net.mat net                     %% ─ NÃO ESPECIFICADO
fprintf('Modelo salvo com softmax/crossentropy.\n');
