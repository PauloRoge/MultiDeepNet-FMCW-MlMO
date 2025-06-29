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

XTrain = T_all(:,:,:,trainIdx);  YTrain = Y_all(:,trainIdx);
XVal   = T_all(:,:,:,valIdx);    YVal   = Y_all(:,valIdx);
clear T_all Y_all;                                   % libera RAM

fprintf("Total=%d  (treino=%d  validação=%d)\n", ...
        Ntot, Ntrain, numel(valIdx));

%% 3) Arquitetura da CNN (idêntica à Tabela 1 do artigo)
inputSize  = size(XTrain,1:3);      % [10 10 3]
numClasses = size(YTrain,1);        % 12 setores de 10°

layers = [
    imageInputLayer(inputSize, 'Normalization','none')

    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer(0.3)

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer(0.3)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer(0.3)

    fullyConnectedLayer(2048)
    reluLayer

    fullyConnectedLayer(512)
    reluLayer

    fullyConnectedLayer(128)
    reluLayer

    fullyConnectedLayer(numClasses)
    sigmoidLayer                             % BCE por classe (1-vs-rest)
];

dlnet = dlnetwork(layerGraph(layers));

%% 4) Hiper-parâmetros de treino
%      β1, β2 e taxa de aprendizagem são iguais aos do artigo.
%      As demais escolhas são locais do script.
learnRate     = 1e-3;                 % η
gradDecay     = 0.9;                  % β1
sqDecay       = 0.999;                % β2

miniBatchSize = 128;                  %% ─ NÃO ESPECIFICADO NO ARTIGO
numEpochs     = 100;                  % conforme artigo
valFreq       = 200;                  %% ─ NÃO ESPECIFICADO NO ARTIGO
patience      = 5;                    %% ─ NÃO ESPECIFICADO NO ARTIGO

% Estados internos do Adam (E[g] e E[g²])
avgGrad   = [];                       % iniciados vazios
avgSqGrad = [];

bestValLoss = inf;
patCount     = 0;
iteration    = 0;
numIterEp    = floor(Ntrain / miniBatchSize);

%% 5) Laço de treinamento personalizado
for epoch = 1:numEpochs
    permEpoch = randperm(Ntrain);        % embaralha a cada época
    for i = 1:numIterEp
        iteration = iteration + 1;
        batchID   = permEpoch((i-1)*miniBatchSize + (1:miniBatchSize));

        % Organiza mini-batch em formato dlarray
        Xb = dlarray(XTrain(:,:,:,batchID), 'SSCB');
        Yb = dlarray(YTrain(:, batchID),    'CB');

        % Forward + back-prop; gradFun implementa BCE manual
        [Ypred, grad, lossTr] = dlfeval(@gradFun, dlnet, Xb, Yb);

        % -------------------------------------------------------------
        % Atualização Adam (detalhada no artigo ↦ Eq.(6) e (7))
        % A chamada MATLAB encapsula β1, β2, bias-corr e passo η.
        [dlnet, avgGrad, avgSqGrad] = adamupdate( ...
            dlnet, grad, avgGrad, avgSqGrad, iteration, ...
            learnRate, gradDecay, sqDecay);
        % -------------------------------------------------------------

        % Validação periódica
        if mod(iteration, valFreq) == 0          %% ─ NÃO ESPECIFICADO
            [lossVal, YvalPred, accVal, precVal, recVal, f1Val] = ...
                valLoss(dlnet, XVal, YVal, miniBatchSize);

            fprintf('Ep %2d | It %5d | Train %.4f | Val %.4f | Acc %.3f | F1 %.3f\n', ...
                    epoch, iteration, lossTr, lossVal, accVal, f1Val);

            % Early-stopping com paciência = 5       %% ─ NÃO ESPECIFICADO
            if lossVal < bestValLoss
                bestValLoss = lossVal; bestNet = dlnet; patCount = 0;
            else
                patCount = patCount + 1;
                if patCount >= patience
                    fprintf('Early-stopping na época %d.\n', epoch);
                    break;
                end
            end
        end
    end
    if patCount >= patience, break; end          % encerra externamente
end

%% 6) Salvar a melhor rede encontrada
save coarseDOA_net.mat bestNet                    %% ─ NÃO ESPECIFICADO
fprintf('Modelo salvo (ValLoss=%.4f)\n', bestValLoss);

% -------------------------------------------------------------------------
% Funções auxiliares
% -------------------------------------------------------------------------
function [YPred, gradients, loss] = gradFun(net, X, Y)
    % Forward
    YPred = forward(net, X);

    % Binary Cross-Entropy (formato classe-independente)
    loss = -mean(Y .* log(YPred + eps) + ...
                (1 - Y) .* log(1 - YPred + eps), 'all');

    % Backward
    gradients = dlgradient(loss, net.Learnables);

    % Converte loss para double (fora do GPU/dlarray)
    loss = double(gather(extractdata(loss)));
end

function [L,YpredTotal, acc, prec, rec, f1] = valLoss(net,Xval,Yval,bs)
    N = size(Yval,2); nIt = floor(N/bs); acc=0;
    YpredTotal = zeros(size(Yval),'single');
    for k = 1:nIt
        idx = (k-1)*bs + (1:bs);
        Xb = dlarray(Xval(:,:,:,idx),'SSCB');
        Yb = Yval(:,idx);
        Ypred = predict(net,Xb);
        loss_k = -mean(Yb.*log(Ypred+eps)+(1-Yb).*log(1-Ypred+eps),'all');
        acc = acc + double(gather(extractdata(loss_k)));
        YpredTotal(:,idx) = gather(extractdata(Ypred));
    end
    L = acc/nIt;

    [acc, prec, rec, f1] = evaluate_metrics(Yval, YpredTotal, 0.5);
end

function [acc, prec, rec, f1] = evaluate_metrics(Ytrue, Ypred, thresh)
    Ybin  = Ypred >= thresh;
    Ytrue = Ytrue > 0.5;

    TP = sum(Ybin & Ytrue, 2);
    FP = sum(Ybin & ~Ytrue, 2);
    FN = sum(~Ybin & Ytrue, 2);
    TN = sum(~Ybin & ~Ytrue, 2);

    prec = mean(TP ./ (TP + FP + eps));
    rec  = mean(TP ./ (TP + FN + eps));
    f1   = mean(2*prec.*rec ./ (prec + rec + eps));
    acc  = mean((TP + TN) ./ (TP + FP + FN + TN + eps));
end
