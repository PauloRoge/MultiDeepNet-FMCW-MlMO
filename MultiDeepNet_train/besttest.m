% train_coarseDOA_stage1.m
% CNN “grossa” de 12 classes (setores de 10°) – primeiro estágio do Multi-DeepNet
% -------------------------------------------------------------------------
clear; clc;

%% >>> 0) Seed externa (para execução em cluster) -------------------------
%      Caso a variável de ambiente ML_SEED exista, usa-se; senão rng default.
if ~isempty(getenv("ML_SEED"))
    rng( str2double(getenv("ML_SEED")) );
else
    rng default;
end
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

%% 2) Dividir treino / validação (90 % / 10 %) – divisão simples
Ntot   = size(Y_all, 2);
perm   = randperm(Ntot);
Ntrain = floor(0.9 * Ntot);                       % 90 % para treino

trainIdx = perm(1:Ntrain);
valIdx   = perm(Ntrain+1:end);

XTrain = T_all(:,:,:,trainIdx);  YTrain = Y_all(:,trainIdx);
XVal   = T_all(:,:,:,valIdx);    YVal   = Y_all(:,valIdx);
clear T_all Y_all;                                   % libera RAM

fprintf("Total=%d  (treino=%d  validação=%d)\n", ...
        Ntot, numel(trainIdx), numel(valIdx));

%% 3) Arquitetura da CNN (Tabela 1 + regularização/droput) ---------------
inputSize  = size(XTrain,1:3);      % [10 10 3]
numClasses = size(YTrain,1);        % 12 setores de 10°

l2 = 5e-4;                          %% >>> fator L2 global

layers = [
    imageInputLayer(inputSize, 'Normalization','none', ...
                    'Name','in')

    convolution2dLayer(3,256,'Padding','same', ...
        'WeightL2Factor',l2,'BiasL2Factor',0,'Name','conv1')
    batchNormalizationLayer('Name','bn1')
    leakyReluLayer(0.3,'Name','lrelu1')

    convolution2dLayer(3,128,'Padding','same', ...
        'WeightL2Factor',l2,'BiasL2Factor',0,'Name','conv2')
    batchNormalizationLayer('Name','bn2')
    leakyReluLayer(0.3,'Name','lrelu2')

    convolution2dLayer(3,64,'Padding','same', ...
        'WeightL2Factor',l2,'BiasL2Factor',0,'Name','conv3')
    batchNormalizationLayer('Name','bn3')
    leakyReluLayer(0.3,'Name','lrelu3')

    fullyConnectedLayer(2048,'WeightL2Factor',l2,'Name','fc1')
    reluLayer('Name','relu_fc1')
    dropoutLayer(0.25,'Name','drop1')             %% >>> Dropout 25 %

    fullyConnectedLayer(512,'WeightL2Factor',l2,'Name','fc2')
    reluLayer('Name','relu_fc2')
    dropoutLayer(0.10,'Name','drop2')             %% >>> Dropout 10 %

    fullyConnectedLayer(128,'WeightL2Factor',l2,'Name','fc3')
    reluLayer('Name','relu_fc3')

    fullyConnectedLayer(numClasses,'WeightL2Factor',l2,'Name','fc_out')
    sigmoidLayer('Name','sigmoid')
];

dlnet = dlnetwork(layerGraph(layers));

%% 4) Hiper-parâmetros de treino -----------------------------------------
learnRate0   = 1e-3;                 % η inicial
decayGamma   = 0.95;                 %% >>> decaimento exponencial por época
gradDecay    = 0.9;                  % β1
sqDecay      = 0.999;                % β2

miniBatchSize = 128;                 % definido empiricamente c/ GTX-3070
numEpochs     = 100;                 % artigo
valFreq       = 200;                 % ~duas validações/época p/ 420 k amostras
patience      = 8;                   %% >>> paciência maior, monitorando F1

avgGrad = []; avgSqGrad = [];        % estados Adam
bestF1  = -inf;  patCount=0; iteration=0;
numIterEp = floor(numel(trainIdx) / miniBatchSize);

%% 5) Laço de treinamento personalizado ----------------------------------
for epoch = 1:numEpochs
    learnRate = learnRate0 * decayGamma^(epoch-1); %% >>> scheduler LR

    permEpoch = randperm(numel(trainIdx));          % embaralha a cada época
    for i = 1:numIterEp
        iteration = iteration + 1;
        batchID   = permEpoch((i-1)*miniBatchSize + (1:miniBatchSize));

        Xb = dlarray(XTrain(:,:,:,batchID), 'SSCB');
        Yb = dlarray(YTrain(:, batchID),    'CB');

        % Forward + back-prop
        [Ypred, grad, lossTr] = dlfeval(@gradFun, dlnet, Xb, Yb);

        % >>> Gradient Clipping (norma global)
        grad = dlupdate(@(g) max(min(g,5),-5), grad);

        % Atualização Adam
        [dlnet, avgGrad, avgSqGrad] = adamupdate( ...
            dlnet, grad, avgGrad, avgSqGrad, iteration, ...
            learnRate, gradDecay, sqDecay);

        % Validação periódica
        if mod(iteration, valFreq) == 0
            [lossVal, YvalPred, accVal, precVal, recVal, f1Val] = ...
                valLoss(dlnet, XVal, YVal, miniBatchSize);

            fprintf(['Ep %2d | It %6d | LR %.3e | Train %.4f | ' ...
                     'Val %.4f | Acc %.3f | Prec %.3f | Rec %.3f | F1 %.3f\n'], ...
                    epoch, iteration, learnRate, lossTr, lossVal, ...
                    accVal, precVal, recVal, f1Val);

            % Early-Stopping baseado em F1
            if f1Val > bestF1
                bestF1 = f1Val;  bestNet = dlnet;  patCount = 0;
            else
                patCount = patCount + 1;
                if patCount >= patience
                    fprintf('Early-stopping na época %d (F1 não melhora há %d val).\n', ...
                            epoch, patience);
                    break;
                end
            end
        end
    end
    if patCount >= patience, break; end
end

%% 6) Salvar a melhor rede encontrada ------------------------------------
save coarseDOA_net.mat bestNet bestF1 epoch iteration
fprintf('Modelo salvo (Melhor F1=%.4f)\n', bestF1);
toc

% -------------------------------------------------------------------------
% Funções auxiliares (inalteradas exceto docstrings) ----------------------
function [YPred, gradients, loss] = gradFun(net, X, Y)
    % Forward
    YPred = forward(net, X);

    % Binary Cross-Entropy multi-rótulo
    loss = -mean(Y .* log(YPred + eps) + ...
                (1 - Y) .* log(1 - YPred + eps), 'all');

    gradients = dlgradient(loss, net.Learnables);
    loss = double(gather(extractdata(loss)));
end

function [L,YpredTotal, acc, prec, rec, f1] = valLoss(net,Xval,Yval,bs)
    N = size(Yval,2); nIt = floor(N/bs); lossSum=0;
    YpredTotal = zeros(size(Yval),'single');
    for k = 1:nIt
        idx = (k-1)*bs + (1:bs);
        Xb = dlarray(Xval(:,:,:,idx),'SSCB');
        Yb = Yval(:,idx);
        Ypred = predict(net,Xb);

        loss_k = -mean(Yb.*log(Ypred+eps)+(1-Yb).*log(1-Ypred+eps),'all');
        lossSum = lossSum + double(gather(extractdata(loss_k)));
        YpredTotal(:,idx) = gather(extractdata(Ypred));
    end
    L = lossSum/nIt;
    [acc, prec, rec, f1] = evaluate_metrics(Yval, YpredTotal, 0.5);
end

function [acc, prec, rec, f1] = evaluate_metrics(Ytrue, Ypred, thresh)
    Ybin  = Ypred >= thresh;      Ytrue = Ytrue > 0.5;

    TP = sum(Ybin & Ytrue, 2);
    FP = sum(Ybin & ~Ytrue, 2);
    FN = sum(~Ybin & Ytrue, 2);
    TN = sum(~Ybin & ~Ytrue, 2);

    prec = mean(TP ./ (TP + FP + eps));
    rec  = mean(TP ./ (TP + FN + eps));
    f1   = mean(2*prec.*rec ./ (prec + rec + eps));
    acc  = mean((TP + TN) ./ (TP + FP + FN + TN + eps));
end
