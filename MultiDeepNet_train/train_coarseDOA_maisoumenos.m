% train_coarseDOA_stage1_v2.m
% CNN “grossa” (12 classes a 10°) – 1º estágio do Multi-DeepNet
% -------------------------------------------------------------------------
clear; clc; rng default;                                   % reprodutibilidade
tic;

%% 1) Carregar dados -------------------------------------------------------
dataDir = fullfile(pwd,'datasets');
files   = dir(fullfile(dataDir,'dataset_coarse_SNR*.mat'));
assert(~isempty(files),"Nenhum dataset encontrado em 'datasets/'.");

T_all = [];  Y_all = [];
for k = 1:numel(files)
    D     = load(fullfile(files(k).folder,files(k).name),'Tcoarse','Ylabel');
    T_all = cat(4,T_all,D.Tcoarse);          % (10×10×3×N)
    Y_all = cat(2,Y_all,D.Ylabel);           % (12×N)
end
T_all = single(T_all);   Y_all = single(Y_all);

%% 2) Split treino / validação (90 / 10) ----------------------------------
Ntot   = size(Y_all,2);
perm   = randperm(Ntot);
Ntrain = floor(0.9*Ntot);

trainIdx = perm(1:Ntrain);     valIdx = perm(Ntrain+1:end);
XTrain   = T_all(:,:,:,trainIdx);   YTrain = Y_all(:,trainIdx);
XVal     = T_all(:,:,:,valIdx);     YVal   = Y_all(:,valIdx);
clear T_all Y_all;

fprintf("Total=%d  (treino=%d  validação=%d)\n",Ntot,Ntrain,numel(valIdx));

%% 3) Arquitetura CNN (reajustada) ----------------------------------------
inputSize  = size(XTrain,1:3);      % [10 10 3]
numClasses = size(YTrain,1);        % 12

layers = [
    imageInputLayer(inputSize,'Normalization','none')

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer(0.3)
    maxPooling2dLayer(2,'Stride',2)         % 10×10 → 5×5

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer(0.3)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer(0.3)

    globalAveragePooling2dLayer            % 1×1×32

    fullyConnectedLayer(128)
    dropoutLayer(0.4)
    reluLayer

    fullyConnectedLayer(numClasses)
    sigmoidLayer
];

dlnet = dlnetwork(layerGraph(layers));

%% 4) Hiper-parâmetros -----------------------------------------------------
learnRateInit   = 5e-4;
learnRate       = learnRateInit;
lrDecayFactor   = 0.5;
lrDecayStep     = 1500;      % iterações

weightDecay     = 5e-5;      % L2
miniBatchSize   = 256;
numEpochs       = 100;
valFreq         = 100;
patience        = 8;

gradientClip    = 1.0;       % L2-norm clip

avgGrad = [];  avgSqGrad = [];
bestValLoss = inf;  patCount = 0; iteration = 0;
numIterEp = floor(Ntrain/miniBatchSize);

% ---- pesos positivos para BCE ponderada -------------------------------
Npos      = sum(YTrain,2);           % positivos por classe
Nneg      = Ntrain - Npos;
posWeight = 2.0 * single(Nneg./(Npos + eps));   % (12×1)

% buffers para curva de aprendizado
maxPts   = ceil((numEpochs*numIterEp)/valFreq);
trainLossVec = nan(1,maxPts); valLossVec = nan(1,maxPts);
iterVec      = nan(1,maxPts);  lossCount = 0;

%% 5) Loop de treinamento personalizado -----------------------------------
for epoch = 1:numEpochs
    permEpoch = randperm(Ntrain);
    for i = 1:numIterEp
        iteration = iteration + 1;
        idx = permEpoch((i-1)*miniBatchSize + (1:miniBatchSize));

        Xb = dlarray(XTrain(:,:,:,idx),'SSCB');
        Yb = dlarray(YTrain(:,idx),'CB');

        [Ypred,grad,lossTr] = dlfeval(@gradFun,dlnet,Xb,Yb, ...
                                      weightDecay,posWeight);

        % clipping de gradiente (L2 global)
        gradNorm = 0;
        for v = 1:numel(grad.Value)
            g = grad.Value{v};
            gradNorm = gradNorm + sum(g.^2,'all');
        end
        gradNorm = sqrt(gradNorm);
        if gradNorm > gradientClip
            scale = gradientClip/gradNorm;
            for v = 1:numel(grad.Value)
                grad.Value{v} = grad.Value{v} * scale;
            end
        end

        % Adam update
        [dlnet,avgGrad,avgSqGrad] = adamupdate( ...
            dlnet,grad,avgGrad,avgSqGrad,iteration, ...
            learnRate,0.9,0.999);

        % decaimento da LR
        if mod(iteration,lrDecayStep)==0
            learnRate = learnRate * lrDecayFactor;
        end

        % Validação periódica
        if mod(iteration,valFreq)==0
            [lossVal,~,accVal,precVal,recVal,f1Val] = ...
                valLoss(dlnet,XVal,YVal,miniBatchSize);

            fprintf('Ep %2d | It %5d | Train %.4f | Val %.4f | Acc %.3f | Prec %.3f | Rec %.3f | F1 %.3f\n', ...
                    epoch,iteration,lossTr,lossVal,accVal,precVal,recVal,f1Val);

            lossCount = lossCount + 1;
            trainLossVec(lossCount) = lossTr;
            valLossVec(lossCount)   = lossVal;
            iterVec(lossCount)      = iteration;

            if lossVal < bestValLoss
                bestValLoss = lossVal; bestNet = dlnet; patCount = 0;
            else
                patCount = patCount + 1;
                if patCount >= patience, break; end
            end
        end
    end
    if patCount >= patience, break; end
end

%% 6) Salvar rede e curva --------------------------------------------------
save coarseDOA_net.mat bestNet
fprintf('Modelo salvo (ValLoss=%.4f)\n',bestValLoss);

valid = 1:lossCount;
figure;
plot(iterVec(valid),trainLossVec(valid),'-b','LineWidth',1.5); hold on;
plot(iterVec(valid),valLossVec(valid),  '-r','LineWidth',1.5);
xlabel('iterations'); ylabel('Loss'); grid on;
legend('trainLoss','valLoss');
title('Curva de Perda por Iteração');

toc;
% -------------------------------------------------------------------------
% Funções
% -------------------------------------------------------------------------
function [YPred,gradients,loss] = gradFun(net,X,Y,weightDecay,posWeight)
    YPred = forward(net,X);

    w = dlarray(posWeight,'CB');             % (C×1) → broadcast
    lossBCE = -mean( w.*Y.*log(YPred+eps) + ...
                     (1-Y).*log(1-YPred+eps), 'all');

    l2 = 0;
    for v = 1:numel(net.Learnables.Value)
        W = net.Learnables.Value{v};
        l2 = l2 + sum(W.^2,'all');
    end
    loss = lossBCE + 0.5*weightDecay*l2;

    gradients = dlgradient(loss,net.Learnables);
    loss = double(gather(extractdata(loss)));
end

function [L,YpredTot,acc,prec,rec,f1] = valLoss(net,Xval,Yval,bs)
    N = size(Yval,2); nIt = floor(N/bs);
    totLoss = 0;  YpredTot = zeros(size(Yval),'single');
    for k = 1:nIt
        idx = (k-1)*bs + (1:bs);
        Xb = dlarray(Xval(:,:,:,idx),'SSCB');
        Yb = Yval(:,idx);
        Ypred = predict(net,Xb);
        totLoss = totLoss + ...
            -mean(Yb.*log(Ypred+eps)+(1-Yb).*log(1-Ypred+eps),'all');
        YpredTot(:,idx) = gather(extractdata(Ypred));
    end
    L = totLoss/nIt;
    [acc,prec,rec,f1] = metrics(Yval,YpredTot,0.5);
end

function [acc,prec,rec,f1] = metrics(Ytrue,Ypred,th)
    Ybin  = Ypred >= th;     Ytrue = Ytrue > 0.5;
    TP = sum(Ybin & Ytrue,2); FP = sum(Ybin & ~Ytrue,2);
    FN = sum(~Ybin & Ytrue,2); TN = sum(~Ybin & ~Ytrue,2);
    prec = mean(TP./(TP+FP+eps)); rec = mean(TP./(TP+FN+eps));
    f1   = mean(2*prec.*rec./(prec+rec+eps));
    acc  = mean((TP+TN)./(TP+FP+FN+TN+eps));
end
