clear; clc; rng('default');
tic;

%% 1) Carrega múltiplos datasets (ex: SNR+15, SNR+20)
dataDir = fullfile(pwd,'datasets');
files = dir(fullfile(dataDir,'dataset_coarse_SNR+10.mat'));
assert(~isempty(files),"Nenhum dataset encontrado.");

T_all = []; Y_all = [];
for k = 1:numel(files)
    D = load(fullfile(files(k).folder,files(k).name),'Tcoarse','Ylabel');
    T_all = cat(4, T_all, single(D.Tcoarse));
    Y_all = cat(2, Y_all, single(D.Ylabel));
end

%% 2) Split treino / validação
Ntot = size(Y_all,2);
perm = randperm(Ntot);
Ntrain = floor(0.9*Ntot);

trainIdx = perm(1:Ntrain);
valIdx   = perm(Ntrain+1:end);

XTrain = T_all(:,:,:,trainIdx);   YTrain = Y_all(:,trainIdx);
XVal   = T_all(:,:,:,valIdx);     YVal   = Y_all(:,valIdx);
clear T_all Y_all;

fprintf("Total=%d  (treino=%d  validação=%d)\n",Ntot,Ntrain,numel(valIdx));

%% 3) Arquitetura com melhorias
inputSize  = size(XTrain,1:3);    
numClasses = size(YTrain,1);      

layers = [
    imageInputLayer(inputSize,'Normalization','none')

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer(0.3)

    fullyConnectedLayer(256,'WeightL2Factor',1e-4)
    reluLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(128,'WeightL2Factor',1e-4)
    reluLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(numClasses)
    sigmoidLayer
];

dlnet = dlnetwork(layerGraph(layers));

%% 4) Hiperparâmetros
learnRate     = 1e-3;
gradDecay     = 0.9;
sqDecay       = 0.999;
miniBatchSize = 64;
numEpochs     = 20;
valFreq       = 200;
patience      = 10;

avgGrad = []; avgSqGrad = [];
bestValLoss = inf; patCount = 0; iteration = 0;
numIterEp = floor(Ntrain/miniBatchSize);

%% 5) Treinamento com métricas
for epoch = 1:numEpochs
    ticEpoch = tic;
    permEpoch = randperm(Ntrain);

    for i = 1:numIterEp
        iteration = iteration + 1;
        batchIDs = permEpoch((i-1)*miniBatchSize + (1:miniBatchSize));

        Xb = dlarray(XTrain(:,:,:,batchIDs),'SSCB');
        Yb = dlarray(YTrain(:,batchIDs),'CB');

        [Yp,grad,lossTr] = dlfeval(@gradFun,dlnet,Xb,Yb);

        [dlnet,avgGrad,avgSqGrad] = adamupdate( ...
            dlnet,grad,avgGrad,avgSqGrad, ...
            iteration,learnRate,gradDecay,sqDecay);

        if mod(iteration,valFreq)==0
            [lossVal, YValPred] = valLoss(dlnet,XVal,YVal,miniBatchSize);
            [acc, prec, rec, f1] = evaluate_metrics(YVal, YValPred, 0.5);

            fprintf('Ep %2d | It %5d | Train %.4f | Val %.4f | Acc %.3f | Prec %.3f | Rec %.3f | F1 %.3f\n', ...
                    epoch, iteration, lossTr, lossVal, acc, prec, rec, f1);

            if lossVal < bestValLoss
                bestValLoss = lossVal; bestNet = dlnet; patCount = 0;
            else
                patCount = patCount+1;
                if patCount >= patience
                    fprintf('Early-stopping na época %d.\n',epoch);
                    break;
                end
            end
        end
    end
    fprintf("Epoch %d concluída em %.1f s\n", epoch, toc(ticEpoch));
    if patCount >= patience, break; end
end

%% 6) Salvar modelo
save coarseDOA_net.mat bestNet
fprintf('Modelo salvo em coarseDOA_net.mat (ValLoss=%.4f)\n',bestValLoss);

%% ------------------------------------------------------------------------
function [YPred,gradients,loss] = gradFun(net,X,Y)
    YPred = forward(net,X);
    loss  = -mean(Y.*log(YPred+eps)+(1-Y).*log(1-YPred+eps),'all');
    gradients = dlgradient(loss,net.Learnables);
    loss = double(gather(extractdata(loss)));
end

function [L,YpredTotal] = valLoss(net,Xval,Yval,bs)
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
