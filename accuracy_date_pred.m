% coarseDOA_MC.m
% Monte Carlo da acurácia global do coarseDOAnet vs. SNR
% ------------------------------------------------------------------------
rng(0,'twister');                              % reprodutibilidade

% Parâmetros do cenário
M         = 10;                                % sensores da ULA
snapshots = 10;                                % snapshots por rodada
fc        = 78.737692e9;                       % [Hz]
lambda    = 3e8/fc;                            % [m]
delta     = lambda/2;                          % espaçamento
SNRdB     = -10:5:15;                          % SNRs avaliados
N_MC      = 1000;                              % rodadas por SNR
nUsersMax = 3;                                 % fontes simultâneas

% Caminhos fixos para Python
pythonExe  = 'C:\Users\JR\AppData\Local\Programs\Python\Python311\python.exe';
scriptPath = fullfile(pwd,'run_coarse_predict.py');

% Pool paralelo (limite-se a poucos workers se tiver pouca RAM)
if isempty(gcp('nocreate'))
    parpool('IdleTimeout',120);                % usa default #workers
end

accuracySNR = zeros(size(SNRdB));

for si = 1:numel(SNRdB)
    snr = SNRdB(si);
    correct = false(N_MC,1);

    parfor mc = 1:N_MC
        % ---------- sorteia parâmetros ----------
        nUsers = randi([1,nUsersMax]);
        AoA    = -60 + 120*rand(1,nUsers);     % graus
        d      = 10*rand(1,nUsers);            % metros

        % ---------- gera sinal e tensor ----------
        [Y,~] = signals(M,snapshots,delta,lambda,AoA,nUsers,d,snr);
        T     = extract_tensor(Y);             % 10×10×3

        % ---------- salva tensor temporário ------
        tmpIn = [tempname '.mat'];
        mf = matfile(tmpIn,'Writable',true);
        mf.T = T;

        % ---------- chamada Python ---------------
        cmd = sprintf('"%s" "%s" "%s"', pythonExe, scriptPath, tmpIn);
        [status,out] = system(cmd);

        if status==0
            predClass = str2double(strtrim(out));
        else
            predClass = NaN;                   % contabiliza erro
        end

        % ---------- rótulo verdadeiro ------------
        gtClass = min(floor((AoA(1)+60)/10),11);

        correct(mc) = (predClass==gtClass);
        delete(tmpIn);
    end

    accuracySNR(si) = mean(correct,'omitnan');
    fprintf('SNR %+3d dB  |  Acc = %.4f\n',snr,accuracySNR(si));
end

% ---------- gráfico final ----------------------
figure
plot(SNRdB,accuracySNR,'-o','LineWidth',1.8);
xlabel('SNR (dB)'), ylabel('Acurácia global'), grid on
title('CoarseDOAnet – Acurácia vs. SNR');

% extract_tensor.m
function T = extract_tensor(Y)
    % Calcula a matriz de covariância e extrai os canais: real, imag, angle
    R = (Y * Y') / size(Y,2);   % M x M
    T(:,:,1) = real(R);         % canal 1
    T(:,:,2) = imag(R);         % canal 2
    T(:,:,3) = angle(R);        % canal 3
end
