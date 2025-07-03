% coarseDOA_MC.m
% Monte Carlo da acurácia global do coarseDOAnet vs. SNR
rng(0,'twister');

% Parâmetros gerais
M           = 10;                   % número de sensores
snapshots   = 10;                   % snapshots por rodada
frequency   = 78.737692e9;          % Frequência central f_c = 78.737692 GHz
lambda      = 3e8 / frequency;      % Comprimento de onda (λ = c/f)
delta       = lambda/2;             % espaçamento entre elementos
SNRdB       = -10:5:15;             % vetor de SNR [dB]
N_MC        = 1000;                 % rodadas Monte Carlo por SNR
nUsersMax   = 3;                    % máximo de fontes simultâneas

if isempty(gcp('nocreate')), parpool; end
accuracySNR = zeros(size(SNRdB));

for si = 1:numel(SNRdB)
    snr = SNRdB(si);
    correct = false(N_MC,1);

    parfor mc = 1:N_MC
        % gera número de fontes, AoA e distâncias aleatórias
        nUsers = randi([1,nUsersMax]);
        AoA     = -60 + 120*rand(1,nUsers);   % em graus
        d       = 10*rand(1,nUsers);          % em metros

        % sinal + ruído via sua função signals.m
        [Y, ~] = signals(M, snapshots, delta, lambda, AoA, nUsers, d, snr);

        % extrai tensor 10×10×3
        T = extract_tensor(Y);

        % gravação temporária via matfile (v7.3)
        tmpIn = [tempname '.mat'];
        mf = matfile(tmpIn,'Writable',true);
        mf.T = T;

        % chamada Python para inferência
        cmd = sprintf('python run_coarse_predict.py "%s"', tmpIn);
        [~, out] = system(cmd);
        predClass = str2double(strtrim(out));

        % rótulo verdadeiro: setor do primeiro usuário
        gtClass = floor((AoA(1) + 60)/10);

        correct(mc) = (predClass == gtClass);
        delete(tmpIn);
    end

    accuracySNR(si) = mean(correct);
    fprintf('SNR %2d dB → Acurácia = %.4f\n', snr, accuracySNR(si));
end

% plot dos resultados
figure;
plot(SNRdB, accuracySNR, '-o', 'LineWidth',1.8);
xlabel('SNR (dB)');
ylabel('Acurácia global');
title('CoarseDOAnet – Acurácia vs. SNR');
grid on;

% extract_tensor.m
function T = extract_tensor(Y)
    R = (Y * Y') / size(Y,2);
    T(:,:,1) = real(R);
    T(:,:,2) = imag(R);
    T(:,:,3) = angle(R);
end
