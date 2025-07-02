function gera_Y(snapshots, AoA, dist, snr_dB, K)
    clc; rng('default');
    
    M = 10;
    frequency = 78.737692e9;
    lambda = 3e8 / frequency;
    delta  = lambda / 2;

    Y = signals(M, snapshots, delta, lambda, AoA, K, dist, snr_dB);
    save('entrada_Y.mat', 'Y');
end


function [Y, Z] = signals(M, snapshots, delta, ...
    lambda, AoA, numSources, d, SNRdB)

    PL   = (lambda/(4*pi)).^2 ./ (d.^2); 
    beta = sqrt(PL);                    

    H = zeros(M, numSources);
    for s = 1:numSources
        a      = responsearray(M, delta, lambda, AoA(s)); 
        H(:,s) = beta(s) * a;
    end

    X = (randn(numSources, snapshots) + ...
        1j*randn(numSources, snapshots)) / sqrt(2);
    Y_sig = H * X;
    P_signal = mean(abs(Y_sig).^2, 'all');
    noiseVar = P_signal / (10^(SNRdB/10));
    Z = sqrt(noiseVar) * (randn(M, snapshots) + ...
        1j*randn(M, snapshots)) / sqrt(2);
    Y = Y_sig + Z;
end

function a = responsearray(M, delta, lambda, theta)
    gamma = 2*pi * delta / lambda;
    a     = exp(-1j * gamma * (0:M-1)' * sind(theta));
end
