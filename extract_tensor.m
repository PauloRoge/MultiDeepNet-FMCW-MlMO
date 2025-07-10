% extract_tensor.m
function T = extract_tensor(Y)
    % Calcula a matriz de covariância e extrai os canais: real, imag, angle
    R = (Y * Y') / size(Y,2);   % M x M
    T(:,:,1) = real(R);         % canal 1
    T(:,:,2) = imag(R);         % canal 2
    T(:,:,3) = angle(R);        % canal 3
end