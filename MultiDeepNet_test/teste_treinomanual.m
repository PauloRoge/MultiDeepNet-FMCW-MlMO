clear; clc; rng('default');

%% 1. Parâmetros da simulação
M         = 10;                  % Número de antenas receptoras (ULA)
frequency = 78.737692e9;         % Frequência central f_c = 78.737692 GHz
lambda    = 3e8 / frequency;     % Comprimento de onda (λ = c/f)
delta     = lambda / 2;          % Espaçamento entre elementos: λ/2
snapshots = 10;                  % Número de snapshots
AoA       = 15;                  % Ângulo real de chegada (graus)
dist      = 9;                   % Distância real do usuário (mesma unidade de r)
snr_dB    = 0;                  % SNR em dB
K         = 1;                   % Número de fontes

%% 2. Geração do sinal
X = signals(M, snapshots, delta, lambda, AoA, K, dist, snr_dB);

%% 3. Construção do tensor de entrada
R = (X * X') / snapshots;
T = zeros(M, M, 3, 'single');
T(:,:,1) = real(R);
T(:,:,2) = imag(R);
T(:,:,3) = angle(R);
Xtest = reshape(T, [M M 3 1]);
dlX   = dlarray(Xtest, 'SSCB');

%% 4. Inferência com a rede treinada
load('coarseDOA_net.mat','bestNet');
YPred = predict(bestNet, dlX);
probs = extractdata(YPred);

%% 5. Visualização de probabilidades (blocos)
centros        = -55:10:55;
intervalos_str = arrayfun(@(c) sprintf('[%d,%d]',c-5,c+5), centros, 'UniformOutput', false);
figure('Units','normalized','Position',[0.25 0.4 0.5 0.2]);
hold on;
cmap       = parula(256);
probs_norm = (probs - min(probs)) / (max(probs) - min(probs));
probs_idx  = max(1, round(probs_norm*255)+1);

for i = 1:12
    x = [i-1 i i i-1];  y = [0 0 1 1];
    fill(x, y, cmap(probs_idx(i),:), 'EdgeColor','k');
    text(i-0.5, -0.15, intervalos_str{i}, ...
         'Rotation',45, 'HorizontalAlignment','right', ...
         'VerticalAlignment','middle','FontSize',8);
end
xlim([0 12]);  ylim([-0.3 1]);
axis off;
colorbar('Ticks',[0 0.5 1],'TickLabels',{'0','0.5','1'});
colormap(parula);

% %% 6. Grid angular e marcação do usuário
% theta = -60:10:60;
% r     = 10;      % raio do setor
% 
% figure; hold on; axis equal; axis off;
% 
% % linhas e rótulos
% for ang = theta
%     phi = 90 - ang;
%     plot([0 r*cosd(phi)], [0 r*sind(phi)], 'Color',[0.5 0.5 0.5]);
%     text(1.1*r*cosd(phi), 1.1*r*sind(phi), sprintf('%d°',ang), ...
%          'HorizontalAlignment','center');
% end
% 
% % arco
% arc     = linspace(min(theta), max(theta), 200);
% phi_arc = 90 - arc;
% plot(r*cosd(phi_arc), r*sind(phi_arc), 'Color',[0.5 0.5 0.5]);
% 
% % usuário
% phi_user = 90 - AoA;
% x_user   = dist*cosd(phi_user);
% y_user   = dist*sind(phi_user);
% h_user   = plot(x_user, y_user, 'x', 'MarkerFaceColor','r', 'MarkerSize',8);
% 
% % % legenda com distância e ângulo
% % legend_str = sprintf('Usuário: dist=%.1f, ang=%d°', dist, AoA);
% % legend(h_user, legend_str, 'Location','northeastoutside');

%% 6. Grid angular com mapa de probabilidade, grades pretas e marcação do usuário
theta      = -60:10:60;                       % ângulos das divisões
r          = 10;                              % raio do setor
probs_norm = (probs - min(probs)) / (max(probs) - min(probs));  
cmap       = parula(256);                     % colormap

figure; hold on; axis equal; axis off;

% 1) Preenche cada setor (wedge) com base em probs
for i = 1:numel(theta)-1
    ang1 = theta(i);
    ang2 = theta(i+1);
    phi1 = 90 - ang1;
    phi2 = 90 - ang2;
    phis = linspace(phi1, phi2, 50);
    xs   = [0, r*cosd(phis), 0];
    ys   = [0, r*sind(phis), 0];
    ci   = max(1, round(probs_norm(i)*255) + 1);
    fill(xs, ys, cmap(ci,:), 'EdgeColor', 'none');
end

% 2) Desenha linhas radiais em preto e rótulos
for ang = theta
    phi = 90 - ang;
    plot([0, r*cosd(phi)], [0, r*sind(phi)], 'Color', 'k', 'LineWidth', 1);
    text(1.1*r*cosd(phi), 1.1*r*sind(phi), sprintf('%d°', ang), ...
         'HorizontalAlignment', 'center');
end

% 3) Desenha o arco superior em preto
arc     = linspace(min(theta), max(theta), 200);
phi_arc = 90 - arc;
plot(r*cosd(phi_arc), r*sind(phi_arc), 'Color', 'k', 'LineWidth', 1);

% 4) Marca a posição real do usuário
phi_user = 90 - AoA;
x_user   = dist * cosd(phi_user);
y_user   = dist * sind(phi_user);
h_user   = plot(x_user, y_user, 'x', 'MarkerEdgeColor', 'r', 'MarkerSize', 8);

% 5) Adiciona colorbar indicando probabilidade
colormap(parula);
c = colorbar('Location', 'eastoutside');
c.Label.String = 'Probabilidade';
caxis([0 1]);

% % 6) Legenda com distância e ângulo do usuário
% legend(h_user, sprintf('Usuário: dist=%.1f, ang=%d°', dist, AoA), ...
%        'Location', 'northeastoutside');



%% 7. Impressão dos resultados
fprintf('\n[INFO] Ângulo verdadeiro: %d°\n', AoA);
fprintf('[INFO] Vetor de probabilidades:\n'); disp(probs);
detected = centros(probs>0.5);
if isempty(detected)
    fprintf('[INFO] Nenhuma classe detectada (prob>0.5)\n');
else
    fprintf('[INFO] Classes detectadas: '); fprintf('%d° ',detected); fprintf('\n');
end
