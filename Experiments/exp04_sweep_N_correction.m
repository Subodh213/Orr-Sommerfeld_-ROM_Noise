%% exp_sweepN_oneNoise_TRAJONLY.m
% One noise level sigma = 1e-3
% For each N: make one figure like your screenshot:
%   Left : modes from clean data (proj + model)
%   Right: modes from noisy data (proj + model + corrected model)

clear; close all; clc;
addpath(genpath('../1src'));

%% ---- SETTINGS ----
Re      = 1000;
alpha   = 1;
beta    = 0;
meanInd = 2;

dt = 1e-2;
K  = 10000;      % match your screenshot length
r  = 25;        % match your screenshot
k0 = 1;

N_list = [32 64 128 208 225];  
sigma  = 1e-3;           % FIXED noise (in WEIGHTED space)

% output folder
ts  = datestr(now,'yyyy-mm-dd_HHMMSS');
OUT = fullfile('..','results','exp_sweepN_TRAJONLY',ts);
if ~exist(OUT,'dir'), mkdir(OUT); end
fprintf('Saving to: %s\n', OUT);

t  = (0:K-1);

%% ---- LOOP OVER N ----
for iN = 1:numel(N_list)
    N = N_list(iN);
    fprintf('\n---- N = %d ----\n', N);

    % OS operator + generator
    [A,M,~] = OS_Op(alpha,beta,N,Re,meanInd);
    n = size(A,1);
    L   = -1i*(M\A);
    Phi = expm(dt*L);

    % weights
    w  = clencurt_WeightsB(N);
    W  = diag(sqrt(w));
    W2 = diag(w);

    % ---- FOM snapshots (clean) ----
    rng(1);                     % deterministic per N
    q0 = randn(n,1); q0 = q0/norm(q0);

    Qf = zeros(n,K);
    q = q0;
    for k=1:K
        q = Phi*q;
        Qf(:,k) = q;
    end

    % ---- CLEAN MODES from CLEAN DATA ----
    Xw  = W*Qf;
    [Uw,~,~] = svd(Xw,'econ');
    Uall = W\Uw;

    Ur0 = W2_orthonormalize(Uall(:,1:r), W);
    Lr0 = Ur0' * W2 * L * Ur0;
    Pr0 = expm(dt*Lr0);

    Ep0 = proj_err(Qf, Ur0, W2);
    Em0 = model_err(Qf, Ur0, W2, Pr0, k0);

    % ---- NOISY MODES from NOISY DATA ----
    % noise injected in WEIGHTED snapshot matrix
    rng(2);  % deterministic noise realization per N 
    XwN = Xw + sigma * randn(size(Xw));
    [UwN,~,~] = svd(XwN,'econ');
    UallN = W\UwN;

    UrN = W2_orthonormalize(UallN(:,1:r), W);
    LrN = UrN' * W2 * L * UrN;
    PrN = expm(dt*LrN);

    EpN = proj_err(Qf, UrN, W2);
    EmN = model_err(Qf, UrN, W2, PrN, k0);

    % ---- CORRECTION (trace–I) on reduced generator ----
    Bias = (sigma^2) * trace(L) * eye(r);
    LrC  = LrN - Bias;
    PrC  = expm(dt*LrC);
    EmC  = model_err(Qf, UrN, W2, PrC, k0);

    %% ---- PLOT (ONE FIG PER N) ----
    figure('Color','w','Position',[100 200 1200 350]);
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

    % Left: modes from clean data
    nexttile; hold on; grid on;
    plot(t, Ep0, 'LineWidth',1.6, 'DisplayName','Projection error');
    plot(t, Em0, 'LineWidth',1.6, 'DisplayName','Model error');
    title('Modes identified from clean data','FontWeight','bold');
    xlabel('Time'); ylabel('Error');
    legend('Location','best');
    axis tight;

    % Right: modes from noisy data
    nexttile; hold on; grid on;
    plot(t, EpN, 'LineWidth',1.6, 'DisplayName','Projection error');
    plot(t, EmN, 'LineWidth',1.6, 'DisplayName','Model error (noisy ROM)');
    plot(t, EmC, '--', 'LineWidth',1.8, 'DisplayName','Model error (corrected)');
    title('Modes identified from noisy data','FontWeight','bold');
    xlabel('Time'); ylabel('Error');
    legend('Location','best');
    axis tight;

    % add annotation like your screenshot
    annotation('textbox',[0.83 0.25 0.15 0.5], ...
        'String',sprintf('r = %d\n\\sigma = 10^{-3}\nN = %d', r, N), ...
        'FitBoxToText','on','EdgeColor','none','FontSize',16);

    exportgraphics(gcf, fullfile(OUT, sprintf('traj_compare_N%d.png',N)));
end

fprintf('\nDone. Figures saved in:\n  %s\n', OUT);

%% ===================== HELPERS =====================

function Ep = proj_err(Qf, Ur, W2)
K = size(Qf,2);
wn = @(v) sqrt(real(v'*(W2*v)));
Ep = zeros(K,1);
for k = 1:K
    qk = Qf(:,k);
    ak = Ur' * W2 * qk;
    Ep(k) = wn(qk - Ur*ak) / max(wn(qk), eps);
end
end

function Em = model_err(Qf, Ur, W2, Pr, k0)
K = size(Qf,2);
wn = @(v) sqrt(real(v'*(W2*v)));
Em = zeros(K,1);
a  = Ur' * W2 * Qf(:,k0);
for k = 1:K
    qk = Qf(:,k);
    Em(k) = wn(qk - Ur*a) / max(wn(qk), eps);
    a = Pr*a;
end
end
