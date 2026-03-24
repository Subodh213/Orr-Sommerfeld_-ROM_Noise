%% exp10_offdiag_damping_correction_ONErun.m
% Cross-term-aware correction using only LrN:
%   Lr_eta = diag(LrN) + eta * offdiag(LrN)
% Compare trajectories in your 2-panel style.
%
% Requires:
%   OS_Op(alpha,beta,N,Re,meanInd)
%   clencurt_WeightsB(N)
%   W2_orthonormalize(U, W)

clear; close all; clc;
addpath(genpath('../1src'));

%% ---------------- SETTINGS ----------------
Re      = 1000;
alpha   = 1;
beta    = 0;
meanInd = 2;

N  = 64;
dt = 1e-2;
K  = 4000;
r  = 25;
k0 = 1;

sigma     = 1e-3;   % fixed in WEIGHTED space
seedNoise = 2;

eta_list = [1.0 0.5 0.2 0.0];  % try a few (1=noisy, 0=diag-only)

%% ---------------- BUILD OPERATOR ----------------
[A,M,~] = OS_Op(alpha,beta,N,Re,meanInd);
n = size(A,1);

L   = -1i*(M\A);
Phi = expm(dt*L);

% weights
w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

t = (0:K-1)'*dt;

%% ---------------- FOM SNAPSHOTS ----------------
rng(1);
q0 = randn(n,1);
q0 = q0 / sqrt(real(q0'*(W2*q0)));  % normalize in W-norm

Qf = zeros(n,K);
q = q0;
for k=1:K
    q = Phi*q;
    Qf(:,k) = q;
end

%% ---------------- CLEAN POD BASIS ----------------
Xw = W*Qf;
[Uw,~,~] = svd(Xw,'econ');
Uall = W\Uw;

Ur0 = W2_orthonormalize(Uall(:,1:r), W);
Lr0 = Ur0' * W2 * L * Ur0;
Pr0 = expm(dt*Lr0);

Ep0 = proj_err(Qf, Ur0, W2);
Em0 = model_err(Qf, Ur0, W2, Pr0, k0);

%% ---------------- NOISY BASIS (mode noise) ----------------
% perturb only the r modes you actually use
Uw_r = W*(Uall(:,1:r));

rng(seedNoise);
Uw_rN = Uw_r + sigma * randn(size(Uw_r));
UrN   = W2_orthonormalize(W\Uw_rN, W);

% noisy reduced generator
LrN = UrN' * W2 * L * UrN;

s = max(real(eig(LrN)));
LrS = LrN - (s + 1e-6)*eye(r);
PrS = expm(dt*LrS);
EmS = model_err(Qf, UrN, W2, PrS, k0);
 plot(EmS)


% projection error for the noisy basis
EpN = proj_err(Qf, UrN, W2);

%% ---------------- Build eta-corrected ROMs ----------------
off = @(X) X - diag(diag(X));

Em_eta = zeros(K, numel(eta_list));
for j = 1:numel(eta_list)
    eta = eta_list(j);

    Lr_eta = diag(diag(LrN)) + eta * off(LrN);
    Pr_eta = expm(dt*Lr_eta);

    Em_eta(:,j) = model_err(Qf, UrN, W2, Pr_eta, k0);
end

%% ---------------- PLOT (your style) ----------------
figure('Color','w','Position',[60 160 1550 520]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% Left: clean
nexttile; hold on; grid on;
plot(t, Ep0, 'LineWidth',1.8, 'DisplayName','Projection error');
plot(t, Em0, 'LineWidth',1.8, 'DisplayName','Model error');

title('Modes identified from clean data','FontWeight','bold');
xlabel('Time'); ylabel('Error');
legend('Location','best'); axis tight;

% Right: noisy + eta-corrections
nexttile; hold on; grid on;
plot(t, EpN, 'LineWidth',1.8, 'DisplayName','Projection error');

% plot each eta model error
for j = 1:numel(eta_list)
    eta = eta_list(j);
    if eta == 1
        ls = '-'; lw = 1.9; name = 'Model error (noisy ROM, \eta=1)';
    elseif eta == 0
        ls = '--'; lw = 2.2; name = 'Model error (diag-only, \eta=0)';
    else
        ls = '--'; lw = 2.0; name = sprintf('Model error (offdiag-damped, \\eta=%.1f)', eta);
    end
    plot(t, Em_eta(:,j), ls, 'LineWidth',lw, 'DisplayName',name);
end

title('Modes identified from noisy modes','FontWeight','bold');
xlabel('Time'); ylabel('Error');
legend('Location','best'); axis tight;

annotation('textbox',[0.78 0.19 0.2 0.65], ...
  'String',sprintf('r = %d\n\\sigma = 10^{-3}\nN = %d\nseed = %d', r, N, seedNoise), ...
  'FitBoxToText','on','EdgeColor','none','FontSize',16);

%% ---------------- Print quick summary ----------------
fprintf('\n=== Offdiag-damping test (one run) ===\n');
fprintf('N=%d, r=%d, sigma=%.1e, seed=%d\n', N, r, sigma, seedNoise);
fprintf('Final clean model error: %.3e\n', Em0(end));
for j=1:numel(eta_list)
    fprintf('Final model error (eta=%.1f): %.3e\n', eta_list(j), Em_eta(end,j));
end

%% ================= HELPERS =================
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
a = Ur' * W2 * Qf(:,k0);
for k = 1:K
    qk = Qf(:,k);
    Em(k) = wn(qk - Ur*a) / max(wn(qk), eps);
    a = Pr*a;
end
end
