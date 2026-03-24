%% exp09_datadriven_correction_from_noisy_basis.m
% Build corrected reduced generator from data:
%   Lr_data ~ argmin || dA - Lr*A ||_F
% and compare trajectories (clean vs noisy Galerkin vs data-corrected).
%
% Requires:
%   OS_Op, clencurt_WeightsB, W2_orthonormalize

clear; close all; clc;
addpath(genpath('../1src'));

%% SETTINGS
Re=1000; alpha=1; beta=0; meanInd=2;
N=128; dt=1e-2; K=10000; r=25; k0=1;
sigma=1e-3; seedNoise=2;

lambda = 1e-8;      % ridge regularization (try 1e-10 ... 1e-6)
stab_shift = true;  % enforce max real(eig(Lr)) <= 0

%% Operator + weights
[A,M,~] = OS_Op(alpha,beta,N,Re,meanInd);
n = size(A,1);
L = -1i*(M\A);
Phi = expm(dt*L);

w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

t = (0:K-1);

%% FOM snapshots
rng(1);
q0 = randn(n,1);
q0 = q0 / sqrt(real(q0'*(W2*q0)));

Qf = zeros(n,K);
q = q0;
for k=1:K
    q = Phi*q;
    Qf(:,k)=q;
end

%% Clean basis + clean ROM
Xw = W*Qf;
[Uw,~,~]=svd(Xw,'econ');
Uall = W\Uw;

Ur0 = W2_orthonormalize(Uall(:,1:r), W);
Lr0 = Ur0' * W2 * L * Ur0;
Pr0 = expm(dt*Lr0);

Em0 = model_err(Qf, Ur0, W2, Pr0, k0);
Ep0 = proj_err(Qf, Ur0, W2);

%% Noisy basis (perturb ONLY r modes)
Uw_r = W*(Uall(:,1:r));
rng(seedNoise);
Uw_rN = Uw_r + sigma*randn(size(Uw_r));
UrN   = W2_orthonormalize(W\Uw_rN, W);

LrN = UrN' * W2 * L * UrN;
PrN = expm(dt*LrN);

EmN = model_err(Qf, UrN, W2, PrN, k0);
EpN = proj_err(Qf, UrN, W2);

%% ===== Data-driven corrected generator in the NOISY coordinates =====
% reduced coefficients in noisy basis
Ared = UrN' * W2 * Qf;              % r×K

% finite difference derivative (use first K-1 points)
dA = (Ared(:,2:K) - Ared(:,1:K-1)) / dt;   % r×(K-1)
A0 = Ared(:,1:K-1);                        % r×(K-1)

% ridge LS: Lr_data = dA*A0'*(A0*A0' + lambda I)^(-1)
G = (A0*A0' + lambda*eye(r));
Lr_data = (dA*A0') / G;

% Optional stability enforcement (very useful for single realization)
if stab_shift
    lam = eig(Lr_data);
    s = max(real(lam));
    if s > 0
        Lr_data = Lr_data - (s + 1e-6)*eye(r);
    end
end

PrD = expm(dt*Lr_data);
EmD = model_err(Qf, UrN, W2, PrD, k0);  % still compares in physical space using UrN

%% PLOT (your style)
figure('Color','w','Position',[60 160 1550 520]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile; hold on; grid on;
plot(t, Ep0, 'LineWidth',1.8,'DisplayName','Projection error');
plot(t, Em0, 'LineWidth',1.8,'DisplayName','Model error');
title('Modes identified from clean data','FontWeight','bold');
xlabel('Time'); ylabel('Error'); legend('Location','best'); axis tight;

nexttile; hold on; grid on;
plot(t, EpN, 'LineWidth',1.8,'DisplayName','Projection error');
%plot(t, EmN, 'LineWidth',1.8,'DisplayName','Model error (noisy Galerkin)');
plot(t, EmD, '--','LineWidth',2.2,'DisplayName','Model error (data-corrected)');
title('Modes identified from noisy modes','FontWeight','bold');
xlabel('Time'); ylabel('Error'); legend('Location','best'); axis tight;

annotation('textbox',[0.78 0.19 0.2 0.65], ...
  'String',sprintf('r = %d\n\\sigma = 10^{-3}\nN = %d\nseed = %d', r, N, seedNoise), ...
  'FitBoxToText','on','EdgeColor','none','FontSize',16);

fprintf('\nFinal model errors: clean=%.3e | noisy=%.3e | data-corr=%.3e\n', Em0(end), EmN(end), EmD(end));

%% ===== helpers =====
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
