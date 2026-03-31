%% exp20_CLASSICAL_PG_ONLY_cleanG_vs_classPG.m
% Compare:
%   (1) FOM on dataset 1
%   (2) Clean Galerkin ROM from CLEAN dataset 1
%   (3) Classical PG ROM with:
%         trial = noisy basis from dataset 1
%         test  = noisy basis from dataset 2 (different IC)
%
% Goal:
%   See what classical PG gives, and whether it fails, relative to
%   a clean Galerkin baseline.

clear; close all; clc;
addpath(genpath('../1src'));

%% -------------------- SETTINGS --------------------
Re = 1000;
N  = 128;
alpha = 1;
beta  = 0;
meanInd = 2;

dt = 1e-2;
K  = 15000;
r  = 8;
k0 = 1;

seed_ic1   = 101;
seed_ic2   = 202;
seed_noise = 3001;

sigma_rel = 5e-5;
idx = [];

%% -------------------- BUILD OS + GENERATOR --------------------
[A,M,y] = OS_Op(alpha,beta,N,Re,meanInd);
n = size(A,1);
if isempty(idx), idx = round(n/2); end

L   = -1i*(M\A);
Phi = expm(dt*L);

% Chebyshev quadrature weights
w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

%% ============================================================
%  DATASET 1 : CLEAN / TRIAL / FOM DATA (IC #1)
% ============================================================
rng(seed_ic1);
q0_1 = randn(n,1);
q0_1 = q0_1 / norm(q0_1);

Qf1 = zeros(n,K);
q = q0_1;
for k = 1:K
    q = Phi*q;
    Qf1(:,k) = q;
end

%% ============================================================
%  DATASET 2 : CLEAN TEST DATA (IC #2)
% ============================================================
rng(seed_ic2);
q0_2 = randn(n,1);
q0_2 = q0_2 / norm(q0_2);

Qf2 = zeros(n,K);
q = q0_2;
for k = 1:K
    q = Phi*q;
    Qf2(:,k) = q;
end

t = (0:K-1)';

%% ============================================================
%  ADD NOISE TO BOTH DATASETS IN WEIGHTED SPACE
% ============================================================
rng(seed_noise);

% noisy dataset 1
Xw1 = W * Qf1;
sigma1 = sigma_rel * std(Xw1(:));
Xw1N = Xw1 + sigma1 * randn(size(Xw1));
Qf1N = W \ Xw1N;

% noisy dataset 2
Xw2 = W * Qf2;
sigma2 = sigma_rel * std(Xw2(:));
Xw2N = Xw2 + sigma2 * randn(size(Xw2));
Qf2N = W \ Xw2N;

fprintf('sigma1 = %.3e\n', sigma1);
fprintf('sigma2 = %.3e\n', sigma2);

fprintf('DATASET 1 noise: ||W(Qf1N-Qf1)||F / ||WQf1||F = %.3e\n', ...
    norm(W*(Qf1N-Qf1),'fro')/max(norm(W*Qf1,'fro'),eps));

fprintf('DATASET 2 noise: ||W(Qf2N-Qf2)||F / ||WQf2||F = %.3e\n', ...
    norm(W*(Qf2N-Qf2),'fro')/max(norm(W*Qf2,'fro'),eps));

%% ============================================================
%  CLEAN GALERKIN ROM FROM CLEAN DATASET 1
% ============================================================
rom_clean = rom_build_clean(Qf1, W, W2, L, r, alpha);
Ur_clean  = rom_clean.Ur;
S1        = rom_clean.s;

chk_clean = norm(Ur_clean' * W2 * Ur_clean - eye(r), 'fro');
fprintf('orth(clean Galerkin basis) = %.2e\n', chk_clean);

Lr_clean = Ur_clean' * W2 * L * Ur_clean;
Pr_clean = expm(dt * Lr_clean);

%% ============================================================
%  CLASSICAL PG: TRIAL BASIS FROM NOISY DATASET 1
% ============================================================
rom_trial = rom_build_clean(Qf1N, W, W2, L, r, alpha);
Ur_trial  = rom_trial.Ur;

chk_trial = norm(Ur_trial' * W2 * Ur_trial - eye(r), 'fro');
fprintf('orth(noisy trial basis) = %.2e\n', chk_trial);

%% ============================================================
%  CLASSICAL PG: TEST BASIS FROM NOISY DATASET 2
% ============================================================
rom_test = rom_build_clean(Qf2N, W, W2, L, r, alpha);
Ur_test  = rom_test.Ur;

chk_test = norm(Ur_test' * W2 * Ur_test - eye(r), 'fro');
fprintf('orth(noisy test basis) = %.2e\n', chk_test);

%% ============================================================
%  CLASSICAL PETROV-GALERKIN
%  trial = noisy basis from dataset 1
%  test  = noisy basis from dataset 2
% ============================================================
M_pg = Ur_test' * W2 * Ur_trial;
K_pg = Ur_test' * W2 * L * Ur_trial;

condMpg = cond(M_pg);
fprintf('cond(M_pg) = %.6e\n', condMpg);
fprintf('||Ur_test'' W2 Ur_trial - I||F = %.3e\n', ...
    norm(Ur_test' * W2 * Ur_trial - eye(r),'fro'));

Lr_pg = M_pg \ K_pg;
Pr_pg = expm(dt * Lr_pg);

%% ============================================================
%  LEAST-STABLE EIGENVALUE INFO
% ============================================================
omF = eig(A,M);
[~,iF] = max(imag(omF));
om_ls = omF(iF);
fprintf('\nFOM least-stable omega: %.6g%+.6gi  (imag=%.6g)\n', ...
    real(om_ls), imag(om_ls), imag(om_ls));

report_least_stable('Clean Galerkin', Lr_clean, alpha);
report_least_stable('Classical PG  ', Lr_pg,    alpha);

%% ============================================================
%  ROLLOUTS
%  Compare on dataset 1 truth (Qf1)
% ============================================================
Qr_clean = roll_Pr(Pr_clean, Ur_clean, W2, Qf1, k0);
Qr_pg    = roll_Pr(Pr_pg,    Ur_trial, W2, Qf1, k0);   % rollout uses TRIAL basis

%% ============================================================
%  ERRORS
% ============================================================
Ep_clean = proj_err(Qf1, Ur_clean, W2, k0);
Ep_pg    = proj_err(Qf1, Ur_trial, W2, k0);   % trial basis determines projection space

[Em_clean, rho_clean] = model_err(Qf1, Ur_clean, W2, Pr_clean, k0);
[Em_pg,    rho_pg]    = model_err(Qf1, Ur_trial, W2, Pr_pg,    k0);

fprintf('\nrho(Pr): cleanG=%.4g classPG=%.4g\n', rho_clean, rho_pg);

%% ============================================================
%  SINGLE-POINT TRAJECTORIES
% ============================================================
qf     = Qf1(idx,:).';
qClean = Qr_clean(idx,:).';
qPG    = Qr_pg(idx,:).';

den_i = max(abs(qf(1)), 1e-14);
rel = @(x) abs(x - qf) ./ den_i;

eClean = rel(qClean);
ePG    = rel(qPG);

%% ============================================================
%  PLOTS
% ============================================================

figure('Color','w');
semilogy(S1,'o-','LineWidth',1.5);
xlabel('Mode index');
ylabel('Singular value');
title('Clean dataset 1 POD singular values');
grid on;
%%
figure('Color','w');
hold on; grid on;
plot(t, real(qf),     'k-',  'LineWidth',2.0, 'DisplayName','FOM');
plot(t, real(qClean), 'b-',  'LineWidth',1.8, 'DisplayName','Clean Galerkin');
%plot(t, real(qPG),    'm-.', 'LineWidth',1.8, 'DisplayName','Classical PG');
xlabel('time');
ylabel(sprintf('Re(q(idx=%d))', idx));
title('FOM vs Clean Galerkin vs Classical PG');
legend('Location','best');
%%
figure('Color','w');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile; hold on; grid on;
plot(t, max(Ep_clean,1e-16), 'b-',  'LineWidth',1.8, 'DisplayName','Proj err clean G');
plot(t, max(Ep_pg,1e-16),    'm-.', 'LineWidth',1.8, 'DisplayName','Proj err classical PG');
xlabel('time'); ylabel('projection error');
title('Projection error');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(Em_clean,1e-16), 'b-',  'LineWidth',1.8, 'DisplayName','Model err clean G');
plot(t, max(Em_pg,1e-16),    'm-.', 'LineWidth',1.8, 'DisplayName','Model err classical PG');
xlabel('time'); ylabel('model error');
title('Model error');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(eClean,1e-16), 'b-',  'LineWidth',1.6, 'DisplayName','Clean G');
plot(t, max(ePG,1e-16),    'm-.', 'LineWidth',1.6, 'DisplayName','Classical PG');
xlabel('time'); ylabel('pointwise rel error');
title('Pointwise error at selected index');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, real(qClean), 'b-',  'LineWidth',1.8, 'DisplayName','Clean G');
plot(t, real(qPG),    'm-.', 'LineWidth',1.8, 'DisplayName','Classical PG');
xlabel('time');
ylabel(sprintf('Re(q(idx=%d))', idx));
title('ROM-only comparison');
legend('Location','best');

figure('Color','w');
imagesc(abs(M_pg)); axis tight; colorbar;
title('|M_{PG}| = |Ur_{test}'' W2 Ur_{trial}|');
xlabel('trial mode');
ylabel('test mode');

%% ===================== LOCAL HELPERS =====================

function Qr = roll_Pr(Pr, Ur, W2, Qf, k0)
K = size(Qf,2);
a = Ur' * W2 * Qf(:,k0);
Qr = zeros(size(Qf));
for k = 1:K
    Qr(:,k) = Ur * a;
    a = Pr * a;
end
end

function Ep = proj_err(Qf, Ur, W2, k0)
K = size(Qf,2);
wn = @(v) sqrt(real(v'*(W2*v)));
den = max(wn(Qf(:,k0)), eps);

Ep = zeros(K,1);
for k = 1:K
    qk = Qf(:,k);
    ak = Ur' * W2 * qk;
    Ep(k) = wn(qk - Ur*ak) / den;
end
end

function [Em, rho] = model_err(Qf, Ur, W2, Pr, k0)
K = size(Qf,2);
wn = @(v) sqrt(real(v'*(W2*v)));
rho = max(abs(eig(Pr)));

Em = zeros(K,1);
a = Ur' * W2 * Qf(:,k0);
den = max(wn(Qf(:,k0)), eps);

for k = 1:K
    qk = Qf(:,k);
    Em(k) = wn(qk - Ur*a) / den;
    a = Pr * a;
end
end

function report_least_stable(name, Lr, alpha)
lam = eig(Lr);
[~,ii] = max(real(lam));
lam_ls = lam(ii);
c_ls = (1i*lam_ls)/alpha;

fprintf('%-16s: lam=%.6g%+.6gi  (Re=%.6g)   c=%.6g%+.6gi\n', ...
    name, real(lam_ls), imag(lam_ls), real(lam_ls), real(c_ls), imag(c_ls));
end