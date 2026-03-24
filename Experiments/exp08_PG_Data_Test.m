%% ============================================================
%  Petrov–Galerkin ROM with TEST SPACE built from SEPARATE DATA
%  (same sensors/state dimension; different trajectory/realization)
%
%  Trial basis  : VrN  (noisy POD modes from forward data Qf)
%  Test basis   : Wr_raw (POD modes from an independent forward dataset Qf2)
%  Bi-orthonorm : enforce Wr' * W2 * VrN = I
%  PG ROM       : LrPG = Wr' * W2 * L * VrN
%
%  Requires your existing functions:
%    OS_Op, clencurt_WeightsB,
%    W2_orthonormalize, proj_err,
%    model_err_galerkin, model_err_petrov
% ============================================================
clear; close all; clc

%% ---------------- SETTINGS ----------------
Re      = 1000;
alpha   = 1;
beta    = 0;
meanInd = 2;

N  = 158;
dt = 1e-2;
K  = 10000;
r  = 25;
k0 = 1;

sigma     = 1e-3;   % fixed noise in WEIGHTED space (modes noise)
seedNoise = 2;

% TEST-DATA settings (separate dataset for test basis)
Ktest   = K;     % snapshot length for test dataset (can be smaller)
nICtest = 1;     % number of independent forward trajectories to enrich test basis
seedTestBase = 11;

% Stabilization epsilon
stab_eps = 1e-6;

%% ---------------- BUILD OPERATOR ----------------
[A,M,~] = OS_Op(alpha,beta,N,Re,meanInd);
n = size(A,1);

L   = -1i*(M\A);
Phi = expm(dt*L);

% weights
w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

t = (0:K-1);

%% ---------------- FOM SNAPSHOTS (FORWARD DATASET #1) ----------------
rng(1);
q0 = randn(n,1);
q0 = q0 / sqrt(real(q0'*(W2*q0)));  % normalize in W2-norm

Qf = zeros(n,K);
q = q0;
for k=1:K
    q = Phi*q;
    Qf(:,k) = q;
end

%% ---------------- CLEAN POD BASIS (trial) ----------------
Xw = W*Qf;
[Uw,~,~] = svd(Xw,'econ');
Uall = W\Uw;

Ur0 = W2_orthonormalize(Uall(:,1:r), W);    % clean trial basis
Lr0 = Ur0' * W2 * L * Ur0;
Pr0 = expm(dt*Lr0);

Ep0 = proj_err(Qf, Ur0, W2);
Em0 = model_err_galerkin(Qf, Ur0, W2, Pr0, k0);

%% ---------------- NOISY TRIAL BASIS (modes noise) ----------------
% Add noise in weighted space to the clean POD modes
Uw_r = W*(Uall(:,1:r));     % weighted representation of clean modes

rng(seedNoise);
Uw_rN = Uw_r + sigma*randn(size(Uw_r));     % add noise in weighted coords
VrN   = W2_orthonormalize(W\Uw_rN, W);      % noisy trial basis (W2-orthonormal)

% Noisy Galerkin reduced generator
LrN = VrN' * W2 * L * VrN;
PrN = expm(dt*LrN);

EpN = proj_err(Qf, VrN, W2);
EmN = model_err_galerkin(Qf, VrN, W2, PrN, k0);

% Noisy Galerkin + stability shift
sN  = max(real(eig(LrN)));
LrS = LrN - (max(sN,0) + stab_eps)*eye(r);
PrS = expm(dt*LrS);
EmS = model_err_galerkin(Qf, VrN, W2, PrS, k0);

%% ---------------- BUILD TEST BASIS FROM SEPARATE FORWARD DATASET #2 ----------------
% Same sensors/state, same operator Phi, but a different run/trajectory.
% In real-data terms: "repeat experiment" or "different initial disturbance",
% not "different sensors".

Qf2 = [];  % will concatenate multiple forward runs if nICtest > 1
for j = 1:nICtest
    rng(seedTestBase + j);
    q02 = randn(n,1);
    q02 = q02 / sqrt(real(q02'*(W2*q02)));

    q2 = q02;
    Qtmp = zeros(n,Ktest);
    for k=1:Ktest
        q2 = Phi*q2;
        Qtmp(:,k) = q2;
    end

    Qf2 = [Qf2, Qtmp]; %#ok<AGROW>
end

% POD on separate forward snapshots (weighted POD)
Xw2 = W*Qf2;
[Uw2,~,~] = svd(Xw2,'econ');
Uall2 = W\Uw2;

Wr_raw = W2_orthonormalize(Uall2(:,1:r), W);   % raw test basis (W2-orthonormal)

% Enforce bi-orthonormality: Wr' * W2 * VrN = I
Mbi = Wr_raw' * W2 * VrN;          % r×r "cross Gram"
fprintf('cond(Mbi) = %.3e\n', cond(Mbi));

% If Mbi is ill-conditioned, this can blow up. Two options:
%   (A) strict inverse (what you had): Wr = Wr_raw / (Mbi')
%   (B) pseudo-inverse (more stable):  Wr = Wr_raw * pinv(Mbi')
use_pinv = false;

if use_pinv
    Wr = Wr_raw * pinv(Mbi');
else
    Wr = Wr_raw / (Mbi');
end

fprintf('||Wr''W2VrN - I||_F = %.3e\n', norm(Wr'*W2*VrN - eye(r),'fro'));

%% ---------------- PETROV–GALERKIN REDUCED GENERATOR ----------------
LrPG = Wr' * W2 * L * VrN;
PrPG = expm(dt*LrPG);

% Optional: stability shift for PG too
sPG   = max(real(eig(LrPG)));
LrPGS = LrPG - (max(sPG,0) + stab_eps)*eye(r);
PrPGS = expm(dt*LrPGS);

% Petrov–Galerkin model errors (coeffs use Wr, reconstruction uses VrN)
EmPG  = model_err_petrov(Qf, VrN, Wr, W2, PrPG,  k0);
EmPGS = model_err_petrov(Qf, VrN, Wr, W2, PrPGS, k0);

%% ---------------- PRINT STABILITY NUMBERS ----------------
fprintf('\n=== Stability check (max Re eig) ===\n');
fprintf('clean Galerkin   : %.3e\n', max(real(eig(Lr0))));
fprintf('noisy Galerkin   : %.3e\n', max(real(eig(LrN))));
fprintf('noisy + shift    : %.3e\n', max(real(eig(LrS))));
fprintf('Petrov–Galerkin  : %.3e\n', max(real(eig(LrPG))));
fprintf('PG + shift       : %.3e\n', max(real(eig(LrPGS))));

fprintf('\nFinal model errors:\n');
fprintf('clean Galerkin   Em(T)=%.3e\n', Em0(end));
fprintf('noisy Galerkin   Em(T)=%.3e\n', EmN(end));
fprintf('noisy + shift    Em(T)=%.3e\n', EmS(end));
fprintf('Petrov–Galerkin  Em(T)=%.3e\n', EmPG(end));
fprintf('PG + shift       Em(T)=%.3e\n', EmPGS(end));

%% ---------------- PLOT (2-panel, your style) ----------------
figure('Color','w','Position',[60 160 1550 520]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% Left: clean
nexttile; hold on; grid on;
plot(t, Ep0, 'LineWidth',1.8, 'DisplayName','Projection error');
plot(t, Em0, 'LineWidth',1.8, 'DisplayName','Model error');
title('Modes identified from clean data','FontWeight','bold');
xlabel('Time'); ylabel('Error');
legend('Location','best'); axis tight;

% Right: noisy comparisons
nexttile; hold on; grid on;
%plot(t, EpN,   'LineWidth',1.8, 'DisplayName','Projection error (noisy basis)');
plot(t, EmPG,  '-.', 'LineWidth',2.2, 'DisplayName','Model error (PG; test from separate data)');
plot(t, EmPGS, ':',  'LineWidth',2.6, 'DisplayName','Model error (PG + stability shift)');

title('Noisy trial modes + PG test space from separate data','FontWeight','bold');
xlabel('Time'); ylabel('Error');
legend('Location','best'); axis tight;

annotation('textbox',[0.76 0.18 0.23 0.70], ...
  'String',sprintf(['r = %d\n\\sigma = 10^{-3}\nN = %d\nseedNoise = %d\n' ...
                    'seedTestBase = %d\nnICtest = %d'], ...
                    r, N, seedNoise, seedTestBase, nICtest), ...
  'FitBoxToText','on','EdgeColor','none','FontSize',14);


%%
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

% Galerkin coefficients: a = Ur' W2 q
function Em = model_err_galerkin(Qf, Ur, W2, Pr, k0)
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

% Petrov coeffs: a = Wr' W2 q, reconstruction uses Vr
function Em = model_err_petrov(Qf, Vr, Wr, W2, Pr, k0)
K = size(Qf,2);
wn = @(v) sqrt(real(v'*(W2*v)));
Em = zeros(K,1);
a = Wr' * W2 * Qf(:,k0);
for k = 1:K
    qk = Qf(:,k);
    Em(k) = wn(qk - Vr*a) / max(wn(qk), eps);
    a = Pr*a;
end
end


function Qhat = simulate_rom_galerkin(Ur, W2, Pr, Qf, k0, K)
% qhat_k = Ur a_k,   a_{k+1} = Pr a_k,   a0 = Ur' W2 q_k0
n = size(Ur,1);
Qhat = zeros(n,K);
a = Ur' * W2 * Qf(:,k0);
for k=1:K
    Qhat(:,k) = Ur*a;
    a = Pr*a;
end
end

function Qhat = simulate_rom_petrov(Vr, Wr, W2, Pr, Qf, k0, K)
% qhat_k = Vr a_k,   a_{k+1} = Pr a_k,   a0 = Wr' W2 q_k0
n = size(Vr,1);
Qhat = zeros(n,K);
a = Wr' * W2 * Qf(:,k0);
for k=1:K
    Qhat(:,k) = Vr*a;
    a = Pr*a;
end
end

