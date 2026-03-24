%% exp11_PetrovGalerkin_vs_Galerkin_ONErun.m
% Compare trajectory errors:
%   - Clean Galerkin ROM
%   - Noisy Galerkin ROM
%   - Noisy Galerkin + stability shift
%   - Noisy Petrov–Galerkin ROM (test basis from adjoint snapshots)
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

N  = 158;
dt = 1e-2;
K  = 15000;
r  = 25;
k0 = 1;

sigma     = 1e-3;   % fixed noise 
seedNoise = 2;

% Adjoint snapshot length 
Kadj = K;

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

%% ---------------- FOM SNAPSHOTS ----------------
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
Uw_r = W*(Uall(:,1:r));

rng(seedNoise);
Uw_rN = Uw_r + sigma*randn(size(Uw_r));
VrN   = W2_orthonormalize(W\Uw_rN, W);     % noisy trial basis V

% Noisy Galerkin reduced generator
LrN = VrN' * W2 * L * VrN;
PrN = expm(dt*LrN);

EpN = proj_err(Qf, VrN, W2);
EmN = model_err_galerkin(Qf, VrN, W2, PrN, k0);

% Noisy Galerkin + stability shift
sN = max(real(eig(LrN)));
LrS = LrN - (max(sN,0) + stab_eps)*eye(r);
PrS = expm(dt*LrS);
EmS = model_err_galerkin(Qf, VrN, W2, PrS, k0);

%% ---------------- BUILD TEST BASIS FROM ADJOINT SNAPSHOTS ----------------
% Adjoint under W2 inner product:
%   <x, Ly>_{W2} = <Ldag x, y>_{W2}
% => Ldag = W2^{-1} L^* W2  (here * = conjugate transpose)
Ldag = (W2 \ (L' * W2));
Phidag = expm(dt*Ldag);

% Generate adjoint snapshots (one simple choice: random initial condition)
rng(7);
p0 = randn(n,1);
p0 = p0 / sqrt(real(p0'*(W2*p0)));

P = zeros(n,Kadj);
p = p0;
for k=1:Kadj
    p = Phidag*p;
    P(:,k) = p;
end

% POD on adjoint snapshots (weighted SVD, same as forward)
Pw = W*P;
[Zw,~,~] = svd(Pw,'econ');
Zall = W\Zw;

Wr_raw = W2_orthonormalize(Zall(:,1:r), W);   % raw test basis (W2-orthonormal)

% Bi-orthonormalize so that Wr' W2 VrN = I
Mbi = Wr_raw' * W2 * VrN;                % r×r
Wr  = Wr_raw / (Mbi');                   % Wr = Wr_raw * inv(Mbi')

% Check (optional)
% disp(norm(Wr'*W2*VrN - eye(r),'fro'))

%% ---------------- PETROV–GALERKIN REDUCED GENERATOR ----------------
LrPG = Wr' * W2 * L * VrN;
PrPG = expm(dt*LrPG);

% Optional: stability shift for PG too (often helpful)
sPG = max(real(eig(LrPG)));
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
plot(t, EpN, 'LineWidth',1.8, 'DisplayName','Projection error (noisy basis)');
%plot(t, EmN, 'LineWidth',1.8, 'DisplayName','Model error (noisy Galerkin)');
%plot(t, EmS, '--', 'LineWidth',2.2, 'DisplayName','Model error (noisy + stability shift)');
plot(t, EmPG,'-.', 'LineWidth',2.2, 'DisplayName','Model error (Petrov–Galerkin)');
plot(t, EmPGS,':', 'LineWidth',2.6, 'DisplayName','Model error (PG + stability shift)');

title('Modes identified from noisy modes','FontWeight','bold');
xlabel('Time'); ylabel('Error');
legend('Location','best'); axis tight;

annotation('textbox',[0.78 0.19 0.2 0.65], ...
  'String',sprintf('r = %d\n\\sigma = 10^{-3}\nN = %d\nseed = %d', r, N, seedNoise), ...
  'FitBoxToText','on','EdgeColor','none','FontSize',16);

%%
%% ---------------- SOLUTION TRACE AT ONE NODE (FOM vs ROM) ----------------
node = round(n/2);   % <-- choose node index here
% node = 10;         % example manual choice

Qhat0   = simulate_rom_galerkin(Ur0, W2, Pr0,   Qf, k0, K);   % clean Galerkin
QhatN   = simulate_rom_galerkin(VrN, W2, PrN,   Qf, k0, K);   % noisy Galerkin
QhatS   = simulate_rom_galerkin(VrN, W2, PrS,   Qf, k0, K);   % noisy + shift
QhatPG  = simulate_rom_petrov (VrN, Wr, W2, PrPG,  Qf, k0, K);% Petrov–Galerkin
QhatPGS = simulate_rom_petrov (VrN, Wr, W2, PrPGS, Qf, k0, K);% PG + shift

qF = Qf(node,:).';
q0 = Qhat0(node,:).';
qN = QhatN(node,:).';
qS = QhatS(node,:).';
qP = QhatPG(node,:).';
qPS= QhatPGS(node,:).';

figure('Color','w','Position',[80 80 1550 520]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% ---- real part trace ----
nexttile; hold on; grid on;
plot(t, real(qF),  'k-',  'LineWidth',1.8, 'DisplayName','FOM');
plot(t, real(q0),  'b--',  'LineWidth',1.4, 'DisplayName','clean Galerkin');
%plot(t, real(qN),  'r-',  'LineWidth',1.4, 'DisplayName','noisy Galerkin');
%plot(t, real(qS),  'm--', 'LineWidth',1.8, 'DisplayName','noisy + shift');
plot(t, real(qP),  'c-.', 'LineWidth',1.8, 'DisplayName','Petrov–Galerkin');
plot(t, real(qPS), 'r--',  'LineWidth',2.2, 'DisplayName','PG + shift');
title(sprintf('Solution trace at node %d (real part)', node),'FontWeight','bold');
xlabel('Time'); ylabel('Re(q)');
legend('Location','best'); axis tight;

% ---- magnitude trace ----
nexttile; hold on; grid on;
plot(t, abs(qF),  'k-',  'LineWidth',1.8, 'DisplayName','FOM');
plot(t, abs(q0),  'b--',  'LineWidth',1.4, 'DisplayName','clean Galerkin');
%plot(t, abs(qN),  'r-',  'LineWidth',1.4, 'DisplayName','noisy Galerkin');
%plot(t, abs(qS),  'm--', 'LineWidth',1.8, 'DisplayName','noisy + shift');
plot(t, abs(qP),  'c-.', 'LineWidth',1.8, 'DisplayName','Petrov–Galerkin');
plot(t, abs(qPS), 'r--',  'LineWidth',2.2, 'DisplayName','PG + shift');
title(sprintf('Solution trace at node %d (magnitude)', node),'FontWeight','bold');
xlabel('Time'); ylabel('|q|');
legend('Location','best'); axis tight;







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
