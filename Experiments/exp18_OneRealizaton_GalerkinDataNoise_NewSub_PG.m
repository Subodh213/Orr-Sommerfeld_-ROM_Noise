%% exp19_COMPARE_MODIFIED_PG_vs_CLASSICAL_PG.m
% Compare:
%   (1) Clean Galerkin
%   (2) Noisy Galerkin
%   (3) Modified PG   : trial = UrNa, test = Unew
%   (4) Classical PG  : trial = Ur0_from_dataset1, test = Utest_from_dataset2
%
% Dataset 2 is generated with a different initial condition.
%
% Rollout always uses the TRIAL basis.

clear; close all; clc;
addpath(genpath('../1src'));

%% -------------------- SETTINGS --------------------
Re=1000; N=128; alpha=1; beta=0; meanInd=2;
dt=1e-2; K=15000;
r=8; k0=1;

seed_data_noise  = 3001;
seed_ic_trial    = 111;
seed_ic_test     = 222;

idx = [];            % empty -> use mid index

% ---- modified PG settings
tail_idx = 7:8;      % dangerous tail columns used to build Unew
pbad     = 2;        % number of bad directions to deflate

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
%  DATASET 1 (trial dataset)
% ============================================================
rng(seed_ic_trial);
q0_1 = randn(n,1);
q0_1 = q0_1 / norm(q0_1);

Qf = zeros(n,K);
q = q0_1;
for k = 1:K
    q = Phi*q;
    Qf(:,k) = q;
end
t = (0:K-1)';

%% ============================================================
%  DATASET 2 (test dataset for classical PG)
% ============================================================
rng(seed_ic_test);
q0_2 = randn(n,1);
q0_2 = q0_2 / norm(q0_2);

Qf2 = zeros(n,K);
q = q0_2;
for k = 1:K
    q = Phi*q;
    Qf2(:,k) = q;
end

%% ============================================================
%  ADD NOISE TO DATASET 1 (for noisy Galerkin / modified PG)
% ============================================================
rng(seed_data_noise);

Xw  = W * Qf;
sigma_rel = 5e-5;
sigma = sigma_rel * std(Xw(:));
fprintf('SIGMA = %.3e\n', sigma);

XwN = Xw + sigma * randn(size(Xw));
QfN = W \ XwN;

fprintf('DATA NOISE: ||W(QfN-Qf)||F / ||WQf||F = %.3e\n', ...
    norm(W*(QfN-Qf),'fro')/max(norm(W*Qf,'fro'),eps));

%% ============================================================
%  CLEAN GALERKIN BASIS FROM DATASET 1
% ============================================================
rom0 = rom_build_clean(Qf, W, W2, L, r, alpha);
Ur0  = rom0.Ur;
S    = rom0.s;

chk0 = norm(Ur0' * W2 * Ur0 - eye(r),'fro');
fprintf('orth(clean Ur0)=%.2e\n', chk0);

Lr0 = Ur0' * W2 * L * Ur0;
Pr0 = expm(dt * Lr0);

figure('Color','w');
semilogy(S,'o-','LineWidth',1.5);
xlabel('Mode index');
ylabel('Singular value');
title('Dataset 1 POD singular values');
grid on;

%% ============================================================
%  NOISY BASIS FROM NOISY DATASET 1
% ============================================================
romN = rom_build_clean(QfN, W, W2, L, r, alpha);
UrN  = romN.Ur;
sing_N = romN.s;

chkN = norm(UrN' * W2 * UrN - eye(r),'fro');
fprintf('orth(noisy-data UrN)=%.2e\n', chkN);

LrN = UrN' * W2 * L * UrN;
PrN = expm(dt * LrN);

%% ============================================================
%  ALIGN NOISY BASIS TO CLEAN BASIS
% ============================================================
G = Ur0' * W2 * UrN;
[Urot,~,Vrot] = svd(G,'econ');
R = Urot * Vrot';

UrNa = UrN * R';
chkA = norm(UrNa' * W2 * UrNa - eye(r),'fro');
fprintf('orth(aligned noisy UrNa)=%.2e\n', chkA);

LrNa = R * LrN * R';
PrNa = expm(dt * LrNa);

Neff = UrNa - Ur0;

fprintf('\nOld perturbation diagnostics:\n');
fprintf('||Neff||_F        = %.3e\n', norm(Neff,'fro'));
fprintf('||Neff||_F/sqrt(r)= %.3e\n', norm(Neff,'fro')/sqrt(r));
fprintf('max |Neff(:)|     = %.3e\n', max(abs(Neff(:))));

%% ============================================================
%  OLD DECOMPOSITION (for reference only)
% ============================================================
A0 = Ur0'   * W2 * L * Ur0;
A1 = Neff'  * W2 * L * Ur0;
A2 = Ur0'   * W2 * L * Neff;
A3 = Neff'  * W2 * L * Neff;

Lr_sum = A0 + A1 + A2 + A3;
fprintf('\nDECOMP CHECK (aligned): ||LrNa - sum(Ak)||F = %.3e\n', ...
    norm(LrNa - Lr_sum,'fro'));

n0  = norm(A0,'fro');
n1  = norm(A1,'fro');
n2  = norm(A2,'fro');
n12 = norm(A1+A2,'fro');
n3  = norm(A3,'fro');

fprintf('\nOLD DECOMP TERM MAGNITUDES:\n');
fprintf('||A0||F    = %.3e\n', n0);
fprintf('||A1||F    = %.3e\n', n1);
fprintf('||A2||F    = %.3e\n', n2);
fprintf('||A1+A2||F = %.3e\n', n12);
fprintf('||A3||F    = %.3e\n', n3);

%% ============================================================
%  BUILD Unew (contamination-aware test basis) FROM DATASET 1
% ============================================================
Bbad = L * Neff(:,tail_idx);

Bw = W * Bbad;
[Ubw,~,~] = svd(Bw,'econ');

pbad_eff = min([pbad, size(Ubw,2)]);
Vbad = W \ Ubw(:,1:pbad_eff);
Vbad = W2_orthonormalize(Vbad, W);

Qclean = Qf - Vbad * (Vbad' * W2 * Qf);

fprintf('\nModified PG test-basis construction:\n');
fprintf('||Vbad'' W2 Qclean||F = %.3e\n', norm(Vbad' * W2 * Qclean,'fro'));

rom_new = rom_build_clean(Qclean, W, W2, L, r, alpha);
Unew = rom_new.Ur;

chkUnew = norm(Unew' * W2 * Unew - eye(r),'fro');
fprintf('orth(Unew)=%.2e\n', chkUnew);

Cold = Ur0'  * W2 * (L * Neff);
Cnew = Unew' * W2 * (L * Neff);

fprintf('\nOld contamination seen by old vs modified test basis:\n');
fprintf('||Cold||F = %.6e\n', norm(Cold,'fro'));
fprintf('||Cnew||F = %.6e\n', norm(Cnew,'fro'));
fprintf('ratio     = %.6e\n', norm(Cnew,'fro')/max(norm(Cold,'fro'),eps));

%% ============================================================
%  CLASSICAL PG TEST BASIS FROM DATASET 2
% ============================================================
rom2 = rom_build_clean(Qf2, W, W2, L, r, alpha);
Utest_classical = rom2.Ur;

chk2 = norm(Utest_classical' * W2 * Utest_classical - eye(r),'fro');
fprintf('\north(classical PG test basis from dataset 2)=%.2e\n', chk2);

%% ============================================================
%  ROM 1: CLEAN GALERKIN
% ============================================================
% Already built:
% Lr0, Pr0
% trial = Ur0

%% ============================================================
%  ROM 2: NOISY GALERKIN
% ============================================================
% Already built:
% LrNa, PrNa
% trial = UrNa

%% ============================================================
%  ROM 3: MODIFIED PG
%  trial = UrNa, test = Unew
% ============================================================
Vtrial_mod = UrNa;
Wtest_mod  = Unew;

M_mod = Wtest_mod' * W2 * Vtrial_mod;
K_mod = Wtest_mod' * W2 * L * Vtrial_mod;
condM_mod = cond(M_mod);
fprintf('\nMODIFIED PG cond(M_mod) = %.6e\n', condM_mod);

LrPG_mod = M_mod \ K_mod;
PrPG_mod = expm(dt * LrPG_mod);

%% ============================================================
%  ROM 4: CLASSICAL PG
%  trial = Ur0, test = basis from dataset 2
% ============================================================
Vtrial_cl = Ur0;
Wtest_cl  = Utest_classical;

M_cl = Wtest_cl' * W2 * Vtrial_cl;
K_cl = Wtest_cl' * W2 * L * Vtrial_cl;
condM_cl = cond(M_cl);
fprintf('CLASSICAL PG cond(M_cl) = %.6e\n', condM_cl);

LrPG_cl = M_cl \ K_cl;
PrPG_cl = expm(dt * LrPG_cl);

%% ============================================================
%  TRACE-I CORRECTION ON OLD NOISY ROM (optional reference)
% ============================================================
sig_r = diag(sing_N(1:r));
sig_inv2 = diag(1 ./ (sig_r).^2);
trA  = trace(L);
bias = (sigma^2) * trA * sig_inv2;
LrC  = LrNa - bias;
PrC  = expm(dt * LrC);

%% ============================================================
%  EIGENVALUE DIAGNOSTICS
% ============================================================
omF = eig(A,M);
[~,iF] = max(imag(omF));
om_ls = omF(iF);
fprintf('\nFOM least-stable omega: %.6g%+.6gi  (imag=%.6g)\n', ...
    real(om_ls), imag(om_ls), imag(om_ls));

report_least_stable('Clean Galerkin    ', Lr0,      alpha);
report_least_stable('Noisy Galerkin    ', LrNa,     alpha);
report_least_stable('Modified PG       ', LrPG_mod, alpha);
report_least_stable('Classical PG      ', LrPG_cl,  alpha);
report_least_stable('Trace-I corrected ', LrC,      alpha);

%% ============================================================
%  ROLLOUTS
% ============================================================

% Clean Galerkin
Qr0 = roll_Pr(Pr0, Ur0, W2, Qf, k0);

% Noisy Galerkin
QrN = roll_Pr(PrNa, UrNa, W2, Qf, k0);

% Modified PG: rollout uses TRIAL basis UrNa
QrPG_mod = roll_Pr(PrPG_mod, Vtrial_mod, W2, Qf, k0);

% Classical PG: rollout uses TRIAL basis Ur0
QrPG_cl = roll_Pr(PrPG_cl, Vtrial_cl, W2, Qf, k0);

% Corrected noisy reference
QrC = roll_Pr(PrC, UrNa, W2, Qf, k0);

%% ============================================================
%  ERRORS
% ============================================================

% Projection errors depend on TRIAL basis
Ep0      = proj_err(Qf, Ur0,       W2, k0);    % clean G
EpN      = proj_err(Qf, UrNa,      W2, k0);    % noisy G
EpPG_mod = proj_err(Qf, Vtrial_mod,W2, k0);    % modified PG
EpPG_cl  = proj_err(Qf, Vtrial_cl, W2, k0);    % classical PG

% Model errors
[Em0,      rho0]      = model_err(Qf, Ur0,        W2, Pr0,      k0);
[EmN,      rhoN]      = model_err(Qf, UrNa,       W2, PrNa,     k0);
[EmPG_mod, rhoPG_mod] = model_err(Qf, Vtrial_mod, W2, PrPG_mod, k0);
[EmPG_cl,  rhoPG_cl]  = model_err(Qf, Vtrial_cl,  W2, PrPG_cl,  k0);
[EmC,      rhoC]      = model_err(Qf, UrNa,       W2, PrC,      k0);

fprintf('\nrho(Pr): cleanG=%.4g noisyG=%.4g modPG=%.4g classPG=%.4g corr=%.4g\n', ...
    rho0, rhoN, rhoPG_mod, rhoPG_cl, rhoC);

%% ============================================================
%  SINGLE-POINT TRAJECTORIES
% ============================================================

qf      = Qf(idx,:).';
q0i     = Qr0(idx,:).';
qNi     = QrN(idx,:).';
qPGmodi = QrPG_mod(idx,:).';
qPGcli  = QrPG_cl(idx,:).';
qCi     = QrC(idx,:).';

den_i = max(abs(qf(1)), 1e-14);
rel = @(x) abs(x - qf) ./ den_i;

e0i     = rel(q0i);
eNi     = rel(qNi);
ePGmodi = rel(qPGmodi);
ePGcli  = rel(qPGcli);
eCi     = rel(qCi);

%% ============================================================
%  MAIN COMPARISON PLOTS
% ============================================================

figure('Color','w');
hold on; grid on;
plot(t, real(qf),      'k-',  'LineWidth',2.0, 'DisplayName','FOM');
plot(t, real(q0i),     'b-',  'LineWidth',1.8, 'DisplayName','Clean Galerkin');
plot(t, real(qNi),     'r--', 'LineWidth',1.8, 'DisplayName','Noisy Galerkin');
plot(t, real(qPGmodi), 'm-.', 'LineWidth',1.8, 'DisplayName','Modified PG');
% plot(t, real(qPGcli),  'c-.', 'LineWidth',1.8, 'DisplayName','Classical PG');
% plot(t, real(qCi),     'g--', 'LineWidth',1.8, 'DisplayName','Trace-I corrected');
xlabel('time');
ylabel(sprintf('Re(q(idx=%d))', idx));
title('Trajectory comparison: Galerkin vs modified PG vs classical PG');
legend('Location','best');
%%
figure('Color','w');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile; hold on; grid on;
plot(t, max(Ep0,1e-16),      'b-',  'LineWidth',1.8, 'DisplayName','Proj err clean G');
plot(t, max(EpN,1e-16),      'r--', 'LineWidth',1.8, 'DisplayName','Proj err noisy G');
plot(t, max(EpPG_mod,1e-16), 'm-.', 'LineWidth',1.8, 'DisplayName','Proj err modified PG');
plot(t, max(EpPG_cl,1e-16),  'c-.', 'LineWidth',1.8, 'DisplayName','Proj err classical PG');
xlabel('time'); ylabel('projection error');
title('Projection error');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(Em0,1e-16),      'b-',  'LineWidth',1.8, 'DisplayName','Model err clean G');
plot(t, max(EmN,1e-16),      'r--', 'LineWidth',1.8, 'DisplayName','Model err noisy G');
plot(t, max(EmPG_mod,1e-16), 'm-.', 'LineWidth',1.8, 'DisplayName','Model err modified PG');
plot(t, max(EmPG_cl,1e-16),  'c-.', 'LineWidth',1.8, 'DisplayName','Model err classical PG');
plot(t, max(EmC,1e-16),      'g--', 'LineWidth',1.8, 'DisplayName','Model err corrected');
xlabel('time'); ylabel('model error');
title('Model error');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),     'b-',  'LineWidth',1.6, 'DisplayName','Clean G');
plot(t, max(eNi,1e-16),     'r--', 'LineWidth',1.6, 'DisplayName','Noisy G');
plot(t, max(ePGmodi,1e-16), 'm-.', 'LineWidth',1.6, 'DisplayName','Modified PG');
plot(t, max(ePGcli,1e-16),  'c-.', 'LineWidth',1.6, 'DisplayName','Classical PG');
plot(t, max(eCi,1e-16),     'g--', 'LineWidth',1.6, 'DisplayName','Corrected');
xlabel('time'); ylabel('pointwise rel error');
title('Pointwise error at selected index');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, real(q0i),     'b-',  'LineWidth',1.8, 'DisplayName','Clean G');
plot(t, real(qNi),     'r--', 'LineWidth',1.8, 'DisplayName','Noisy G');
plot(t, real(qPGmodi), 'm-.', 'LineWidth',1.8, 'DisplayName','Modified PG');
plot(t, real(qPGcli),  'c-.', 'LineWidth',1.8, 'DisplayName','Classical PG');
xlabel('time');
ylabel(sprintf('Re(q(idx=%d))', idx));
title('ROM-only comparison');
legend('Location','best');

%% ============================================================
%  GEOMETRY / DIAGNOSTICS PLOTS
% ============================================================

figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
imagesc(abs(Cold)); axis tight; colorbar;
title('|Ur0'' W2 (L Neff_{old})|');
xlabel('old noisy column'); ylabel('clean mode');

nexttile;
imagesc(abs(Cnew)); axis tight; colorbar;
title('|Unew'' W2 (L Neff_{old})|');
xlabel('old noisy column'); ylabel('modified PG test mode');

%% ============================================================
%  MASS MATRIX DIAGNOSTICS
% ============================================================

fprintf('\nMass / biorthogonality diagnostics:\n');
fprintf('||Unew'' W2 UrNa - I||F = %.3e\n', norm(M_mod - eye(r),'fro'));
fprintf('||Utest_classical'' W2 Ur0 - I||F = %.3e\n', norm(M_cl - eye(r),'fro'));

figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
imagesc(abs(M_mod)); axis tight; colorbar;
title('|M_{mod}| = |Unew'' W2 UrNa|');
xlabel('trial mode'); ylabel('test mode');

nexttile;
imagesc(abs(M_cl)); axis tight; colorbar;
title('|M_{class}| = |Utest\_class'' W2 Ur0|');
xlabel('trial mode'); ylabel('test mode');

%% ============================================================
%  PRINCIPAL ANGLES
% ============================================================

Msub = Ur0' * W2 * UrN;
s = svd(Msub);
s = min(max(real(s),-1),1);
theta = acos(s);

figure('Color','w');
semilogy(1:length(theta), theta, 'o-','LineWidth',1.5);
grid on;
xlabel('mode index');
ylabel('principal angle (rad)');
title('Principal angles: clean vs noisy-data subspaces');

fprintf('\nPrincipal angles clean vs noisy:\n');
fprintf('max principal angle  = %.3e rad\n', max(theta));
fprintf('mean principal angle = %.3e rad\n', mean(theta));

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

fprintf('%-18s: lam=%.6g%+.6gi  (Re=%.6g)   c=%.6g%+.6gi\n', ...
    name, real(lam_ls), imag(lam_ls), real(lam_ls), real(c_ls), imag(c_ls));
end