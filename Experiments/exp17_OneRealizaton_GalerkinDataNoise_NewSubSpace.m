%% exp18_DATA_NOISE_PG_with_Unew.m
% Noise is added to DATA snapshots Qf (in WEIGHTED space),
% then POD basis is built from the noisy snapshots.
%
% OLD story:
%   - clean ROM
%   - noisy-data ROM
%   - old decomposition A1/A2/A3
%
% NEW story:
%   - build Unew from clean data with bad tail directions removed
%   - use Unew as TEST basis
%   - use Ur0  as TRIAL basis
%   - build Petrov-Galerkin ROM:
%         Mr_pg = Unew' * W2 * Ur0
%         Kr_pg = Unew' * W2 * L * Ur0
%         LrPG  = Mr_pg \ Kr_pg
%
% Rollout still uses Ur0 as the trial/reconstruction basis.

clear; close all; clc;
addpath(genpath('../1src'));

%% -------------------- SETTINGS --------------------
Re=1000; N=128; alpha=1; beta=0; meanInd=2;
dt=1e-2; K=15000;
r=8; k0=1;

seed  = 3001;
idx = [];            % empty -> use mid index

% ---- build Unew from dangerous tail directions only
tail_idx = 7:8;      % use old tail columns to define bad directions
pbad     = 2;        % number of bad directions to deflate

%% -------------------- BUILD OS + GENERATOR --------------------
[A,M,y] = OS_Op(alpha,beta,N,Re,meanInd);
n = size(A,1);
if isempty(idx), idx = round(n/2); end

L   = -1i*(M\A);      % full generator
Phi = expm(dt*L);     % full propagator

% Chebyshev quadrature weights (interior)
w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

%% -------------------- FOM TRUTH SNAPSHOTS --------------------
q0 = randn(n,1);
q0 = q0 / norm(q0);

Qf = zeros(n,K);
q = q0;
for k = 1:K
    q = Phi*q;
    Qf(:,k) = q;
end
t = (0:K-1)';

%% ============================================================
%  DATA NOISE: Qf -> QfN (noise added in weighted space)
% ============================================================
rng(seed);

Xw = W * Qf;
sigma_rel = 5e-5;
sigma = sigma_rel * std(Xw(:));
fprintf('SIGMA = %.3e\n', sigma);

XwN = Xw + sigma * randn(size(Xw));
QfN = W \ XwN;

fprintf('DATA NOISE: ||W(QfN-Qf)||F / ||WQf||F = %.3e\n', ...
    norm(W*(QfN-Qf),'fro') / max(norm(W*Qf,'fro'), eps));

%% -------------------- CLEAN ROM --------------------
rom0 = rom_build_clean(Qf, W, W2, L, r, alpha);
Ur0  = rom0.Ur;
S    = rom0.s;

chk0 = norm(Ur0' * W2 * Ur0 - eye(r), 'fro');
fprintf('orth(clean) = %.2e\n', chk0);

Lr0 = Ur0' * W2 * L * Ur0;
Pr0 = expm(dt * Lr0);

figure('Color','w');
semilogy(S,'o-','LineWidth',1.5);
xlabel('Mode index');
ylabel('Singular value');
title('POD singular values');
grid on;

%% -------------------- NOISY-DATA ROM --------------------
romN = rom_build_clean(QfN, W, W2, L, r, alpha);
UrN  = romN.Ur;
sing_N = romN.s;

chkN = norm(UrN' * W2 * UrN - eye(r), 'fro');
fprintf('orth(noisy-data basis) = %.2e\n', chkN);

LrN = UrN' * W2 * L * UrN;
PrN = expm(dt * LrN);

%% ============================================================
%  ALIGNMENT + OLD DECOMPOSITION
% ============================================================

% ---- Align noisy basis to clean basis
G = Ur0' * W2 * UrN;
[Urot,~,Vrot] = svd(G,'econ');
R = Urot * Vrot';

UrNa = UrN * R';
chkA = norm(UrNa' * W2 * UrNa - eye(r), 'fro');
fprintf('orth(aligned noisy-data) = %.2e\n', chkA);

% ---- Rotate noisy reduced generator
LrNa = R * LrN * R';

% ---- Old perturbation
Neff = UrNa - Ur0;

fprintf('||Neff||_F        = %.3e\n', norm(Neff,'fro'));
fprintf('||Neff||_F/sqrt(r)= %.3e\n', norm(Neff,'fro')/sqrt(r));
fprintf('max |Neff(:)|     = %.3e\n', max(abs(Neff(:))));

% ---- Old decomposition
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

fprintf('\nRELATIVE to ||A0||F:\n');
fprintf('||A1||/||A0||    = %.3e\n', n1/max(n0,eps));
fprintf('||A2||/||A0||    = %.3e\n', n2/max(n0,eps));
fprintf('||A1+A2||/||A0|| = %.3e\n', n12/max(n0,eps));
fprintf('||A3||/||A0||    = %.3e\n', n3/max(n0,eps));

% ---- Old "remove-term" ROMs
Lr_rmA1   = LrNa - A1;
Lr_rmA2   = LrNa - A2;
Lr_rmLIN  = LrNa - (A1 + A2);
Lr_rmQUAD = LrNa - A3;

PrNa      = expm(dt * LrNa);
Pr_rmA1   = expm(dt * Lr_rmA1);
Pr_rmA2   = expm(dt * Lr_rmA2);
Pr_rmLIN  = expm(dt * Lr_rmLIN);
Pr_rmQUAD = expm(dt * Lr_rmQUAD);

%% ============================================================
%  BUILD Unew FROM CLEAN DATA WITH BAD TAIL DIRECTIONS REMOVED
% ============================================================

% Dangerous directions from OLD perturbation mechanism, tail only
Bbad = L * Neff(:,tail_idx);

Bw = W * Bbad;
[Ubw,~,~] = svd(Bw,'econ');

pbad_eff = min([pbad, size(Ubw,2)]);
Vbad = W \ Ubw(:,1:pbad_eff);
Vbad = W2_orthonormalize(Vbad, W);

% Remove bad directions from FULL nOISY data
Qclean = QfN - Vbad * (Vbad' * W2 * QfN);

fprintf('\nUnew construction checks:\n');
fprintf('||Vbad'' W2 Qclean||F = %.3e\n', norm(Vbad' * W2 * Qclean,'fro'));

% Build completely new POD basis from cleaned data
rom_new = rom_build_clean(Qclean, W, W2, L, r, alpha);
Unew = rom_new.Ur;

chkUnew = norm(Unew' * W2 * Unew - eye(r), 'fro');
fprintf('orth(Unew) = %.2e\n', chkUnew);

% Compare old contamination seen by old vs new test basis
Cold = Ur0'  * W2 * (L * Neff);
Cnew = Unew' * W2 * (L * Neff);

fprintf('\nOld contamination seen by old vs new test basis:\n');
fprintf('||Cold||F = %.6e\n', norm(Cold,'fro'));
fprintf('||Cnew||F = %.6e\n', norm(Cnew,'fro'));
fprintf('ratio     = %.6e\n', norm(Cnew,'fro')/max(norm(Cold,'fro'),eps));
fprintf('max|Cold| = %.6e\n', max(abs(Cold(:))));
fprintf('max|Cnew| = %.6e\n', max(abs(Cnew(:))));

%% ============================================================
%  PETROV-GALERKIN ROM: trial = Ur0, test = Unew
% ============================================================

Mr_pg = Unew' * W2 * UrN;
Kr_pg = Unew' * W2 * L   * UrN;

condMr = cond(Mr_pg);
fprintf('\nPG mass matrix cond(Unew'' W2 Ur0) = %.6e\n', condMr);

LrPG = Mr_pg \ Kr_pg;
PrPG = expm(dt * LrPG);

%% -------------------- TRACE–I CORRECTION (OLD noisy basis only) --------------------
sig_r = diag(sing_N(1:r));
sig_inv2 = diag(1 ./ (sig_r).^2);
trA  = trace(L);
bias = (sigma^2) * trA * sig_inv2;
LrC  = LrNa - bias;
PrC  = expm(dt * LrC);

fprintf('\ntrace(L)=%.6g%+.6gi, |bias|=%.3e\n', ...
    real(trA), imag(trA), norm(bias,'fro'));

%% ============================================================
%  LEAST STABLE EIGENVALUE INFO
% ============================================================

omF = eig(A,M);
[~,iF] = max(imag(omF));
om_ls = omF(iF);
fprintf('\nFOM least-stable omega: %.6g%+.6gi  (imag=%.6g)\n', ...
    real(om_ls), imag(om_ls), imag(om_ls));

report_least_stable('ROM clean    Lr0',   Lr0,  alpha);
report_least_stable('ROM noisyD   LrNa',  LrNa, alpha);
report_least_stable('ROM corr     LrC',   LrC,  alpha);
report_least_stable('ROM PG(Unew) LrPG',  LrPG, alpha);
report_least_stable('ROM noisyD-A1',      Lr_rmA1, alpha);
report_least_stable('ROM noisyD-A2',      Lr_rmA2, alpha);
report_least_stable('ROM noisyD-LIN',     Lr_rmLIN, alpha);
report_least_stable('ROM noisyD-QUAD',    Lr_rmQUAD, alpha);

%% ============================================================
%  ROLLOUTS
% ============================================================

% ---- OLD decomposition story
Qr0       = roll_Pr(Pr0,       Ur0,  W2, QfN, k0);
QrN       = roll_Pr(PrNa,      UrNa, W2, QfN, k0);
QrC       = roll_Pr(PrC,       UrNa, W2, QfN, k0);
Qr_rmA1   = roll_Pr(Pr_rmA1,   UrNa, W2, QfN, k0);
Qr_rmA2   = roll_Pr(Pr_rmA2,   UrNa, W2, QfN, k0);
Qr_rmLIN  = roll_Pr(Pr_rmLIN,  UrNa, W2, QfN, k0);
Qr_rmQUAD = roll_Pr(Pr_rmQUAD, UrNa, W2, QfN, k0);

% ---- NEW PG story: rollout uses TRIAL basis Ur0
QrPG      = roll_Pr(PrPG,      UrN,  W2, QfN, k0);

%% ============================================================
%  ERRORS
% ============================================================

% Projection errors
Ep0  = proj_err(Qf, Ur0,  W2, k0);
EpN  = proj_err(Qf, UrNa, W2, k0);

% For PG, trial space is Ur0, so projection space is still Ur0
% If you want the approximation-space projection error, compare to Ur0.
EpPG = Ep0;

% Model errors
[Em0, rho0]       = model_err(Qf, Ur0,  W2, Pr0,       k0);
[EmN, rhoN]       = model_err(Qf, UrNa, W2, PrNa,      k0);
[EmC, rhoC]       = model_err(Qf, UrNa, W2, PrC,       k0);
[EmPG, rhoPG]     = model_err(Qf, UrNa,  W2, PrPG,      k0);

[Em_rmA1,   rho_rmA1]   = model_err(Qf, UrNa, W2, Pr_rmA1,   k0);
[Em_rmA2,   rho_rmA2]   = model_err(Qf, UrNa, W2, Pr_rmA2,   k0);
[Em_rmLIN,  rho_rmLIN]  = model_err(Qf, UrNa, W2, Pr_rmLIN,  k0);
[Em_rmQUAD, rho_rmQUAD] = model_err(Qf, UrNa, W2, Pr_rmQUAD, k0);

fprintf('\nrho(Pr): clean=%.4g noisyD=%.4g corr=%.4g PG=%.4g rmA1=%.4g rmA2=%.4g rmLIN=%.4g rmQUAD=%.4g\n', ...
    rho0, rhoN, rhoC, rhoPG, rho_rmA1, rho_rmA2, rho_rmLIN, rho_rmQUAD);

%% ============================================================
%  EXTRACT SINGLE-POINT TRAJECTORIES
% ============================================================

qf    = Qf(idx,:).';
q0i   = Qr0(idx,:).';
qNi   = QrN(idx,:).';
qCi   = QrC(idx,:).';
qPGi  = QrPG(idx,:).';

qA1i = Qr_rmA1(idx,:).';
qA2i = Qr_rmA2(idx,:).';
qLNi = Qr_rmLIN(idx,:).';
qQDi = Qr_rmQUAD(idx,:).';

den_i = max(abs(qf(1)), 1e-14);
rel = @(x) abs(x - qf) ./ den_i;

e0i  = rel(q0i);
eNi  = rel(qNi);
eCi  = rel(qCi);
ePGi = rel(qPGi);
eA1i = rel(qA1i);
eA2i = rel(qA2i);
eLNi = rel(qLNi);
eQDi = rel(qQDi);

%% ============================================================
%  STORY A: OLD DECOMPOSITION ONLY
% ============================================================

figure('Color','w');
hold on; grid on;
plot(t, real(qf),  'k-',  'LineWidth',1.8, 'DisplayName','FOM');
plot(t, real(q0i), 'b-',  'LineWidth',1.8, 'DisplayName','ROM clean');
plot(t, real(qNi), 'r-',  'LineWidth',1.8, 'DisplayName','ROM noisy');
plot(t, real(qCi), 'g--', 'LineWidth',1.8, 'DisplayName','ROM corrected');
plot(t, real(qA1i),'m--', 'LineWidth',1.6, 'DisplayName','ROM (noisy - A1)');
plot(t, real(qA2i),'c--', 'LineWidth',1.6, 'DisplayName','ROM (noisy - A2)');
plot(t, real(qLNi),'k--', 'LineWidth',1.6, 'DisplayName','ROM (noisy - (A1+A2))');
plot(t, real(qQDi),'y--', 'LineWidth',1.6, 'DisplayName','ROM (noisy - A3)');
xlabel('time');
ylabel(sprintf('Re(q(idx=%d))', idx));
title('decomposition : A1/A2/A3 removal on noisy ROM');
legend('Location','best');

figure('Color','w');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile; hold on; grid on;
plot(t, max(e0i,1e-16), 'b-',  'LineWidth',1.6, 'DisplayName','clean');
plot(t, max(eNi,1e-16), 'r-',  'LineWidth',1.6, 'DisplayName','noisy');
plot(t, max(eCi,1e-16), 'g--', 'LineWidth',1.6, 'DisplayName','corrected');
xlabel('time'); ylabel('rel error');
title('Clean vs noisy vs corrected');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','clean');
plot(t, max(eA1i,1e-16), 'm--', 'LineWidth',1.6, 'DisplayName','noisy - A1');
plot(t, max(eA2i,1e-16), 'c--', 'LineWidth',1.6, 'DisplayName','noisy - A2');
xlabel('time'); ylabel('rel error');
title('Effect of removing A1 or A2');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','clean');
plot(t, max(eLNi,1e-16), 'k--', 'LineWidth',1.6, 'DisplayName','noisy - (A1+A2)');
plot(t, max(eQDi,1e-16), 'y--', 'LineWidth',1.6, 'DisplayName','noisy - A3');
xlabel('time'); ylabel('rel error');
title('Removing linear pair vs quadratic term');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(Em0,1e-16),      'b-',  'LineWidth',1.6, 'DisplayName','Model err clean');
plot(t, max(EmN,1e-16),      'r-',  'LineWidth',1.6, 'DisplayName','Model err noisy');
plot(t, max(EmC,1e-16),      'g--', 'LineWidth',1.6, 'DisplayName','Model err corrected');
plot(t, max(Em_rmA1,1e-16),  'm--', 'LineWidth',1.4, 'DisplayName','Model err noisy-A1');
plot(t, max(Em_rmA2,1e-16),  'c--', 'LineWidth',1.4, 'DisplayName','Model err noisy-A2');
xlabel('time'); ylabel('model error');
title('decomposition model errors');
legend('Location','best');
set(gca,'YScale','log');

%% ============================================================
%  STORY B: NEW PG ROM ONLY
% ============================================================

figure('Color','w');
hold on; grid on;
plot(t, real(qf),  'k-',  'LineWidth',2.0, 'DisplayName','FOM');
plot(t, real(q0i), 'b-',  'LineWidth',1.8, 'DisplayName','ROM clean');
plot(t, real(qNi), 'r--', 'LineWidth',1.8, 'DisplayName','ROM noisy');
plot(t, real(qPGi),'m-.', 'LineWidth',1.8, 'DisplayName','ROM New(test=Unew, trial=Ur0)');
xlabel('time');
ylabel(sprintf('Re(q(idx=%d))', idx));
title(' FOM vs clean vs noisy vs ROM with opt test space');
legend('Location','best');

figure('Color','w');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile; hold on; grid on;
plot(t, max(Ep0,1e-16), 'b-',  'LineWidth',1.8, 'DisplayName','Proj err clean(trial=Ur0)');
plot(t, max(EpN,1e-16), 'r--', 'LineWidth',1.8, 'DisplayName','Proj err noisy');
plot(t, max(EpPG,1e-16),'m-.', 'LineWidth',1.8, 'DisplayName','Proj err ROM New(trial=Ur0)');
xlabel('time'); ylabel('projection error');
title('Projection error');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(Em0,1e-16), 'b-',  'LineWidth',1.8, 'DisplayName','Model err clean');
plot(t, max(EmN,1e-16), 'r--', 'LineWidth',1.8, 'DisplayName','Model err noisy');
plot(t, max(EmPG,1e-16),'m-.', 'LineWidth',1.8, 'DisplayName','Model err ROM New');
xlabel('time'); ylabel('model error');
title('Model error');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(e0i,1e-16), 'b-',  'LineWidth',1.6, 'DisplayName','clean');
plot(t, max(eNi,1e-16), 'r--', 'LineWidth',1.6, 'DisplayName','noisy');
plot(t, max(ePGi,1e-16),'m-.', 'LineWidth',1.6, 'DisplayName','ROM New');
xlabel('time'); ylabel('pointwise rel error');
title('Pointwise error at selected index');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, real(q0i), 'b-',  'LineWidth',1.8, 'DisplayName','ROM clean');
plot(t, real(qNi), 'r--', 'LineWidth',1.8, 'DisplayName','ROM noisy');
plot(t, real(qPGi),'m-.', 'LineWidth',1.8, 'DisplayName','ROM New');
xlabel('time');
ylabel(sprintf('Re(q(idx=%d))', idx));
title('ROM-only comparison');
legend('Location','best');

%% ============================================================
%  OLD TERM MAGNITUDES
% ============================================================

figure('Color','w');
vals   = [n1 n2 n12 n3];
labels = {'||A1||','||A2||','||A1+A2||','||A3||'};
bar(vals);
set(gca,'XTick',1:numel(vals),'XTickLabel',labels);
ylabel('Frobenius norm');
title(sprintf('decomposition term magnitudes (sigma=%.0e)', sigma));
grid on;
set(gca,'YScale','log');

%% ============================================================
%  A2 GEOMETRY (OLD)
% ============================================================

X = W * Ur0;
Xnew = W* Unew;
Y = W * (L * Neff);
A2hat = X' * Y;

fprintf('\nA2hat diagnostics (raw, OLD):\n');
fprintf('  ||X||F      = %.6e\n', norm(X,'fro'));
fprintf('  ||Y||F      = %.6e\n', norm(Y,'fro'));
fprintf('  ||A2hat||F  = %.6e\n', norm(A2hat,'fro'));
fprintf('  ||A2hat||F/(||X||F||Y||F) = %.6e\n', ...
    norm(A2hat,'fro')/max(norm(X,'fro')*norm(Y,'fro'), eps));

figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
imagesc(abs(X)); axis tight; colorbar;
title('|W * Ur0|');
xlabel('mode'); ylabel('state index');

nexttile;
imagesc(abs(Y)); axis tight; colorbar;
title('|W * (L * Neff_{old})|');
xlabel('noisy column'); ylabel('state index');

figure('Color','w');
imagesc(abs(A2hat)); axis tight; colorbar;
title('|A2_{old}| = |(W*Ur0)^H (W*(L*Neff_{old}))|');
xlabel('noisy column'); ylabel('clean mode');
%%
figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile;
imagesc(abs(Xnew)); axis tight; colorbar;
title('|W * Urnew|');
xlabel('mode'); ylabel('state index');
%% ============================================================
%  OLD contamination seen by old vs new test space
% ============================================================

figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
imagesc(abs(Cold)); axis tight; colorbar;
title('|Ur0'' W2 (L Neff_{old})|');
xlabel('noisy column'); ylabel('clean mode');

nexttile;
imagesc(abs(Cnew)); axis tight; colorbar;
title('|Unew'' W2 (L Neff_{old})|');
xlabel('noisy column'); ylabel('new test mode');

%% ============================================================
%  PRINCIPAL ANGLES: clean vs noisy-data basis
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

fprintf('max principal angle = %.3e rad\n', max(theta));
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
den = max( wn(Qf(:,k0)), eps );

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
den = max( wn(Qf(:,k0)), eps );

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