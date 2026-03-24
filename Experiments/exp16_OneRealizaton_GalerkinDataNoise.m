%% exp04_all_plots_DECOMP_ALIGN_REMOVE_FULL_dataNoise.m
% FOM vs ROM clean vs ROM from noisy DATA
% PLUS (one realization, GAUGE-FIXED):
%   Align noisy-data basis to clean basis (W2-Procrustes),
%   decompose LrNa = A0 + A1 + A2 + A3,
%   test "remove-term" what-if ROMs:
%      LrNa - A1,  LrNa - A2,  LrNa - (A1+A2),  LrNa - A3
%
% IMPORTANT CHANGE:
%   Noise is added to SNAPSHOTS, not directly to modes.
%
% Plots:
%   (1) trajectory at idx + relative errors
%   (2) projection error
%   (3) model error
%   (4) bar plot of term magnitudes: ||A1||,||A2||,||A1+A2||,||A3||
%
% Removed: eigenvector/mode-shape plots.
% Included: least stable eigenvalue info (FOM + ROM variants)

clear; close all; clc;
addpath(genpath('../1src'));

%% -------------------- SETTINGS --------------------
Re=1000; N=128; alpha=1; beta=0; meanInd=2;
dt=1e-2; K=15000;
r=10; k0=1;

sigma = 5e-12;        % std of noise in WEIGHTED SNAPSHOT space
seed  = 3001;

% which interior state index to plot
idx = [];
if isempty(idx), idx = round((N)/2); end

%% -------------------- BUILD OS + GENERATOR --------------------
[A,M,y] = OS_Op(alpha,beta,N,Re,meanInd);
n = size(A,1);
if isempty(idx), idx = round(n/2); end

L   = -1i*(M\A);        % full generator
Phi = expm(dt*L);       % full step propagator

% Chebyshev quadrature weights (interior)
w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

%% -------------------- FOM TRUTH SNAPSHOTS --------------------
rng(1);
q0 = randn(n,1);
q0 = q0 / norm(q0);

Qf = zeros(n,K);
q = q0;
for k = 1:K
    q = Phi*q;
    Qf(:,k) = q;
end
t = (0:K-1)' * dt;

%% -------------------- CLEAN ROM (from clean data) --------------------
rom0 = rom_build_clean(Qf, W, W2, L, r, alpha);

Ur0  = rom0.Ur;     % clean basis
Uall0 = rom0.U;     % all clean POD modes
Lr0  = Ur0' * W2 * L * Ur0;
Pr0  = expm(dt * Lr0);

chk0 = norm(Ur0' * W2 * Ur0 - eye(r), 'fro');
fprintf('orth(clean)=%.2e\n', chk0);

%% ============================================================
%  NOISY DATA: add noise to snapshots, then rebuild POD basis
% ============================================================
rng(seed);

Qfw  = W * Qf;                              % weighted clean snapshots
QfwN = Qfw + sigma * randn(size(Qfw));      % i.i.d. noise in weighted snapshot space
QfN  = W \ QfwN;                            % unweighted noisy snapshots

% Build ROM from NOISY DATA
romN = rom_build_clean(QfN, W, W2, L, r, alpha);

UrN   = romN.Ur;       % POD basis extracted from noisy data
UallN = romN.U;        % all POD modes extracted from noisy data
LrN   = UrN' * W2 * L * UrN;     % noisy-data reduced generator
PrN   = expm(dt * LrN);

chkN = norm(UrN' * W2 * UrN - eye(r), 'fro');
fprintf('orth(noisy-data basis)=%.2e\n', chkN);

%% ============================================================
%  ALIGNMENT (W2-Procrustes) + DECOMP + REMOVALS
% ============================================================

% Align noisy-data basis to clean basis
G = Ur0' * W2 * UrN;
[U,S,V] = svd(G,'econ');
R = U*V';

%UrNa = UrN * R';                 % aligned noisy-data basis
UrNa = UrN ;
chkA = norm(UrNa' * W2 * UrNa - eye(r), 'fro');
fprintf('orth(aligned noisy-data basis)=%.2e\n', chkA);

% Rotate reduced operator into aligned coordinates
LrNa = R * LrN * R';

% Effective basis perturbation caused by noisy data
Neff = UrNa - Ur0;

%% -------------------- DECOMPOSE LrNa = A0 + A1 + A2 + A3 --------------------
A0 = Ur0'  * W2 * L * Ur0;       % clean
A1 = Neff' * W2 * L * Ur0;       % linear #1
A2 = Ur0'  * W2 * L * Neff;      % linear #2
A3 = Neff' * W2 * L * Neff;      % quadratic

Lr_sum = A0 + A1 + A2 + A3;

fprintf('\nDECOMP CHECK (aligned): ||LrNa - sum(Ak)||F = %.3e\n', ...
    norm(LrNa - Lr_sum,'fro'));
fprintf('DECOMP CHECK (raw noisy): ||LrN - R''*sum(Ak)*R||F = %.3e\n', ...
    norm(LrN - (R' * Lr_sum * R),'fro'));

%% -------------------- TERM MAGNITUDES --------------------
n0  = norm(A0,'fro');
n1  = norm(A1,'fro');
n2  = norm(A2,'fro');
n12 = norm(A1+A2,'fro');
n3  = norm(A3,'fro');

fprintf('\nTERM MAGNITUDES (noise in data -> basis error):\n');
fprintf('||Neff||F  = %.3e\n', norm(Neff,'fro'));
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

%% -------------------- REMOVE-TERM OPERATORS --------------------
Lr_rmA1   = LrNa - A1;
Lr_rmA2   = LrNa - A2;
Lr_rmLIN  = LrNa - (A1 + A2);
Lr_rmQUAD = LrNa - A3;

PrNa      = expm(dt * LrNa);
Pr_rmA1   = expm(dt * Lr_rmA1);
Pr_rmA2   = expm(dt * Lr_rmA2);
Pr_rmLIN  = expm(dt * Lr_rmLIN);
Pr_rmQUAD = expm(dt * Lr_rmQUAD);

%% -------------------- OPTIONAL HEURISTIC TRACE CORRECTION --------------------
% NOTE:
% This same trace-I correction was originally motivated for direct basis noise.
% For noise added to DATA, it is no longer a clean derivation.
% Kept here only as a heuristic comparison.
trA  = trace(L);
Sigma = romN.S;   % singular values from SVD of weighted snapshots
Sigma_r = Sigma(1:r);

bias = (sigma^2) * trA * diag(1 ./ (Sigma_r.^2));
%bias = (sigma^2) * trA * eye(r);
LrC  = LrNa - bias;
PrC  = expm(dt * LrC);

fprintf('\ntrace(L)=%.6g%+.6gi,  |bias|=%.3e\n', real(trA), imag(trA), norm(bias,'fro'));
fprintf('NOTE: trace-I correction here is heuristic for noisy-data POD.\n');

%% ============================================================
%  LEAST STABLE EIGENVALUE INFO
% ============================================================
omF = eig(A,M);
[~,iF] = max(imag(omF));
om_ls = omF(iF);
fprintf('\nFOM least-stable omega: %.6g%+.6gi  (imag=%.6g)\n', ...
    real(om_ls), imag(om_ls), imag(om_ls));

% report_ls('ROM clean     Lr0',  Lr0,      alpha);
% report_ls('ROM noisyData LrNa', LrNa,     alpha);
% report_ls('ROM corr      LrC',  LrC,      alpha);
% report_ls('ROM noisy-A1',       Lr_rmA1,  alpha);
% report_ls('ROM noisy-A2',       Lr_rmA2,  alpha);
% report_ls('ROM noisy-LIN',      Lr_rmLIN, alpha);
% report_ls('ROM noisy-QUAD',     Lr_rmQUAD,alpha);

%% ============================================================
%  ROLLOUTS
% ============================================================
Qr0       = roll_Pr(Pr0,      Ur0,  W2, Qf, k0);   % clean ROM
QrN       = roll_Pr(PrNa,     UrNa, W2, Qf, k0);   % ROM from noisy data
QrC       = roll_Pr(PrC,      UrNa, W2, Qf, k0);   % heuristic corrected

Qr_rmA1   = roll_Pr(Pr_rmA1,  UrNa, W2, Qf, k0);
Qr_rmA2   = roll_Pr(Pr_rmA2,  UrNa, W2, Qf, k0);
Qr_rmLIN  = roll_Pr(Pr_rmLIN, UrNa, W2, Qf, k0);
Qr_rmQUAD = roll_Pr(Pr_rmQUAD,UrNa, W2, Qf, k0);

%% ============================================================
%  ERRORS
% ============================================================
Ep0 = proj_err(Qf, Ur0,  W2, k0);
EpN = proj_err(Qf, UrNa, W2, k0);

[Em0, rho0] = model_err(Qf, Ur0,  W2, Pr0,      k0);
[EmN, rhoN] = model_err(Qf, UrNa, W2, PrNa,     k0);
[EmC, rhoC] = model_err(Qf, UrNa, W2, PrC,      k0);

[Em_rmA1,   rho_rmA1]   = model_err(Qf, UrNa, W2, Pr_rmA1,   k0);
[Em_rmA2,   rho_rmA2]   = model_err(Qf, UrNa, W2, Pr_rmA2,   k0);
[Em_rmLIN,  rho_rmLIN]  = model_err(Qf, UrNa, W2, Pr_rmLIN,  k0);
[Em_rmQUAD, rho_rmQUAD] = model_err(Qf, UrNa, W2, Pr_rmQUAD, k0);

fprintf('\nrho(Pr): clean=%.4g noisyData=%.4g corr=%.4g rmA1=%.4g rmA2=%.4g rmLIN=%.4g rmQUAD=%.4g\n', ...
    rho0, rhoN, rhoC, rho_rmA1, rho_rmA2, rho_rmLIN, rho_rmQUAD);

%% ============================================================
%  PLOT 4: TERM MAGNITUDE COMPARISON
% ============================================================
figure('Color','w');
vals   = [n1 n2 n12 n3];
labels = {'||A1||','||A2||','||A1+A2||','||A3||'};
bar(vals);
set(gca,'XTick',1:numel(vals),'XTickLabel',labels);
ylabel('Frobenius norm');
title(sprintf('Term magnitudes from noisy-data POD (sigma=%.0e)', sigma));
grid on;
set(gca,'YScale','log');

%% ============================================================
%  PLOT 1: POINTWISE RELATIVE ERRORS
% ============================================================
qf  = Qf(idx,:).';

q0i = Qr0(idx,:).';
qNi = QrN(idx,:).';
qCi = QrC(idx,:).';

qA1i = Qr_rmA1(idx,:).';
qA2i = Qr_rmA2(idx,:).';
qLNi = Qr_rmLIN(idx,:).';
qQDi = Qr_rmQUAD(idx,:).';

den_i = max(abs(qf(1)), 1e-14);
rel = @(x) abs(x - qf) ./ den_i;

e0i  = rel(q0i);
eNi  = rel(qNi);
eCi  = rel(qCi);
eA1i = rel(qA1i);
eA2i = rel(qA2i);
eLNi = rel(qLNi);
eQDi = rel(qQDi);

figure('Color','w');
tiledlayout(3,2,'Padding','compact','TileSpacing','compact');

nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','clean');
plot(t, max(eNi,1e-16),  'r-',  'LineWidth',1.6, 'DisplayName','noisy-data aligned');
plot(t, max(eCi,1e-16),  'g--', 'LineWidth',1.6, 'DisplayName','heuristic corrected');
xlabel('time'); ylabel('rel error');
title('Pointwise relative error');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','clean');
plot(t, max(eLNi,1e-16), 'm--', 'LineWidth',1.6, 'DisplayName','noisy - (A1+A2)');
xlabel('time'); ylabel('rel error');
title('Remove linear terms');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','clean');
plot(t, max(eA2i,1e-16), 'c--', 'LineWidth',1.6, 'DisplayName','noisy - A2');
xlabel('time'); ylabel('rel error');
title('Remove A2');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','clean');
plot(t, max(eA1i,1e-16), 'm--', 'LineWidth',1.6, 'DisplayName','noisy - A1');
xlabel('time'); ylabel('rel error');
title('Remove A1');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','clean');
plot(t, max(eQDi,1e-16), 'r--', 'LineWidth',1.6, 'DisplayName','noisy - A3');
xlabel('time'); ylabel('rel error');
title('Remove A3');
legend('Location','best');
set(gca,'YScale','log');

nexttile; hold on; grid on;
plot(t, max(eNi,1e-16),  'r-',  'LineWidth',1.6, 'DisplayName','noisy-data aligned');
plot(t, max(eA1i,1e-16), 'm--', 'LineWidth',1.4, 'DisplayName','-A1');
plot(t, max(eA2i,1e-16), 'c--', 'LineWidth',1.4, 'DisplayName','-A2');
plot(t, max(eLNi,1e-16), 'k--', 'LineWidth',1.4, 'DisplayName','-(A1+A2)');
plot(t, max(eQDi,1e-16), 'g--', 'LineWidth',1.4, 'DisplayName','-A3');
xlabel('time'); ylabel('rel error');
title('Compare removals');
legend('Location','best');
set(gca,'YScale','log');

%% ============================================================
%  PLOT 2: PROJECTION ERROR
% ============================================================
figure('Color','w'); hold on; grid on;
plot(t, max(Ep0,1e-16), 'b-',  'LineWidth',1.8, 'DisplayName','Proj err (clean basis)');
plot(t, max(EpN,1e-16), 'r--', 'LineWidth',1.8, 'DisplayName','Proj err (basis from noisy data)');
plot(t, max(Em0,1e-16), 'g--', 'LineWidth',1.8, 'DisplayName','Model err clean');
xlabel('time'); ylabel('Error');
title('Projection error');
legend('Location','best');
set(gca,'YScale','log');

%% ============================================================
%  PLOT 3: MODEL ERROR
% ============================================================
figure('Color','w'); hold on; grid on;
plot(t, max(Em0,1e-16),      'b-',  'LineWidth',1.8, 'DisplayName','Model err clean');
plot(t, max(EmN,1e-16),      'r-',  'LineWidth',1.8, 'DisplayName','Model err noisy-data');
plot(t, max(EmC,1e-16),      'g--', 'LineWidth',1.8, 'DisplayName','Model err corrected');
plot(t, max(Em_rmA1,1e-16),  'm-',  'LineWidth',1.5, 'DisplayName','Model err (noisy-A1)');
%plot(t, max(Em_rmA2,1e-16),  'c-',  'LineWidth',1.5, 'DisplayName','Model err (noisy-A2)');
%plot(t, max(Em_rmQUAD,1e-16),'k--', 'LineWidth',1.5, 'DisplayName','Model err (noisy-A3)');
xlabel('time'); ylabel('model error');
title('Model error');
legend('Location','best');
set(gca,'YScale','log');

%% ============================================================
%  PROJECTION vs MODEL ERROR FOR NOISY-DATA POD
% ============================================================
figure('Color','w'); hold on; grid on;
plot(t, EpN, 'b-', 'LineWidth',2.0, 'DisplayName','Projection error noisy-data basis');
plot(t, EmN, 'r-', 'LineWidth',2.0, 'DisplayName','Model error noisy-data ROM');
plot(t, EmC, 'g--', 'LineWidth',2.0, 'DisplayName','Model error corrected (heuristic)');
xlabel('Time');
ylabel('Error');
title('Basis identified from noisy data');
legend('Location','best');
set(gca,'YScale','log');

%% ============================================================
%  SIMPLE TRAJECTORY: FOM vs ROM vs NOISY-DATA ROM
% ============================================================
qf  = Qf(idx,:).';
q0i = Qr0(idx,:).';
qNi = QrN(idx,:).';

figure('Color','w'); hold on; grid on;
plot(t, real(qf),  'k-',  'LineWidth',2.0, 'DisplayName','FOM');
plot(t, real(q0i), 'g--', 'LineWidth',1.8, 'DisplayName','ROM clean');
plot(t, real(qNi), 'r--', 'LineWidth',1.5, ...
     'DisplayName',sprintf('ROM from noisy data (\\sigma=%.0e)',sigma));
xlabel('time');
ylabel(sprintf('Re(q(idx=%d))', idx));
title('FOM vs Clean ROM vs ROM from noisy data');
legend('Location','best');

%% ============================================================
%  A2 CHANNEL
% ============================================================
X = W * Ur0;
Y = W * (L * Neff);
A2hat = X' * Y;

fprintf('\nA2hat diagnostics (raw):\n');
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
title('|W * (L * Neff)|');
xlabel('mode'); ylabel('state index');

figure('Color','w');
imagesc(abs(A2hat)); axis tight; colorbar;
title('|A2| = |(W*Ur0)^H (W*(L*Neff))|');
xlabel('mode'); ylabel('mode');

rr = size(X,2);
dotCol = zeros(1,rr);
cosCol = zeros(1,rr);
for j = 1:rr
    xj = X(:,j);
    yj = Y(:,j);
    dotCol(j) = abs(xj' * yj);
    cosCol(j) = abs(xj' * yj) / max(norm(xj)*norm(yj), eps);
end

figure('Color','w');
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

nexttile;
bar(dotCol); grid on;
xlabel('mode'); ylabel('|x_j^H y_j|');
title('A2 channel: dot product magnitude');

nexttile;
bar(cosCol); ylim([0 1]); grid on;
xlabel('mode'); ylabel('cosine');
title('A2 channel: per-mode cosine');

%% ============================================================
%  A1 CHANNEL
% ============================================================
NX = W * Neff;
AU = W * (L * Ur0);
A1hat = NX' * AU;

fprintf('\nA1hat diagnostics (raw):\n');
fprintf('  ||NX||F      = %.6e\n', norm(NX,'fro'));
fprintf('  ||AU||F      = %.6e\n', norm(AU,'fro'));
fprintf('  ||A1hat||F   = %.6e\n', norm(A1hat,'fro'));
fprintf('  ||A1hat||F/(||NX||F||AU||F) = %.6e\n', ...
    norm(A1hat,'fro')/max(norm(NX,'fro')*norm(AU,'fro'), eps));

figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile;
imagesc(abs(NX)); axis tight; colorbar;
title('|W * Neff|');
xlabel('mode'); ylabel('state index');

nexttile;
imagesc(abs(AU)); axis tight; colorbar;
title('|W * (L * Ur0)|');
xlabel('mode'); ylabel('state index');

figure('Color','w');
imagesc(abs(A1hat)); axis tight; colorbar;
title('|A1|');
xlabel('mode'); ylabel('mode');

dotColA1 = zeros(1,rr);
cosColA1 = zeros(1,rr);
for j = 1:rr
    nj  = NX(:,j);
    auj = AU(:,j);
    dotColA1(j) = abs(nj' * auj);
    cosColA1(j) = abs(nj' * auj) / max(norm(nj)*norm(auj), eps);
end

figure('Color','w');
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

nexttile;
bar(dotColA1); grid on;
xlabel('mode'); ylabel('|n_j^H (LUr)_j|');
title('A1 channel: dot product magnitude');

nexttile;
bar(cosColA1); ylim([0 1]); grid on;
xlabel('mode'); ylabel('cosine');
title('A1 channel: per-mode cosine');

diagEnergyA1 = norm(diag(diag(A1hat)),'fro');
fullEnergyA1 = norm(A1hat,'fro');
fprintf('  diag(A1hat) energy / full = %.6e\n', diagEnergyA1/max(fullEnergyA1,eps));

fprintf('||A0||F = %.6e\n', norm(A0,'fro'));
fprintf('max |A1hat_ij| = %.6e\n', max(abs(A1hat(:))));
fprintf('max |A0_ij| = %.6e\n', max(abs(A0(:))));

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