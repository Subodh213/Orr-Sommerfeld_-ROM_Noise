%% exp04_FULL_PG_NOISY_ALIGN_DECOMP.m
% Full script: Clean Galerkin ROM + Clean Petrov–Galerkin ROM
% plus: noisy trial/test (noise in WEIGHTED-mode space), PG rollout + errors,
% and (optional) PG decomposition A0..A3 in aligned coordinates.
%
% Assumes these are on path (as in your setup):
%   OS_Op, clencurt_WeightsB, W2_orthonormalize

clear; close all; clc;
addpath(genpath('../1src'));

%% -------------------- SETTINGS --------------------
Re=1000; N=128; alpha=1; beta=0; meanInd=2;
dt=1e-2; K=15000;
r=8; k0=1;

sigma = 5e-8;        % std of noise in WEIGHTED space
seed  = 3001;

idx = [];            % empty -> mid index

%% -------------------- BUILD OS + GENERATOR --------------------
[A,M,y] = OS_Op(alpha,beta,N,Re,meanInd);
n = size(A,1);
if isempty(idx), idx = round(n/2); end

L   = -1i*(M\A);        % full generator
Phi = expm(dt*L);       % full step propagator

w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

%% -------------------- FOM TRUTH SNAPSHOTS --------------------
rng(1);
q0 = randn(n,1); q0 = q0/norm(q0);

Qf = zeros(n,K);
q = q0;
for k = 1:K
    q = Phi*q;
    Qf(:,k) = q;
end
t = (0:K-1)';

%% -------------------- NEW  DATASET (TEST SPACE snapshots) --------------------
% rng(seed + 1);
% q0t = randn(n,1);
% q0t = q0t / norm(q0t);
% 
% Qtest = zeros(n,K);
% qt = q0t;
% for k = 1:K
%     qt = Phi * qt;
%     Qtest(:,k) = qt;
% end
%% -------------------- TEST DATASET (for TEST SPACE): perturbed copy of Qf --------------------
rng(seed + 1);                     % independent randomness from basis-noise
epsTest = 1e-6;                   % small perturbation level 
Qtest  = Qf + epsTest * randn(size(Qf));
%% -------------------- CLEAN GALERKIN ROM --------------------
romG = rom_build_clean(Qf, W, W2, L, r, alpha);
Ur0  = romG.Ur;          % clean trial basis
Uall = romG.U;           % all POD modes (trial)

fprintf('orth(clean Gal)=%.2e\n', norm(Ur0'*W2*Ur0 - eye(r),'fro'));
Lr0 = Ur0' * W2 * L * Ur0;
Pr0 = expm(dt * Lr0);

%% -------------------- CLEAN PETROV–GALERKIN ROM --------------------
romPG = rom_build_pg(Qf, Qtest, W, W2, L, r, alpha);
Ur_pg = romPG.Ur;         % trial
Vr_pg = romPG.Vr;         % test (biorth to trial)
Utrial = romPG.Utrial;    % all trial POD modes (unweighted)
Vtest  = romPG.Vtest;     % all test  POD modes (unweighted)

fprintf('orthU(PG)=%.2e | bio(PG)=%.2e | cond(M0)=%.2e\n', ...
    romPG.orthErrU, romPG.bioErr, romPG.condM);

Lr_pg = romPG.Lr;
Pr_pg = expm(dt * Lr_pg);

%% -------------------- NOISY GALERKIN TRIAL BASIS (your original block) --------------------
rng(seed);
Uw  = W * Uall;
UwN = Uw + sigma * randn(size(Uw));     % i.i.d. in weighted space
UN  = W \ UwN;

UrN = W2_orthonormalize(UN(:,1:r), W);
fprintf('orth(noisy Gal)=%.2e\n', norm(UrN'*W2*UrN - eye(r),'fro'));

LrN = UrN' * W2 * L * UrN;
PrN = expm(dt * LrN);

% ---- Align noisy Galerkin basis to clean Galerkin basis
G = Ur0' * W2 * UrN;
[U,S,V] = svd(G,'econ');
R = U*V';
UrNa = UrN * R';
fprintf('orth(aligned noisy Gal)=%.2e\n', norm(UrNa'*W2*UrNa - eye(r),'fro'));

LrNa = R * LrN * R';           % aligned operator in clean coords
Neff = UrNa - Ur0;

% ---- Galerkin decomposition (your A0..A3)
A0 = Ur0'  * W2 * L * Ur0;
A1 = Neff' * W2 * L * Ur0;
A2 = Ur0'  * W2 * L * Neff;
A3 = Neff' * W2 * L * Neff;

fprintf('Gal DECOMP CHECK: ||LrNa - sum||F = %.3e\n', norm(LrNa - (A0+A1+A2+A3),'fro'));

PrNa = expm(dt * LrNa);

%% -------------------- NOISY PETROV–GALERKIN (noise in BOTH trial+test modes) --------------------
% Noise injected in WEIGHTED mode space for BOTH Utrial and Vtest,
% then rebuild UrN_pg and VrN_pg (biorth enforced).

rng(seed + 10);          % separate stream from Galerkin noise

Uwall = W * Utrial;
Vwall = W * Vtest;

UwallN = Uwall + sigma * randn(size(Uwall));
VwallN = Vwall + sigma * randn(size(Vwall));

UtrialN = W \ UwallN;
VtestN  = W \ VwallN;

UrN_pg = W2_orthonormalize(UtrialN(:,1:r), W);
V0N_pg = W2_orthonormalize(VtestN(:,1:r),  W);

M_pgN  = V0N_pg' * W2 * UrN_pg;
VrN_pg = V0N_pg / (M_pgN');          % enforce Vr'W2Ur = I

fprintf('PG noisy: orthU=%.2e | bio=%.2e | cond(M)=%.2e\n', ...
    norm(UrN_pg'*W2*UrN_pg - eye(r),'fro'), ...
    norm(VrN_pg'*W2*UrN_pg - eye(r),'fro'), ...
    cond(M_pgN));

LrN_pg = VrN_pg' * W2 * L * UrN_pg;
PrN_pg = expm(dt * LrN_pg);

%% -------------------- ALIGN noisy PG bases to clean PG bases (gauge-fix) --------------------
% Align UrN_pg -> Ur_pg, and VrN_pg -> Vr_pg, then re-biorth.

Gu = Ur_pg' * W2 * UrN_pg;
[Uu,~,Vu] = svd(Gu,'econ');
Ru = Uu*Vu';
UrNa_pg = UrN_pg * Ru';

Gv = Vr_pg' * W2 * VrN_pg;
[Uv,~,Vv] = svd(Gv,'econ');
Rv = Uv*Vv';
VrNa_pg = VrN_pg * Rv';

% Re-enforce biorth AFTER alignment
M_pgNa = VrNa_pg' * W2 * UrNa_pg;
VrNa_pg = VrNa_pg / (M_pgNa');

fprintf('PG aligned: bio=%.2e | cond(M)=%.2e\n', ...
    norm(VrNa_pg'*W2*UrNa_pg - eye(r),'fro'), cond(M_pgNa));

LrNa_pg = VrNa_pg' * W2 * L * UrNa_pg;
PrNa_pg = expm(dt * LrNa_pg);

%% -------------------- PG decomposition  --------------------
Nu = UrNa_pg - Ur_pg;
Nv = VrNa_pg - Vr_pg;

A0pg = Vr_pg' * W2 * L * Ur_pg;
A1pg = Nv'    * W2 * L * Ur_pg;
A2pg = Vr_pg' * W2 * L * Nu;
A3pg = Nv'    * W2 * L * Nu;

fprintf('PG DECOMP CHECK: ||LrNa_pg - sum||F = %.3e\n', norm(LrNa_pg - (A0pg+A1pg+A2pg+A3pg),'fro'));
%%
%% ============================================================
%  PG REMOVE-TERM WHAT-IF ROMs (aligned coords)
%  Build operators: LrNa_pg - A1pg, -A2pg, -(A1pg+A2pg), -A3pg
%  Then rollout using UrNa_pg,VrNa_pg (PG!)
% ============================================================

% ---- Build "remove-term" reduced generators FROM ALIGNED noisy PG operator
LrPG_rmA1   = LrNa_pg - A1pg;
LrPG_rmA2   = LrNa_pg - A2pg;
LrPG_rmLIN  = LrNa_pg - (A1pg + A2pg);
LrPG_rmA3   = LrNa_pg - A3pg;

% ---- Corresponding propagators
PrPG_Na     = expm(dt * LrNa_pg);
PrPG_rmA1   = expm(dt * LrPG_rmA1);
PrPG_rmA2   = expm(dt * LrPG_rmA2);
PrPG_rmLIN  = expm(dt * LrPG_rmLIN);
PrPG_rmA3   = expm(dt * LrPG_rmA3);

% ---- Least-stable eigenvalue report (optional, but useful)
% report_ls('PG noisy aligned LrNa_pg', LrNa_pg);
% report_ls('PG noisy -A1',            LrPG_rmA1);
% report_ls('PG noisy -A2',            LrPG_rmA2);
% report_ls('PG noisy -(A1+A2)',       LrPG_rmLIN);
% report_ls('PG noisy -A3',            LrPG_rmA3);

%% -------------------- PG ROLLOUTS (remove-term) --------------------
%% -------------------- ROLLOUTS --------------------
QrPG0  = roll_PG(Pr_pg,   Ur_pg,   Vr_pg,   W2, Qf, k0);     % PG clean
%QrPGNa = roll_PG(PrNa_pg, UrNa_pg, VrNa_pg, W2, Qf, k0);     % PG noisy aligned
QrPG_Na    = roll_PG(PrPG_Na,    UrNa_pg, VrNa_pg, W2, Qf, k0);
QrPG_A1    = roll_PG(PrPG_rmA1,  UrNa_pg, VrNa_pg, W2, Qf, k0);
QrPG_A2    = roll_PG(PrPG_rmA2,  UrNa_pg, VrNa_pg, W2, Qf, k0);
QrPG_LIN   = roll_PG(PrPG_rmLIN, UrNa_pg, VrNa_pg, W2, Qf, k0);
QrPG_A3    = roll_PG(PrPG_rmA3,  UrNa_pg, VrNa_pg, W2, Qf, k0);

%% -------------------- PG MODEL ERRORS (remove-term) --------------------
% Model error must use VrNa_pg in coordinates:
[Em_PGNa,   rho_PGNa]   = model_err_PG(Qf, UrNa_pg, VrNa_pg, W2, PrPG_Na,    k0);
[Em_PG_A1,  rho_PG_A1]  = model_err_PG(Qf, UrNa_pg, VrNa_pg, W2, PrPG_rmA1,  k0);
[Em_PG_A2,  rho_PG_A2]  = model_err_PG(Qf, UrNa_pg, VrNa_pg, W2, PrPG_rmA2,  k0);
[Em_PG_LIN, rho_PG_LIN] = model_err_PG(Qf, UrNa_pg, VrNa_pg, W2, PrPG_rmLIN, k0);
[Em_PG_A3,  rho_PG_A3]  = model_err_PG(Qf, UrNa_pg, VrNa_pg, W2, PrPG_rmA3,  k0);

fprintf('\nPG remove-term rho(Pr): noisy=%.4g  -A1=%.4g  -A2=%.4g  -(A1+A2)=%.4g  -A3=%.4g\n', ...
    rho_PGNa, rho_PG_A1, rho_PG_A2, rho_PG_LIN, rho_PG_A3);

%% -------------------- Pointwise relative error at idx --------------------
% --- Pointwise relative errors at idx (like your Gal plot)
qf = Qf(idx,:).';

q0i  = QrPG0(idx,:).';     % PG clean
qNi  = QrPG_Na(idx,:).';    % PG noisy aligned

qA1i = QrPG_A1(idx,:).';
qA2i = QrPG_A2(idx,:).';
qLNi = QrPG_LIN(idx,:).';
qQDi = QrPG_A3(idx,:).';

den_i = max(abs(qf(1)), 1e-14);
rel = @(x) abs(x - qf) ./ den_i;

e0i  = rel(q0i);
eN_i  = rel(qNi);
eA1_i = rel(qA1i);
eA2_i = rel(qA2i);
eLN_i = rel(qLNi);
eA3_i = rel(qQDi);

figure('Color','w');
tiledlayout(1,1,'Padding','compact','TileSpacing','compact');

% ---- Error curves 
% nexttile; hold on; grid on;
% plot(t, max(eN_i,1e-16), 'k-',  'LineWidth',1.8, 'DisplayName',sprintf('PG noisy (\\rho=%.3g)',rho_PGNa));
% plot(t, max(eA1_i,  1e-16), 'm--', 'LineWidth',1.6, 'DisplayName',sprintf('PG (noisy - A1) (\\rho=%.3g)',rho_PG_A1));
% plot(t, max(eA2_i,  1e-16), 'c--', 'LineWidth',1.6, 'DisplayName',sprintf('PG (noisy - A2) (\\rho=%.3g)',rho_PG_A2));
% plot(t, max(eLN_i, 1e-16), 'r--', 'LineWidth',1.6, 'DisplayName',sprintf('PG (noisy - (A1+A2)) (\\rho=%.3g)',rho_PG_LIN));
% plot(t, max(eA3_i,  1e-16), 'g--', 'LineWidth',1.6, 'DisplayName',sprintf('PG (noisy - A3) (\\rho=%.3g)',rho_PG_A3));
% set(gca,'YScale','log');
% xlabel('time'); ylabel('rel error at idx');
% title(sprintf('PG remove-term: pointwise rel error (idx=%d)', idx));
% legend('Location','best');

% ---- Model error curves (global W2 error)
nexttile; hold on; grid on;
plot(t, max(Em_PGNa,  1e-16), 'k-',  'LineWidth',1.8, 'DisplayName','PG noisy');
plot(t, max(Em_PG_A1, 1e-16), 'm--', 'LineWidth',1.6, 'DisplayName','PG (noisy - A1)');
plot(t, max(Em_PG_A2, 1e-16), 'c--', 'LineWidth',1.6, 'DisplayName','PG (noisy - A2)');
plot(t, max(Em_PG_LIN,1e-16), 'r--', 'LineWidth',1.6, 'DisplayName','PG (noisy - (A1+A2))');
plot(t, max(Em_PG_A3, 1e-16), 'g--', 'LineWidth',1.6, 'DisplayName','PG (noisy - A3)');
set(gca,'YScale','log');
xlabel('time'); ylabel('model error');
title('PG remove-term: model error');
legend('Location','best');


%% ============================================================
%  PG PLOT: 6 tiles (clean/noisy + remove-term variants)
%  
% ============================================================

figure('Color','w');
tiledlayout(3,2,'Padding','compact','TileSpacing','compact');

% Tile 1: clean + noisy + (optional) corr (PG has no trace-corr here)
nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','PG clean');
plot(t, max(eN_i,1e-16),  'r-',  'LineWidth',1.6, 'DisplayName','PG noisy ');
% plot(t, max(eCi,1e-16),  'g--', 'LineWidth',1.6, 'DisplayName','PG noisy-corr'); % if you add a PG correction later
xlabel('time'); ylabel('rel error');
title('PG: clean vs noisy ');
legend('Location','best');
set(gca,'YScale','log');

% Tile 2: -(A1+A2)
nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','PG clean');
plot(t, max(eLN_i,1e-16), 'm--', 'LineWidth',1.6, 'DisplayName','PG noisy - (A1+A2)');
xlabel('time'); ylabel('rel error');
title('PG: remove (A1+A2)');
legend('Location','best');
set(gca,'YScale','log');

% Tile 3: -A2
nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','PG clean');
plot(t, max(eA2_i,1e-16), 'c--', 'LineWidth',1.6, 'DisplayName','PG noisy - A2');
xlabel('time'); ylabel('rel error');
title('PG: remove A2');
legend('Location','best');
set(gca,'YScale','log');

% Tile 4: -A1
nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','PG clean');
plot(t, max(eA1_i,1e-16), 'm--', 'LineWidth',1.6, 'DisplayName','PG noisy - A1');
xlabel('time'); ylabel('rel error');
title('PG: remove A1');
legend('Location','best');
set(gca,'YScale','log');

% Tile 5: -A3
nexttile; hold on; grid on;
plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','PG clean');
plot(t, max(eA3_i,1e-16), 'r--', 'LineWidth',1.6, 'DisplayName','PG noisy - A3');
xlabel('time'); ylabel('rel error');
title('PG: remove A3');
legend('Location','best');
set(gca,'YScale','log');

% Tile 6: noisy aligned alone vs clean (optional comparison)
% nexttile; hold on; grid on;
% plot(t, max(e0i,1e-16),  'b-',  'LineWidth',1.6, 'DisplayName','PG clean');
% plot(t, max(eN_i,1e-16),  'r-',  'LineWidth',1.6, 'DisplayName','PG noisy aligned');
% xlabel('time'); ylabel('rel error');
% title('PG: (repeat) clean vs noisy');
% legend('Location','best');
% set(gca,'YScale','log');

%% ============================================================
%  PG TERM-ANALYSIS PLOTS (A1pg, A2pg, A3pg)  
% ============================================================

% --- Term magnitudes (Frobenius)
n0pg  = norm(A0pg,'fro');
n1pg  = norm(A1pg,'fro');
n2pg  = norm(A2pg,'fro');
n12pg = norm(A1pg + A2pg,'fro');
n3pg  = norm(A3pg,'fro');

fprintf('\nPG TERM MAGNITUDES (aligned):\n');
fprintf('||Nu||F  = %.3e,  ||Nv||F  = %.3e\n', norm(Nu,'fro'), norm(Nv,'fro'));
fprintf('||A0||F  = %.3e\n', n0pg);
fprintf('||A1||F  = %.3e\n', n1pg);
fprintf('||A2||F  = %.3e\n', n2pg);
fprintf('||A1+A2||F = %.3e\n', n12pg);
fprintf('||A3||F  = %.3e\n', n3pg);

fprintf('RELATIVE to ||A0||F:\n');
fprintf('||A1||/||A0||    = %.3e\n', n1pg/max(n0pg,eps));
fprintf('||A2||/||A0||    = %.3e\n', n2pg/max(n0pg,eps));
fprintf('||A1+A2||/||A0|| = %.3e\n', n12pg/max(n0pg,eps));
fprintf('||A3||/||A0||    = %.3e\n', n3pg/max(n0pg,eps));

%% -------------------- PLOT 1: bar of PG term magnitudes --------------------
figure('Color','w');
vals   = [n1pg n2pg n12pg n3pg];
labels = {'||A1_{pg}||','||A2_{pg}||','||A1_{pg}+A2_{pg}||','||A3_{pg}||'};
bar(vals);
set(gca,'XTick',1:numel(vals),'XTickLabel',labels);
ylabel('Frobenius norm');
title(sprintf('PG term magnitudes (sigma=%.0e)', sigma));
grid on;
set(gca,'YScale','log');

%% -------------------- PLOT 2: heatmaps of A1pg,A2pg,A3pg --------------------
figure('Color','w');
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

nexttile; imagesc(abs(A1pg)); axis tight; colorbar;
title('|A1_{pg}| = |Nv^H W2 L Ur|'); xlabel('mode'); ylabel('mode');

nexttile; imagesc(abs(A2pg)); axis tight; colorbar;
title('|A2_{pg}| = |Vr^H W2 L Nu|'); xlabel('mode'); ylabel('mode');

nexttile; imagesc(abs(A3pg)); axis tight; colorbar;
title('|A3_{pg}| = |Nv^H W2 L Nu|'); xlabel('mode'); ylabel('mode');

%% -------------------- PLOT 3:  A1/A2 diagnostics --------------------
% A2pg corresponds to (Vr)^H W2 (L Nu). In weighted coordinates:
% A2pg = (W*Vr)^H * (W*(L*Nu))
X2 = W * Vr_pg;         % n x r
Y2 = W * (L * Nu);      % n x r
A2hat_pg = X2' * Y2;    % should match A2pg (up to numerical noise)

% A1pg corresponds to (Nv)^H W2 (L Ur). In weighted coordinates:
% A1pg = (W*Nv)^H * (W*(L*Ur))
X1 = (W2*L)'*Nv;            % n x r
Y1 =  Ur_pg;   % n x r
A1hat_pg = X1' * Y1;    % should match A1pg

% A3pg corresponds to (Nv)^H W2 (L Nu). In weighted coordinates:
X3 = W * Nv;
Y3 = W * (L * Nu);
A3hat_pg = X3' * Y3;

fprintf('\nPG channel checks:\n');
fprintf('  ||A1pg - A1hat_pg||F = %.3e\n', norm(A1pg - A1hat_pg,'fro'));
fprintf('  ||A2pg - A2hat_pg||F = %.3e\n', norm(A2pg - A2hat_pg,'fro'));
fprintf('  ||A3pg - A3hat_pg||F = %.3e\n', norm(A3pg - A3hat_pg,'fro'));

% --- Plot the "inputs" and "outputs" for each channel
figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

nexttile; imagesc(abs(X1)); axis tight; colorbar;
title("|(W2*L)'*Nv| (A1 )"); xlabel('mode'); ylabel('state');

nexttile; imagesc(abs(Y1)); axis tight; colorbar;
title('|(Ur_PG)| (A1 )'); xlabel('mode'); ylabel('state');
%%
figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; imagesc(abs(X2)); axis tight; colorbar;
title('|W*Vr| (A2 )'); xlabel('mode'); ylabel('state');

nexttile; imagesc(abs(Y2)); axis tight; colorbar;
title('|W*(L*Nu)| (A2 )'); xlabel('mode'); ylabel('state');
%%
nexttile; imagesc(abs(X3)); axis tight; colorbar;
title('|W*Nv| (A3 channel input)'); xlabel('mode'); ylabel('state');

nexttile; imagesc(abs(Y3)); axis tight; colorbar;
title('|W*(L*Nu)| (A3 channel output)'); xlabel('mode'); ylabel('state');

%% -------------------- PLOT 4: per-mode diagonal dot + cosine (direction) --------------------
dotdiag = @(X,Y) arrayfun(@(j) abs(X(:,j)'*Y(:,j)), 1:size(X,2));
cosdiag = @(X,Y) arrayfun(@(j) abs(X(:,j)'*Y(:,j))/max(norm(X(:,j))*norm(Y(:,j)),eps), 1:size(X,2));

dotA1 = dotdiag(X1,Y1);  cosA1 = cosdiag(X1,Y1);
dotA2 = dotdiag(X2,Y2);  cosA2 = cosdiag(X2,Y2);
dotA3 = dotdiag(X3,Y3);  cosA3 = cosdiag(X3,Y3);

figure('Color','w');
tiledlayout(3,2,'Padding','compact','TileSpacing','compact');

nexttile; bar(dotA1); grid on;
title('A1pg diag dot |(W*Nv)_j^H (W*(L*Ur))_j|'); xlabel('mode'); ylabel('dot');

nexttile; bar(cosA1); grid on; ylim([0 1]);
title('A1pg per-mode cosine'); xlabel('mode'); ylabel('cos');

nexttile; bar(dotA2); grid on;
title('A2pg diag dot |(W*Vr)_j^H (W*(L*Nu))_j|'); xlabel('mode'); ylabel('dot');

nexttile; bar(cosA2); grid on; ylim([0 1]);
title('A2pg per-mode cosine'); xlabel('mode'); ylabel('cos');

nexttile; bar(dotA3); grid on;
title('A3pg diag dot |(W*Nv)_j^H (W*(L*Nu))_j|'); xlabel('mode'); ylabel('dot');

nexttile; bar(cosA3); grid on; ylim([0 1]);
title('A3pg per-mode cosine'); xlabel('mode'); ylabel('cos');

%% -------------------- PLOT 5: diagonal-energy ratio (off-diagonal dominance check) --------------------
% diagEnergy = @(A) norm(diag(diag(A)),'fro')/max(norm(A,'fro'),eps);
% 
% fprintf('\nPG off-diagonal dominance (diag energy / full energy):\n');
% fprintf('  A1pg: %.3e\n', diagEnergy(A1pg));
% fprintf('  A2pg: %.3e\n', diagEnergy(A2pg));
% fprintf('  A3pg: %.3e\n', diagEnergy(A3pg));
% 
% figure('Color','w'); hold on; grid on;
% bar([diagEnergy(A1pg), diagEnergy(A2pg), diagEnergy(A3pg)]);
% set(gca,'XTick',1:3,'XTickLabel',{'A1pg','A2pg','A3pg'});
% ylabel('||diag(A)||_F / ||A||_F');
% title('PG diagonal energy ratio (smaller = more off-diagonal)');
%% ============================================================
%  LEAST STABLE EIGENVALUE INFO (FOM + ROM variants)
% ============================================================
omF = eig(A,M);
[~,iF] = max(imag(omF));
om_ls = omF(iF);
fprintf('\nFOM least-stable omega: %.6g%+.6gi  (imag=%.6g)\n', real(om_ls), imag(om_ls), imag(om_ls));

report_ls = @(name, Lr) report_least_stable(name, Lr, alpha);

report_ls('Gal clean   Lr0',   Lr0);
report_ls('Gal noisy   LrNa',  LrNa);
report_ls('PG clean    Lr_pg', Lr_pg);
report_ls('PG noisy    LrNa_pg',LrNa_pg);

%% ============================================================
%  ROLLOUTS (IMPORTANT: PG uses roll_PG, not roll_Pr)
% ============================================================
Qr0      = roll_Pr(Pr0,     Ur0,   W2, Qf, k0);          % Gal clean
QrNa     = roll_Pr(PrNa,    UrNa,  W2, Qf, k0);          % Gal noisy aligned

QrPG0    = roll_PG(Pr_pg,   Ur_pg, Vr_pg,   W2, Qf, k0); % PG clean
QrPGNa   = roll_PG(PrNa_pg, UrNa_pg, VrNa_pg, W2, Qf, k0); % PG noisy aligned

%% ============================================================
%  ERRORS (Gal vs PG)
% ============================================================
Ep_G0  = proj_err(Qf, Ur0,  W2, k0);
Ep_GNa = proj_err(Qf, UrNa, W2, k0);

[Em_G0,  rho_G0]  = model_err(Qf, Ur0,  W2, Pr0,  k0);
[Em_GNa, rho_GNa] = model_err(Qf, UrNa, W2, PrNa, k0);

Ep_PG0  = proj_err_PG(Qf, Ur_pg,  Vr_pg,  W2, k0);
Ep_PGNa = proj_err_PG(Qf, UrNa_pg,VrNa_pg,W2, k0);

[Em_PG0,  rho_PG0]  = model_err_PG(Qf, Ur_pg,  Vr_pg,  W2, Pr_pg,  k0);
[Em_PGNa, rho_PGNa] = model_err_PG(Qf, UrNa_pg,VrNa_pg,W2, PrNa_pg,k0);

fprintf('\nrho(Pr): Gal clean=%.4g Gal noisy=%.4g | PG clean=%.4g PG noisy=%.4g\n', ...
    rho_G0, rho_GNa, rho_PG0, rho_PGNa);

%% ============================================================
%  PLOTS 
% ============================================================
qf      = Qf(idx,:).';
qG0     = Qr0(idx,:).';
qGNa    = QrNa(idx,:).';
qPG0    = QrPG0(idx,:).';
qPGNa   = QrPGNa(idx,:).';

den_i = max(abs(qf(1)), 1e-14);
rel = @(x) abs(x - qf) ./ den_i;

eG0   = rel(qG0);
eGNa  = rel(qGNa);
ePG0  = rel(qPG0);
ePGNa = rel(qPGNa);

figure('Color','w'); hold on; grid on;
plot(t, real(qf),   'k-',  'LineWidth',3.0, 'DisplayName','FOM');
plot(t, real(qG0),  'b--', 'LineWidth',2.5, 'DisplayName','Gal clean');
plot(t, real(qGNa), 'r--', 'LineWidth',1.6, 'DisplayName','Gal noisy');
plot(t, real(qPG0), 'g--', 'LineWidth',1.6, 'DisplayName','PG clean');
plot(t, real(qPGNa),'m--', 'LineWidth',1.6, 'DisplayName','PG noisy');
xlabel('time'); ylabel(sprintf('Re(q(idx=%d))', idx));
title('Trajectories');
legend('Location','best');
%%
figure('Color','w'); hold on; grid on;
plot(t, max(eG0,1e-16),  'b-', 'LineWidth',1.6, 'DisplayName','Gal clean');
plot(t, max(eGNa,1e-16), 'r-', 'LineWidth',1.6, 'DisplayName','Gal noisy ');
plot(t, max(ePG0,1e-16), 'g-', 'LineWidth',1.6, 'DisplayName','PG clean');
plot(t, max(ePGNa,1e-16),'m-', 'LineWidth',1.6, 'DisplayName','PG noisy ');
set(gca,'YScale','log');
xlabel('time'); ylabel('relative error');
title('Pointwise relative error');
legend('Location','best');
%%
figure('Color','w'); hold on; grid on;
plot(t, max(Ep_G0,1e-16),  'b-', 'LineWidth',1.6, 'DisplayName','Proj Gal clean');
plot(t, max(Ep_GNa,1e-16), 'r-', 'LineWidth',1.6, 'DisplayName','Proj Gal noisy');
plot(t, max(Ep_PG0,1e-16), 'g-', 'LineWidth',1.6, 'DisplayName','Proj PG clean');
plot(t, max(Ep_PGNa,1e-16),'m-', 'LineWidth',1.6, 'DisplayName','Proj PG noisy');
set(gca,'YScale','log');
xlabel('time'); ylabel('projection error');
title('Projection error');
legend('Location','best');

figure('Color','w'); hold on; grid on;
plot(t, max(Em_G0,1e-16),  'b-', 'LineWidth',1.6, 'DisplayName',sprintf('Model Gal clean (rho=%.3g)',rho_G0));
plot(t, max(Em_GNa,1e-16), 'r-', 'LineWidth',1.6, 'DisplayName',sprintf('Model Gal noisy (rho=%.3g)',rho_GNa));
plot(t, max(Em_PG0,1e-16), 'g-', 'LineWidth',1.6, 'DisplayName',sprintf('Model PG clean (rho=%.3g)',rho_PG0));
plot(t, max(Em_PGNa,1e-16),'m-', 'LineWidth',1.6, 'DisplayName',sprintf('Model PG noisy (rho=%.3g)',rho_PGNa));
set(gca,'YScale','log');
xlabel('time'); ylabel('model error');
title('Model error');
legend('Location','best');

%% ===================== LOCAL HELPERS =====================

function rom = rom_build_clean(Q, W, W2, L, r, alpha)
    Xw = W * Q;
    [Uw, ~, ~] = svd(Xw, 'econ');
    U = W \ Uw;
    rom.U = U;

    Ur = W2_orthonormalize(U(:,1:r), W);
    rom.Ur = Ur;
    rom.orthErr = norm(Ur' * W2 * Ur - eye(r), 'fro');

    Lr = Ur' * W2 * L * Ur;
    rom.Lr = Lr;

    [Vr, Dr] = eig(Lr);
    lam = diag(Dr);
    c = (1i * lam) / alpha;

    rom.Vr = Vr;
    rom.c  = c(:);

    [~, ord] = sort(imag(c), 'descend');
    rom.c_sort  = c(ord);
    rom.Vr_sort = Vr(:, ord);
end

function rom = rom_build_pg(Qtrial, Qtest, W, W2, L, r, alpha)
    % --- Trial POD modes
    Xw_tr = W * Qtrial;
    [Uw_tr, ~, ~] = svd(Xw_tr, 'econ');
    Utrial = W \ Uw_tr;
    rom.Utrial = Utrial;

    % --- Test POD modes
    Xw_te = W * Qtest;
    [Uw_te, ~, ~] = svd(Xw_te, 'econ');
    Vtest = W \ Uw_te;
    rom.Vtest = Vtest;

    % --- Bases
    Ur = W2_orthonormalize(Utrial(:,1:r), W);
    V0 = W2_orthonormalize(Vtest(:,1:r),  W);

    M0 = V0' * W2 * Ur;         % coupling
    Vr = V0 / (M0');            % enforce Vr'W2Ur = I

    rom.Ur = Ur;
    rom.Vr = Vr;

    rom.orthErrU = norm(Ur' * W2 * Ur - eye(r), 'fro');
    rom.bioErr   = norm(Vr' * W2 * Ur - eye(r), 'fro');
    rom.condM    = cond(M0);

    % --- PG reduced generator
    Lr = Vr' * W2 * L * Ur;
    rom.Lr = Lr;

    % --- eigendata
    [Veig, Dr] = eig(Lr);
    lam = diag(Dr);
    c = (1i * lam) / alpha;

    rom.Veig = Veig;
    rom.c    = c(:);

    [~, ord] = sort(imag(c), 'descend');
    rom.c_sort    = c(ord);
    rom.Veig_sort = Veig(:, ord);
end

function Qr = roll_Pr(Pr, Ur, W2, Qf, k0)
    K = size(Qf,2);
    a = Ur' * W2 * Qf(:,k0);
    Qr = zeros(size(Qf));
    for k = 1:K
        Qr(:,k) = Ur * a;
        a = Pr * a;
    end
end

function Qr = roll_PG(Pr, Ur, Vr, W2, Qf, k0)
    K = size(Qf,2);
    a = Vr' * W2 * Qf(:,k0);
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

function Ep = proj_err_PG(Qf, Ur, Vr, W2, k0)
    K = size(Qf,2);
    wn = @(v) sqrt(real(v'*(W2*v)));
    den = max( wn(Qf(:,k0)), eps );
    Ep = zeros(K,1);
    for k = 1:K
        qk = Qf(:,k);
        ak = Vr' * W2 * qk;
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

function [Em, rho] = model_err_PG(Qf, Ur, Vr, W2, Pr, k0)
    K = size(Qf,2);
    wn = @(v) sqrt(real(v'*(W2*v)));
    rho = max(abs(eig(Pr)));
    Em = zeros(K,1);
    a = Vr' * W2 * Qf(:,k0);
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