%% exp04_all_plots.m
% FOM vs ROM clean vs ROM noisy vs ROM noisy-corrected
% Plots:
%   (1) trajectory at idx (Re) + relative errors
%   (2) projection error
%   (3) model error (noisy vs corrected)
%   (4) mode shapes: FOM vs ROM clean vs ROM noisy vs ROM corrected

clear; close all; clc;
addpath(genpath('../1src'));

%% -------------------- SETTINGS --------------------
Re=1000; N=60; alpha=1; beta=0; meanInd=2;
dt=1e-2; K=4000;
r=25; k0=1;

sigma = 1e-3;        % std of noise in WEIGHTED space (variance = sigma^2)
seed  = 3001;

nmodes_plot = 3;     % how many leading modes to show in shape plot

% which index to plot (interior index)
idx = [];            % empty -> will use mid index

%rng(1);

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
q0 = randn(n,1); q0 = q0/norm(q0);

Qf = zeros(n,K);
q = q0;
for k = 1:K
    q = Phi*q;
    Qf(:,k) = q;
end
t = (0:K-1)'*dt;

%% -------------------- CLEAN ROM (basis + reduced operator) --------------------
rom = rom_build_clean(Qf, W, W2, L, r, alpha);  % your function
Ur0  = rom.Ur;     % clean W2-orth basis (n×r)
Uall = rom.U;      % all modes (n×Keff)

chk0 = norm(Ur0'*W2*Ur0 - eye(r),'fro');
fprintf('orth(clean)=%.2e\n', chk0);

Lr0 = Ur0' * W2 * L * Ur0;     % clean reduced generator
Pr0 = expm(dt * Lr0);          % clean reduced propagator

%% -------------------- NOISY BASIS (constant variance on Chebyshev) --------------------
rng(seed);
Uw  = W * Uall;
UwN = Uw + sigma * randn(size(Uw));   % i.i.d. in weighted space
UN  = W \ UwN;

UrN = W2_orthonormalize(UN(:,1:r), W);
chkN = norm(UrN'*W2*UrN - eye(r),'fro');
fprintf('orth(noisy)=%.2e\n', chkN);

LrN = UrN' * W2 * L * UrN;     % noisy reduced generator
PrN = expm(dt * LrN);          % noisy reduced propagator

%% -------------------- TRACE–I CORRECTION  --------------------
trA  = trace(L);                            % A = L here (consistent)
bias = (sigma^2) * trA * eye(r);
LrC  = LrN - bias;                          % corrected generator
PrC  = expm(dt * LrC);                      % corrected reduced propagator

fprintf('trace(L)=%.6g%+.6gi,  |bias|=%.3e\n', real(trA), imag(trA), norm(bias,'fro'));

%% -------------------- ROLLOUTS (clean / noisy / corrected) --------------------
Qr0 = roll_Pr(Pr0, Ur0, W2, Qf, k0);   % ROM clean rollout
QrN = roll_Pr(PrN, UrN, W2, Qf, k0);   % ROM noisy rollout
QrC = roll_Pr(PrC, UrN, W2, Qf, k0);   % ROM corrected rollout (same noisy basis)

%% -------------------- ERRORS --------------------
% Projection errors: depend only on basis (best fit each time)
Ep0 = proj_err(Qf, Ur0, W2);
EpN = proj_err(Qf, UrN, W2);

% Model errors: depend on Pr and basis
[Em0, rho0] = model_err(Qf, Ur0, W2, Pr0, k0);
[EmN, rhoN] = model_err(Qf, UrN, W2, PrN, k0);
[EmC, rhoC] = model_err(Qf, UrN, W2, PrC, k0);

fprintf('rho(Pr0)=%.6g, rho(PrN)=%.6g, rho(PrC)=%.6g\n', rho0, rhoN, rhoC);

%% ============================================================
%  PLOT 1: TRAJECTORY at idx (Re) + relative errors
% ============================================================
qf = Qf(idx,:).';
q0i = Qr0(idx,:).';
qNi = QrN(idx,:).';
qCi = QrC(idx,:).';

eps0 = 1e-14;
e0i = abs(q0i - qf) ./ max(abs(qf), eps0);
eNi = abs(qNi - qf) ./ max(abs(qf), eps0);
eCi = abs(qCi - qf) ./ max(abs(qf), eps0);

figure('Color','w');
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

nexttile; hold on; grid on;
plot(t, real(qf), 'k-',  'LineWidth',1.8, 'DisplayName','FOM');
plot(t, real(q0i),'b--', 'LineWidth',1.6, 'DisplayName','ROM clean');
plot(t, real(qNi),'r--', 'LineWidth',1.6, 'DisplayName',sprintf('ROM noisy (\\sigma=%.0e)',sigma));
plot(t, real(qCi),'g--', 'LineWidth',1.6, 'DisplayName','ROM noisy-corr');
ylabel(sprintf('Re(q(idx=%d))', idx));
title('Trajectory at one location');
legend('Location','best');

nexttile; hold on; grid on;
semilogy(t, max(e0i,1e-16), 'b-', 'LineWidth',1.6, 'DisplayName','clean');
semilogy(t, max(eNi,1e-16), 'r-', 'LineWidth',1.6, 'DisplayName','noisy');
semilogy(t, max(eCi,1e-16), 'g--', 'LineWidth',1.6, 'DisplayName','noisy-corr');
xlabel('time'); ylabel('rel error (log)');
title('Pointwise relative error');
legend('Location','best');

%% ============================================================
%  PLOT 2: PROJECTION ERROR (clean basis vs noisy basis)
% ============================================================
figure('Color','w'); hold on; grid on;
semilogy(t, max(Ep0,1e-16), 'b-', 'LineWidth',1.8, 'DisplayName','Proj err (clean basis)');
semilogy(t, max(EpN,1e-16), 'r--', 'LineWidth',1.8, 'DisplayName',sprintf('Proj err (noisy basis, \\sigma=%.0e)',sigma));
xlabel('time'); ylabel('projection error (log)');
title('Projection error (FOM -> ROM subspace)');
legend('Location','best');

%% ============================================================
%  PLOT 3: MODEL ERROR (clean vs noisy vs corrected)
% ============================================================
figure('Color','w'); hold on; grid on;
semilogy(t, max(Em0,1e-16), 'b-', 'LineWidth',1.8, 'DisplayName',sprintf('Model err clean (\\rho=%.4g)',rho0));
semilogy(t, max(EmN,1e-16), 'r-', 'LineWidth',1.8, 'DisplayName',sprintf('Model err noisy (\\rho=%.4g)',rhoN));
semilogy(t, max(EmC,1e-16), 'g--', 'LineWidth',1.8, 'DisplayName',sprintf('Model err noisy-corr (\\rho=%.4g)',rhoC));
xlabel('time'); ylabel('model error (log)');
title('Model error (ROM rollout vs FOM)');
legend('Location','best');

%% ============================================================
%  PLOT 4: MODE SHAPES (FOM vs ROM clean vs ROM noisy vs ROM corrected)
% ============================================================
% Choose nmodes_plot least-stable modes (by imag(omega) descending)

% --- FOM eigenpairs ---
[Vf,Df] = eig(A,M);         % A v = omega M v
om = diag(Df);
[~,ordF] = sort(imag(om),'descend');
om = om(ordF); Vf = Vf(:,ordF);

% pick first nmodes_plot FOM modes
mids = 1:nmodes_plot;

% --- ROM eigenpairs ---
[Vc0,Dc0] = eig(Lr0); lam0 = diag(Dc0);
[VcN,DcN] = eig(LrN); lamN = diag(DcN);
[VcC,DcC] = eig(LrC); lamC = diag(DcC);

% sort ROM modes by imag(c) descending (c = i*lambda/alpha)
c0 = (1i*lam0)/alpha;  [~,o0] = sort(imag(c0),'descend'); Vc0=Vc0(:,o0); c0=c0(o0);
cN = (1i*lamN)/alpha;  [~,oN] = sort(imag(cN),'descend'); VcN=VcN(:,oN); cN=cN(oN);
cC = (1i*lamC)/alpha;  [~,oC] = sort(imag(cC),'descend'); VcC=VcC(:,oC); cC=cC(oC);

figure('Color','w');
tiledlayout(nmodes_plot,1,'Padding','compact','TileSpacing','compact');

for j = 1:nmodes_plot
    % Lift to full space
    pf = Vf(:,mids(j));
    p0 = Ur0 * Vc0(:,j);
    pN = UrN * VcN(:,j);
    pC = UrN * VcC(:,j);

    % phase-align everything to FOM mode
    % p0 = align_phase(pf, p0);
    % pN = align_phase(pf, pN);
    % pC = align_phase(pf, pC);

    % normalize by max abs (add boundary zeros for plotting on y)
    pf = pf / max(abs(pf)+eps);
    p0 = p0 / max(abs(p0)+eps);
    pN = pN / max(abs(pN)+eps);
    pC = pC / max(abs(pC)+eps);

    pfp = [0; pf; 0];
    p0p = [0; p0; 0];
    pNp = [0; pN; 0];
    pCp = [0; pC; 0];

    nexttile; hold on; grid on;
    plot(y, real(pfp), 'k-',  'LineWidth',1.8, 'DisplayName','FOM');
    plot(y, real(p0p), 'b--', 'LineWidth',1.4, 'DisplayName','ROM clean');
    plot(y, real(pNp), 'r-', 'LineWidth',1.4, 'DisplayName','ROM noisy');
    plot(y, real(pCp), 'g--', 'LineWidth',1.4, 'DisplayName','ROM corr');

    title(sprintf('Mode %d | FOM \\omega=%.3g%+.3gi | ROM c0=%.3g%+.3gi', ...
        j, real(om(mids(j))), imag(om(mids(j))), real(c0(j)), imag(c0(j))));

    xlabel('y'); ylabel('Re(mode) (norm)');
    if j==1, legend('Location','best'); end
end

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

function [Em, rho] = model_err(Qf, Ur, W2, Pr, k0)
K = size(Qf,2);
wn = @(v) sqrt(real(v'*(W2*v)));
rho = max(abs(eig(Pr)));

Em = zeros(K,1);
a = Ur' * W2 * Qf(:,k0);
for k = 1:K
    qk = Qf(:,k);
    Em(k) = wn(qk - Ur*a) / max(wn(qk), eps);
    a = Pr * a;
end
end

function b = align_phase(a, b)
den = (b'*b);
if abs(den) < eps, return; end
ph = (b'*a) / den;
if abs(ph) < eps, return; end
b = b * (ph / abs(ph));
end
