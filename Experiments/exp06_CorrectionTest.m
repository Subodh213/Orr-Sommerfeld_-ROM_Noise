%% exp08_oneRun_errors_plus_generator_decomp.m
% One run, one N, one noise realization (MODE noise).
% Figure style like yours:
%   Left  : clean basis -> Projection error + Model error
%   Right : noisy modes -> Projection error + Model errors (noisy, aligned, trace-I)
% Plus: generator decomposition of ΔLr into T1,T2,T3 and bar diagnostics
%
% Requires on path:
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

N  = 128;          % <-- set this
dt = 1e-2;
K  = 4000;        % time horizon (matches your plot)
r  = 25;
k0 = 1;

sigma     = 1e-3; % fixed mode-noise std (WEIGHTED space)
seedNoise = 2;    % single realization

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

%% ---------------- FOM SNAPSHOTS (clean) ----------------
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

% clean errors
Ep0 = proj_err(Qf, Ur0, W2);
Em0 = model_err(Qf, Ur0, W2, Pr0, k0);

%% ---------------- NOISY MODES (perturb only r modes) ----------------
Uw_r = W*(Uall(:,1:r));                 % weighted r modes

rng(seedNoise);
Uw_rN = Uw_r + sigma * randn(size(Uw_r)); % noise in weighted space
UrN   = W2_orthonormalize(W\Uw_rN, W);    % W2-orthonormalize

LrN = UrN' * W2 * L * UrN;
PrN = expm(dt*LrN);

% noisy errors
EpN = proj_err(Qf, UrN, W2);
EmN = model_err(Qf, UrN, W2, PrN, k0);

%% ---------------- ALIGNMENT (remove coordinate drift) ----------------
G = Ur0' * W2 * UrN;        % r×r
[Uq,~,Vq] = svd(G,'econ');
R = Vq*Uq';

UrA = UrN * R;              % aligned basis
LrA = UrA' * W2 * L * UrA;
PrA = expm(dt*LrA);

EpA = proj_err(Qf, UrA, W2);
EmA = model_err(Qf, UrA, W2, PrA, k0);

%% ---------------- TRACE–I CORRECTION (mean-bias) ----------------
Bias = (sigma^2) * trace(L) * eye(r);  % your correction model
LrC  = LrN - Bias;
PrC  = expm(dt*LrC);
EmC  = model_err(Qf, UrN, W2, PrC, k0);

%% ---------------- GENERATOR DECOMPOSITION (effective Neff) ----------------
Neff  = UrN - Ur0;                 % effective perturbation after orthonormalize
T1 = Neff' * W2 * L * Ur0;         % N^* L U
T2 = Ur0'  * W2 * L * Neff;        % U^* L N
T3 = Neff' * W2 * L * Neff;        % N^* L N

Delta = LrN - Lr0;
resid = (T1+T2+T3) - Delta;

off = @(X) X - diag(diag(X));

% scalar summaries
nDelta = norm(Delta,'fro');
nT1    = norm(T1,'fro');
nT2    = norm(T2,'fro');
nT3    = norm(T3,'fro');
nRes   = norm(resid,'fro');

offDelta = norm(off(Delta),'fro') / max(nDelta,eps);
offT12   = norm(off(T1+T2),'fro') / max(nDelta,eps);
offT3    = norm(off(T3),'fro')    / max(nDelta,eps);

%% ---------------- MAIN FIGURE (your style) ----------------
figure('Color','w','Position',[60 160 1550 520]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% Left panel: clean
nexttile; hold on; grid on;
plot(t, Ep0, 'LineWidth',1.8, 'DisplayName','Projection error');
plot(t, Em0, 'LineWidth',1.8, 'DisplayName','Model error');
title('Modes identified from clean data','FontWeight','bold');
xlabel('Time'); ylabel('Error');
legend('Location','best');
axis tight;

% Right panel: noisy
nexttile; hold on; grid on;
plot(t, EpN, 'LineWidth',1.8, 'DisplayName','Projection error');
plot(t, EmN, 'LineWidth',1.8, 'DisplayName','Model error (noisy ROM)');
plot(t, EmA, '--', 'LineWidth',2.0, 'DisplayName','Model error (aligned ROM)');
plot(t, EmC, ':',  'LineWidth',2.2, 'DisplayName','Model error (trace-I corrected)');
title('Modes identified from noisy modes','FontWeight','bold');
xlabel('Time'); ylabel('Error');
legend('Location','best');
axis tight;

annotation('textbox',[0.78 0.19 0.2 0.65], ...
    'String',sprintf('r = %d\n\\sigma = 10^{-3}\nN = %d\nseed = %d', r, N, seedNoise), ...
    'FitBoxToText','on','EdgeColor','none','FontSize',16);

%% ---------------- DIAGNOSTIC FIGURE (terms) ----------------
figure('Color','w','Position',[60 60 1550 420]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

nexttile; grid on; hold on;
bar([nT1 nT2 nT3 nDelta]);
set(gca,'XTickLabel',{'||T1||','||T2||','||T3||','||\Delta||'});
title('Generator term norms (Fro)');

nexttile; grid on; hold on;
bar([offT12 offT3 offDelta]);
set(gca,'XTickLabel',{'off(T1+T2)/||\Delta||','off(T3)/||\Delta||','off(\Delta)/||\Delta||'});
title('Off-diagonal content (relative)');

nexttile; imagesc(real(Delta)); colorbar;
title('Re(\Delta L_r) = Re(L_{r,N} - L_{r,0})');
xlabel('j'); ylabel('i');

%% ---------------- PRINT SUMMARY ----------------
fprintf('\n=== One-run summary (generator + trajectory) ===\n');
fprintf('N=%d, r=%d, sigma=%.1e, seed=%d\n', N, r, sigma, seedNoise);

fprintf('\nTrajectory (final time) model error:\n');
fprintf('clean   Em0(T)=%.3e\n', Em0(end));
fprintf('noisy   EmN(T)=%.3e\n', EmN(end));
fprintf('aligned EmA(T)=%.3e\n', EmA(end));
fprintf('corr    EmC(T)=%.3e\n', EmC(end));

fprintf('\nGenerator decomposition:\n');
fprintf('||Delta|| = %.3e\n', nDelta);
fprintf('||T1||    = %.3e (%.2f x ||Delta||)\n', nT1, nT1/max(nDelta,eps));
fprintf('||T2||    = %.3e (%.2f x ||Delta||)\n', nT2, nT2/max(nDelta,eps));
fprintf('||T3||    = %.3e (%.2f x ||Delta||)\n', nT3, nT3/max(nDelta,eps));
fprintf('||resid|| = %.3e (should be ~0)\n', nRes);

fprintf('\nOff-diagonal ratios (relative to ||Delta||):\n');
fprintf('off(Delta)/||Delta||   = %.3f\n', offDelta);
fprintf('off(T1+T2)/||Delta||   = %.3f\n', offT12);
fprintf('off(T3)/||Delta||      = %.3f\n', offT3);

fprintf('\nInterpretation (quick):\n');
fprintf('- If ||T1|| and ||T2|| are big and off(T1+T2) is big, cross terms dominate -> trace-I won''t fix.\n');
fprintf('- If ||T3|| dominates and Delta mostly diagonal, mean-bias correction is plausible.\n');

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
    a = Pr * a;
end
end
