%% exp_eps_pseudospectrum_terms.m
% epsilon-pseudospectra: contours of sigma_min(zI-A) = eps
% (single realization, gauge-fixed via W2-Procrustes alignment)
%
% Requires:
%   OS_Op, clencurt_WeightsB, rom_build_clean, W2_orthonormalize
%
% Definition used:
%   z is in eps-pseudospectrum <=> sigma_min(zI - A) <= eps

clear; close all; clc;
addpath(genpath('../1src'));

%% ---------------- SETTINGS ----------------
Re=1000; N=128; alpha=1; beta=0; meanInd=2;
dt = 1e-2;  K = 4000;  r = 25;

sigma = 1e-6;
seed  = 3001;

ngrid     = 280;
padFrac   = 0.8;
focusBand = 2.0;      % clamp to rightmost: Re >= maxRe - focusBand

% Choose eps levels (ABSOLUTE eps in 2-norm)
% We'll auto-scale these from the clean A0 grid so contours show up.
nEpsLevels = 8;       % number of eps contours to draw

%% ---------------- BUILD OS + GENERATOR ----------------
[A,M,~] = OS_Op(alpha,beta,N,Re,meanInd);
n = size(A,1);

L   = -1i*(M\A);
Phi = expm(dt*L);

w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

%% ---------------- SNAPSHOTS for POD ----------------
rng(1);
q0 = randn(n,1); q0 = q0/norm(q0);

Qf = zeros(n,K);
q  = q0;
for k=1:K
    q = Phi*q;
    Qf(:,k) = q;
end

%% ---------------- CLEAN ROM ----------------
rom = rom_build_clean(Qf, W, W2, L, r, alpha);
Ur0  = rom.Ur;
Uall = rom.U;

Lr0 = Ur0' * W2 * L * Ur0;   % A0

%% ---------------- NOISY + ALIGN ----------------
rng(seed);
Uw  = W * Uall;
UwN = Uw + sigma * randn(size(Uw));
UN  = W \ UwN;

UrN = W2_orthonormalize(UN(:,1:r), W);
LrN = UrN' * W2 * L * UrN;

G = Ur0' * W2 * UrN;
[U,~,V] = svd(G,'econ');
R = U*V';

UrNa = UrN * R';
LrNa = R * LrN * R';

%% ---------------- DECOMPOSE ----------------
Neff = UrNa - Ur0;

A0 = Ur0'  * W2 * L * Ur0;
A1 = Neff' * W2 * L * Ur0;
A2 = Ur0'  * W2 * L * Neff;
A3 = Neff' * W2 * L * Neff;

fprintf('DECOMP CHECK: ||LrNa-(A0+A1+A2+A3)||F = %.3e\n', norm(LrNa-(A0+A1+A2+A3),'fro'));

%% ---------------- CASES ----------------
Mats  = { A0, A0+A1, A0+A2, A0+A3, A0+A1+A2, LrNa };
names = { 'A0 (clean)', 'A0+A1', 'A0+A2', 'A0+A3', 'A0+A1+A2', 'LrNa (full noisy)' };

%% ---------------- GRID: HARD CLAMP to rightmost band ----------------
allLam = [];
for j=1:numel(Mats)
    allLam = [allLam; eig(Mats{j})]; 
end
maxRe = max(real(allLam));
lam_focus = allLam(real(allLam) >= maxRe - focusBand);

% bbox + minimum span
xr = [min(real(lam_focus)) max(real(lam_focus))];
yi = [min(imag(lam_focus)) max(imag(lam_focus))];

dx = max(xr(2)-xr(1), 0.5);
dy = max(yi(2)-yi(1), 0.5);

xc = mean(xr); yc = mean(yi);

x = linspace(xc - (1+padFrac)*dx/2, xc + (1+padFrac)*dx/2, ngrid);
y = linspace(yc - (1+padFrac)*dy/2, yc + (1+padFrac)*dy/2, ngrid);
[X,Y] = meshgrid(x,y);
Z = X + 1i*Y;

xlims = [min(x) max(x)];
ylims = [min(y) max(y)];

fprintf('Zoom window: Re in [%.3g, %.3g], Im in [%.3g, %.3g]\n', xlims(1), xlims(2), ylims(1), ylims(2));

%% ---------------- AUTO-SELECT eps LEVELS (from A0) ----------------
smin0 = sigma_min_grid(A0, Z);
s0 = smin0(:);
s0 = s0(isfinite(s0) & s0>0);

% eps should live where contours exist: between small and moderate sigma_min
eps_lo = quantile(s0, 0.02);
eps_hi = quantile(s0, 0.30);

eps_lo = max(eps_lo, 1e-16);
eps_hi = max(eps_hi, 10*eps_lo);

eps_levels = logspace(log10(eps_lo), log10(eps_hi), nEpsLevels);

fprintf('eps levels: [%.2e ... %.2e]\n', eps_levels(1), eps_levels(end));

%% ---------------- PLOT: epsilon-pseudospectrum contours ----------------
figure('Color','w');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

for j=1:numel(Mats)
    Atest = Mats{j};

    smin = sigma_min_grid(Atest, Z);

    nexttile; hold on; grid on;

    % eps-pseudospectrum boundary: sigma_min = eps
    %contour(X,Y,smin,eps_levels,'LineWidth',1.1);
[Cs,h] = contour(X,Y,smin,eps_levels,'LineWidth',1.1);
%clabel(Cs,h,'FontSize',8,'Color','k','LabelSpacing',400);
% contourf(X,Y,log10(smin),20,'LineStyle','none');
% colorbar;
    % plot eigenvalues inside window
    lam = eig(Atest);
    in  = (real(lam) >= xlims(1)) & (real(lam) <= xlims(2)) & ...
          (imag(lam) >= ylims(1)) & (imag(lam) <= ylims(2));
    plot(real(lam(in)), imag(lam(in)), 'k.', 'MarkerSize',10);

    xlabel('Re(z)'); ylabel('Im(z)');
    title(names{j});

    xlim(xlims); ylim(ylims);
end

sgtitle('\epsilon-pseudospectra: contours of \sigma_{min}(zI-A)=\epsilon');

%% ---------------- helper ----------------
function smin = sigma_min_grid(A, Z)
r = size(A,1);
I = eye(r);
smin = zeros(size(Z));
for k = 1:numel(Z)
    smin(k) = min(svd(Z(k)*I - A));
end
end