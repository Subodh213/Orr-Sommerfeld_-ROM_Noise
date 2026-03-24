%% exp_psp_terms_A0_A1_A2_A3_ZOOM_AUTOLEVELS.m
% Clean standalone pseudospectra script:
%   - build clean ROM + noisy ROM (aligned)
%   - decompose LrNa = A0 + A1 + A2 + A3 (aligned Neff)
%   - ZOOM near rightmost eigenvalues (where sensitivity matters)
%   - AUTO-CHOOSE contour levels from data (so contours actually appear)
%   - Plot log10(||(zI-A)^(-1)||2) using sigma_min via 1/sigma_min

clear; close all; clc;
addpath(genpath('../1src'));

%% ---------------- SETTINGS ----------------
Re=1000; N=128; alpha=1; beta=0; meanInd=2;
dt = 1e-2;      % for snapshot generation only
K  = 4000;      % snapshots for POD
r  = 25;        % ROM size

sigma = 5e-8;   % noise std in WEIGHTED space
seed  = 3001;

ngrid   = 260;      % grid resolution
padFrac = 0.60;     % zoom padding around focused eigenvalues
focusBand = 1.0;    % keep eigs with Re > (maxRe - focusBand)
nlev    = 10;       % number of contour levels (auto)

%% ---------------- BUILD OS + GENERATOR ----------------
[A,M,~] = OS_Op(alpha,beta,N,Re,meanInd);
n = size(A,1);

L   = -1i*(M\A);        % full generator
Phi = expm(dt*L);       % full step map

w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

%% ---------------- FOM SNAPSHOTS (for POD basis only) ----------------
rng(1);
q0 = randn(n,1); q0 = q0/norm(q0);

Qf = zeros(n,K);
q  = q0;
for k=1:K
    q = Phi*q;
    Qf(:,k) = q;
end

%% ---------------- CLEAN ROM BASIS ----------------
rom = rom_build_clean(Qf, W, W2, L, r, alpha);
Ur0  = rom.Ur;
Uall = rom.U;

fprintf('orth(clean)   = %.2e\n', norm(Ur0'*W2*Ur0 - eye(r),'fro'));

Lr0 = Ur0' * W2 * L * Ur0;      % A0

%% ---------------- NOISY BASIS + ALIGNMENT ----------------
rng(seed);
Uw  = W * Uall;
UwN = Uw + sigma * randn(size(Uw));   % i.i.d. in WEIGHTED space
UN  = W \ UwN;

UrN = W2_orthonormalize(UN(:,1:r), W);
fprintf('orth(noisy)   = %.2e\n', norm(UrN'*W2*UrN - eye(r),'fro'));

LrN = UrN' * W2 * L * UrN;            % in UrN coords

% W2-Procrustes alignment (gauge fix)
G = Ur0' * W2 * UrN;
[U,~,V] = svd(G,'econ');
R = U*V';

UrNa = UrN * R';
fprintf('orth(aligned) = %.2e\n', norm(UrNa'*W2*UrNa - eye(r),'fro'));

LrNa = R * LrN * R';                  % aligned reduced generator

%% ---------------- DECOMPOSE LrNa = A0 + A1 + A2 + A3 ----------------
Neff = UrNa - Ur0;

A0 = Ur0'  * W2 * L * Ur0;
A1 = Neff' * W2 * L * Ur0;
A2 = Ur0'  * W2 * L * Neff;
A3 = Neff' * W2 * L * Neff;

fprintf('DECOMP CHECK: ||LrNa - (A0+A1+A2+A3)||F = %.3e\n', norm(LrNa-(A0+A1+A2+A3),'fro'));

% print term magnitudes
n0 = norm(A0,'fro'); n1 = norm(A1,'fro'); n2 = norm(A2,'fro'); n3 = norm(A3,'fro');
fprintf('||A0||F=%.3e  ||A1||F=%.3e  ||A2||F=%.3e  ||A3||F=%.3e\n', n0,n1,n2,n3);
fprintf('rel: A1/A0=%.3e  A2/A0=%.3e  A3/A0=%.3e\n', n1/max(n0,eps), n2/max(n0,eps), n3/max(n0,eps));

%% ---------------- CASES ----------------
Mats  = { A0, A0+A1, A0+A2, A0+A3, A0+A1+A2, LrNa };
names = { 'A0 (clean)', 'A0+A1', 'A0+A2', 'A0+A3', 'A0+A1+A2', 'LrNa (full noisy)' };

%% ---------------- ZOOMED GRID AROUND RIGHTMOST EIGENVALUES ----------------
%% ---------------- ZOOMED GRID (HARD CLAMP to rightmost band) ----------------
allLam = [];
for j=1:numel(Mats)
    allLam = [allLam; eig(Mats{j})]; %#ok<AGROW>
end

maxRe = max(real(allLam));

% Hard clamp: only consider eigenvalues with Re >= maxRe - focusBand
% This guarantees we never include the -200 junk.
lam_focus = allLam(real(allLam) >= maxRe - focusBand);

% If still too few points, relax band gradually (but never to allLam)
if numel(lam_focus) < 5
    lam_focus = allLam(real(allLam) >= maxRe - 5*focusBand);
end
if numel(lam_focus) < 5
    lam_focus = allLam(real(allLam) >= maxRe - 20*focusBand);
end

% If STILL too few, just use a fixed window around the rightmost eigenvalue
if numel(lam_focus) < 5
    lam_focus = maxRe + 1i*imag(allLam(abs(real(allLam)-maxRe)==min(abs(real(allLam)-maxRe))));
end

% Bounding box of focus
xr = [min(real(lam_focus)) max(real(lam_focus))];
yi = [min(imag(lam_focus)) max(imag(lam_focus))];

% Enforce minimum spans so it doesn't look squished
dx = max(xr(2)-xr(1), 0.5);   % try 0.2 / 0.5 / 1.0
dy = max(yi(2)-yi(1), 0.5);

xc = mean(xr); yc = mean(yi);

x = linspace(xc - (1+padFrac)*dx/2, xc + (1+padFrac)*dx/2, ngrid);
y = linspace(yc - (1+padFrac)*dy/2, yc + (1+padFrac)*dy/2, ngrid);

[X,Y] = meshgrid(x,y);
Z = X + 1i*Y;

fprintf('maxRe = %.6g\n', maxRe);
fprintf('focusBand = %.3g\n', focusBand);
fprintf('Zoom window: Re in [%.3g, %.3g], Im in [%.3g, %.3g]\n', ...
    min(x), max(x), min(y), max(y));
%% ---------------- AUTO CONTOUR LEVELS (from A0 data) ----------------
smin0 = sigma_min_grid(A0, Z);
s0 = smin0(:);
s0 = s0(isfinite(s0) & s0>0);

% pick a useful log-range for sigma_min; adjust quantiles if needed
lo = quantile(s0, 0.01);
hi = quantile(s0, 0.20);

lo = max(lo, 1e-16);
hi = max(hi, 10*lo);

levels_smin = logspace(log10(lo), log10(hi), nlev);

fprintf('AUTO sigma_min levels: [%.2e ... %.2e]\n', levels_smin(1), levels_smin(end));

% We'll plot log10(resolvent norm) = log10(1/sigma_min)
levels_logR = log10(1./levels_smin);  % increasing with sensitivity

%% ---------------- PLOT: log10 ||(zI-A)^(-1)|| contours (tiled) ----------------
figure('Color','w');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
% lock limits from the zoom grid
xlims = [min(X(:)) max(X(:))];
ylims = [min(Y(:)) max(Y(:))];
for j=1:numel(Mats)
    Atest = Mats{j};

    smin = sigma_min_grid(Atest, Z);
    Rlog = log10(1 ./ max(smin, 1e-16));  % log10 resolvent norm

    nexttile; hold on; grid on;
    contour(X,Y,Rlog,levels_logR,'LineWidth',1.1);
    plot(real(eig(Atest)), imag(eig(Atest)), 'k.', 'MarkerSize',10);

    xlabel('Re(z)'); ylabel('Im(z)');
    title(names{j});
      % --- IMPORTANT: force axis limits to the zoom window
    xlim(xlims);
    ylim(ylims);
end

sgtitle('Pseudospectra via log_{10}||(zI-A)^{-1}||_2 (computed from \sigma_{min})');

%%

%% ================= LOCAL HELPERS =================
function smin = sigma_min_grid(A, Z)
% sigma_min(zI - A) on grid Z (r small -> full svd is fine)
r = size(A,1);
I = eye(r);
smin = zeros(size(Z));
for k = 1:numel(Z)
    smin(k) = min(svd(Z(k)*I - A));
end
end