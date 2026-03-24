%% exp02_noisy_basis.m
% Clean driver script: build baseline ROM, add constant-variance mode noise,
% compute errors, and plot using existing plotting functions.

clear; close all; clc;
addpath(genpath('../1src'));

%% --- settings ---
Re=1000; N=208; alpha=1; beta=0; meanInd=2;
dt=1e-2; K=5000;
r=25; k0=1;

s = 1e-3;   % <-- THIS is your constant std in weighted space (variance = sigma^2)
rng(1);

%% --- results folder ---
ts = datestr(now,'yyyy-mm-dd_HHMMSS');
OUT_DIR = fullfile('..','results','exp02_noisy_basis',ts);
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

%% --- OS + propagator ---
[A,M,y] = OS_Op(alpha,beta,N,Re,meanInd);
n = size(A,1);

L   = -1i*(M\A);
Phi = expm(dt*L);

w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

%% --- truth snapshots ---
q0 = randn(n,1); q0=q0/norm(q0);
Qtrue = zeros(n,K);
q=q0;
for k=1:K
    q = Phi*q;
    Qtrue(:,k)=q;
end
t = (0:K-1)'*dt;
idx = round(size(Qtrue,1)/2);   % middle interior point
%% --- clean ROM ---
rom = rom_build_clean(Qtrue, W, W2, L, r, alpha);
Ur_clean = rom.Ur;     % basis
U_all    = rom.U;      % all modes

%% --- noisy basis + errors (ONE call) ---
out = rom_make_noisy_basis_and_errors(Qtrue, U_all, Ur_clean, W, W2, Phi, r, k0, s);
Ur_noisy = out.Ur_noisy;
fprintf('rho(clean)=%.6g, rho(noisy)=%.6g  (sigma=%.0e)\n', out.rho_clean, out.rho_noisy, s);

%% --- plots using your existing functions ---
plt_svals(Qtrue, W, 60, 1);
exportgraphics(gcf, fullfile(OUT_DIR,'fig_svals.png'));

plt_proj_model_errors(t, out.Eproj_clean, out.Emod_clean, out.Eproj_noisy, out.Emod_noisy, ...
                      sprintf('\\sigma=%.0e', s), out.rho_noisy);
exportgraphics(gcf, fullfile(OUT_DIR,'fig_errors.png'));

Qr_clean = rom_roll(Qtrue, Ur_clean, W2, Phi, k0);
Qr_noisy = rom_roll(Qtrue, Ur_noisy, W2, Phi, k0);

plt_fom_vs_rom_idx(t, Qtrue, Qr_clean, idx, 'FOM vs CLEAN ROM');
plt_fom_vs_rom_idx(t, Qtrue, Qr_noisy, idx, sprintf('FOM vs NOISY ROM (\\sigma=%.0e)', s));

[Ep_c, Em_c, rho_c] = rom_errors_proj_model(Qtrue, Ur_clean, W2, Phi, k0);
[Ep_n, Em_n, rho_n] = rom_errors_proj_model(Qtrue, Ur_noisy, W2, Phi, k0);

plt_proj_model_errors(t, Ep_c, Em_c, Ep_n, Em_n, sprintf('\\sigma=%.0e', s), rho_n);
plt_svals(Qtrue, W, 60, 1);


fprintf('Saved to: %s\n', OUT_DIR);
