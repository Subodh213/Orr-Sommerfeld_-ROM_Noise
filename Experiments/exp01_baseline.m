%% exp01_baseline.m
% Baseline experiment:
%  - Build OS operator + FOM spectrum in c-plane
%  - Build truth snapshots from full propagator Phi = expm(dt*L)
%  - Build CLEAN weighted POD ROM (Hilbert POD) basis + reduced generator
%  - Compare FOM vs ROM spectra, time trace at one index, singular values
%  - Compute projection + model errors for CLEAN basis

clear; close all; clc;

%% -------------------- PATHS --------------------
% Run this script from: experiments/
addpath(genpath('../1src'));

%% -------------------- SETTINGS --------------------
Re      = 1000;
N       = 208;         % Chebyshev total nodes including boundaries
alpha   = 1;
beta    = 0.0;
meanInd = 2;           % 2=Poiseuille, 1=Couette (based on your OS_Op)

dt = 1e-2;
K  = 5000;             % keep moderate for baseline; increase later

r  = 25;               % ROM dimension
k0 = 1;                % initial index for ROM evolution

rng(1);

%% -------------------- RESULTS FOLDER --------------------
EXP_NAME = 'exp01_baseline';
ts = datestr(now,'yyyy-mm-dd_HHMMSS');
OUT_DIR = fullfile('..','results',EXP_NAME,ts);
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

%% -------------------- BUILD OS OPERATORS --------------------
[A,M,y] = OS_Op(alpha,beta,N,Re,meanInd);   % A v = omega M v
n = size(A,1);
alpha_k = alpha;

% FOM spectrum (c-plane): c = omega/alpha
omega_fom = eig(A,M);
c_fom = omega_fom(:) / alpha_k;
c_fom = c_fom(isfinite(real(c_fom)) & isfinite(imag(c_fom)));

%% -------------------- BUILD GENERATOR + PROPAGATOR --------------------
L   = -1i*(M\A);        % time generator
Phi = expm(dt*L);       % discrete propagator

%% -------------------- TRUTH SNAPSHOTS --------------------
q0 = randn(n,1); q0 = q0 / norm(q0);

Qtrue = zeros(n,K);
q = q0;
for k = 1:K
    q = Phi*q;
    Qtrue(:,k) = q;
end
t = (0:K-1)'*dt;

%% -------------------- WEIGHTS --------------------
% Your function should return interior weights length n (=N-2)
w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

%% -------------------- CLEAN ROM BUILD --------------------
% Requires: src/rom/rom_build_clean.m and src/rom/W2_orthonormalize.m
rom = rom_build_clean(Qtrue, W, W2, L, r, alpha_k);

Ur = rom.Ur;
Lr = rom.Lr;

% ROM spectrum in c-plane: c = (i*lambda(Lr))/alpha
c_rom = rom.c;

fprintf('Clean basis orthErr (Ur''W2Ur - I): %.3e\n', rom.orthErr);
fprintf('FOM:  median|c|=%.3e, max|c|=%.3e\n', median(abs(c_fom)), max(abs(c_fom)));
fprintf('ROM:  median|c|=%.3e, max|c|=%.3e\n', median(abs(c_rom)), max(abs(c_rom)));

%% -------------------- CLEAN ERRORS (projection + model) --------------------
% Requires: src/rom/rom_errors_proj_model.m
Ur_clean = rom.Ur;   % clean reduced basis from rom_build_clean

% --- clean basis errors ---
[Eproj_clean, Emod_clean, rhoPhi_clean] = rom_errors_proj_model(Qtrue, Ur_clean, W2, Phi, k0);
fprintf('rho(Phi_r_clean) = %.6g\n', rhoPhi_clean);

% ROM/noisy errors  ✅ this creates Eproj_rom and Emod_rom
Ur_rom = rom.Ur;
[Eproj_rom, Emod_rom, rhoPhi_rom] = rom_errors_proj_model(Qtrue, Ur_rom, W2, Phi, k0);
fprintf('rho(Phi_r_rom) = %.6g\n', rhoPhi_rom);

%% -------------------- BUILD ROM RECONSTRUCTION FOR INDEX TRACE --------------------
% Evolve reduced coefficients using projected Phi_r
Phi_r = Ur' * W2 * Phi * Ur;

Arom = zeros(r,K);
Arom(:,1) = Ur' * W2 * Qtrue(:,k0);
for k = 2:K
    Arom(:,k) = Phi_r * Arom(:,k-1);
end
Qrom = Ur * Arom;   % (n x K) ROM reconstructed snapshots

%% -------------------- PLOTS --------------------

%% -------------------- PLOTS (simple driver calls) --------------------

% choose an index to track
idx = round(n/2);

% 1) FOM trace at idx (optional)
plt_signal_idx(t, Qtrue, idx, 'FOM');

% 2) FOM vs ROM trace at idx + error
plt_fom_vs_rom_idx(t, Qtrue, Qrom, idx, sprintf('FOM vs ROM at idx=%d', idx));

% 3) Singular values spectrum (weighted snapshots)
plt_svals(Qtrue, W, 60, 1);

% 4) Projection + model error comparison (clean vs ROM)

% (if you computed Eproj_clean, Emod_clean, and also Eproj_rom, Emod_rom)
plt_proj_model_errors(t, Eproj_clean, Emod_clean, Eproj_rom, Emod_rom,'',[]);

exportgraphics(gcf, fullfile(OUT_DIR,'fig5_errors_clean.png'));

%% -------------------- SAVE DATA --------------------
params = struct();
params.Re = Re; params.N = N; params.alpha = alpha; params.beta = beta; params.meanInd = meanInd;
params.dt = dt; params.K = K; params.r = r; params.k0 = k0; params.idx = idx;

save(fullfile(OUT_DIR,'params.mat'), 'params');

baseline = struct();
baseline.c_fom = c_fom;
baseline.c_rom = c_rom;
baseline.Eproj_clean = Eproj_clean;
baseline.Emod_clean  = Emod_clean;
baseline.rhoPhi_clean = rhoPhi_clean;
baseline.rom = rom;                 % includes Ur, Lr, etc.
% NOTE: Qtrue/Qrom can be huge; uncomment only if needed
% baseline.Qtrue = Qtrue;
% baseline.Qrom  = Qrom;

save(fullfile(OUT_DIR,'baseline.mat'), 'baseline', '-v7.3');

fprintf('\nEXP01 done. Results saved to:\n  %s\n', OUT_DIR);
