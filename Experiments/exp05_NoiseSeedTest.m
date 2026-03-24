%% exp_seedAveraging_modeNoise_FULL.m
% Purpose:
%   Test PI's "averaging" claim properly for MODE noise:
%   - Fix N, sigma, r, dt, K
%   - Add i.i.d. noise to MODES (in WEIGHTED space)
%   - Increase number of seeds S
%   - Check if mean perturbation in reduced operator becomes:
%       (i) off-diagonal -> 0
%       (ii) close to scalar identity
%   - Also show that off-diagonal fluctuations plateau (do not vanish)
%
% Requires on path:
%   OS_Op(alpha,beta,N,Re,meanInd)
%   clencurt_WeightsB(N)
%   W2_orthonormalize(U, W)

clear; close all; clc;
addpath(genpath('../1src'));

%% -------------------- SETTINGS --------------------
Re      = 1000;
alpha   = 1;
beta    = 0;
meanInd = 2;

N  = 128;        % <-- choose ONE N for the averaging test
dt = 1e-2;
K  = 1000;       % snapshot length
r  = 10;         % number of modes used

sigma = 1e-3;    % FIXED mode-noise std (WEIGHTED space)

seedList = [2 5 10 20 50 100 200 300 500];  % increasing realizations = true averaging

% output folder
ts  = datestr(now,'yyyy-mm-dd_HHMMSS');
OUT = fullfile('..','results','exp_seedAveraging_modeNoise',ts);
if ~exist(OUT,'dir'), mkdir(OUT); end
fprintf('Saving to: %s\n', OUT);

%% -------------------- BUILD FOM + CLEAN BASIS --------------------
% OS operator
[A,M,~] = OS_Op(alpha,beta,N,Re,meanInd);
n = size(A,1);

L   = -1i*(M\A);
Phi = expm(dt*L);

% weights
w  = clencurt_WeightsB(N);
W  = diag(sqrt(w));
W2 = diag(w);

% FOM snapshots (clean) -- deterministic
rng(1);
q0 = randn(n,1);

% (Optional) normalize in W-norm (recommended for cross-N fairness)
q0 = q0 / sqrt(real(q0'*(W2*q0)));

Qf = zeros(n,K);
q = q0;
for k=1:K
    q = Phi*q;
    Qf(:,k) = q;
end

% Clean POD modes from clean snapshots (weighted SVD)
Xw = W*Qf;
[Uw,~,~] = svd(Xw,'econ');
Uall = W\Uw;

Ur0 = W2_orthonormalize(Uall(:,1:r), W);
Lr0 = Ur0' * W2 * L * Ur0;

fprintf('Built clean ROM: N=%d, n=%d, r=%d\n', N, n, r);

%% -------------------- SEED-AVERAGING TEST --------------------
offMean = zeros(numel(seedList),1);   % ||offdiag(meanΔ)||_F
offStd  = zeros(numel(seedList),1);   % std of offdiag entries across seeds
idFit   = zeros(numel(seedList),1);   % ||meanΔ - cI|| / ||meanΔ||
diagFit = zeros(numel(seedList),1);   % ||offdiag(meanΔ)|| / ||meanΔ||

offdiag_only = @(X) X - diag(diag(X));

for iS = 1:numel(seedList)
    S = seedList(iS);

    dLs = zeros(r,r,S);

    for s = 1:S
        rng(1000 + s);

        % ---- MODE noise in WEIGHTED space ----
        UwN = (W*Uall) + sigma * randn(size(W*Uall));  % noisy weighted modes
        UN  = W\UwN;                                   % unweighted
        UrN = W2_orthonormalize(UN(:,1:r), W);         % W-orthonormal basis

        LrN = UrN' * W2 * L * UrN;
        dLs(:,:,s) = LrN - Lr0;
    end

    mean_dL = mean(dLs,3);

    % (A) off-diagonal of the MEAN
    offPart = offdiag_only(mean_dL);
    offMean(iS) = norm(offPart,'fro');

    % relative off-diagonal ratio of the mean (more interpretable)
    diagFit(iS) = norm(offPart,'fro') / max(norm(mean_dL,'fro'), eps);

    % (B) identity-fit of the MEAN
    c = trace(mean_dL)/r;
    idFit(iS) = norm(mean_dL - c*eye(r),'fro') / max(norm(mean_dL,'fro'), eps);

    % (C) std of off-diagonal entries across seeds (fluctuations)
    vals = zeros(r*(r-1), S);
    idx = 0;
    for i=1:r
        for j=1:r
            if i~=j
                idx = idx + 1;
                vals(idx,:) = squeeze(dLs(i,j,:));
            end
        end
    end
    offStd(iS) = std(vals(:));

    fprintf('S=%4d | ||offdiag(meanΔ)||=%.3e | offRatio=%.3e | idFit=%.3e | offStd=%.3e\n', ...
        S, offMean(iS), diagFit(iS), idFit(iS), offStd(iS));
end

%% -------------------- PLOTS --------------------
% Plot 1: off-diagonal of mean vs seeds (should drop ~ 1/sqrt(S) if E[off]=0)
figure('Color','w');
loglog(seedList, offMean, 'o-','LineWidth',1.6); grid on;
xlabel('Number of seeds S');
ylabel('||offdiag(mean(\Delta L_r))||_F');
title(sprintf('Mode-noise averaging: offdiag of mean vs seeds (N=%d, r=%d, \\sigma=%.1e)', N, r, sigma));
exportgraphics(gcf, fullfile(OUT,'offdiag_mean_vs_seeds.png'));

% Plot 2: relative identity fit vs seeds (should drop if mean -> cI)
figure('Color','w');
semilogx(seedList, idFit, 's-','LineWidth',1.6); grid on;
xlabel('Number of seeds S');
ylabel('||mean(\Delta L_r) - cI||_F / ||mean(\Delta L_r)||_F');
title('Does the mean bias approach scalar identity?');
exportgraphics(gcf, fullfile(OUT,'idFit_vs_seeds.png'));

% Plot 3: off-diagonal fluctuations vs seeds (should plateau)
figure('Color','w');
semilogx(seedList, offStd, '^-','LineWidth',1.6); grid on;
xlabel('Number of seeds S');
ylabel('std( offdiag entries of \Delta L_r )');
title('Off-diagonal fluctuations across seeds (should plateau)');
exportgraphics(gcf, fullfile(OUT,'offdiag_fluct_std_vs_seeds.png'));

%% -------------------- SAVE RESULTS --------------------
T = table(seedList(:), offMean, diagFit, idFit, offStd, ...
          'VariableNames', {'Seeds','offMean','offRatioMean','idFit','offStd'});
disp(T);

save(fullfile(OUT,'results.mat'), 'T','seedList','N','Re','alpha','beta','meanInd','dt','K','r','sigma');
writetable(T, fullfile(OUT,'results.csv'));

fprintf('\nDone. Outputs saved in:\n  %s\n', OUT);
