function plt_svals(Q, W, rmax, normalizeFlag)
%PLT_SVALS  Plot singular values of weighted snapshots Xw = W*Q.
% Q: (nxK), W: (nxn), rmax: number of points to show, normalizeFlag: 1/0

if nargin < 3 || isempty(rmax), rmax = 50; end
if nargin < 4 || isempty(normalizeFlag), normalizeFlag = 1; end

s = svd(W*Q, 'econ');
rmax = min(rmax, numel(s));
s = s(1:rmax);

if normalizeFlag
    s = s / max(s(1), eps);
    ylab = 'sigma_i / sigma_1';
else
    ylab = 'sigma_i';
end

figure('Color','w'); grid on; hold on;
semilogy(1:rmax, s, 'o-','LineWidth',1.6);
xlabel('mode index'); ylabel(ylab);
title('Singular value spectrum (weighted snapshots)');
end
