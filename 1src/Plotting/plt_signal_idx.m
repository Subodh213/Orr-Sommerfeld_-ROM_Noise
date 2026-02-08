function plt_signal_idx(t, Q, idx, nameStr)
%PLT_SIGNAL_IDX  Plot Re/Im of Q(idx,:) vs time.
% t: (Kx1), Q: (nxK), idx: integer, nameStr: label string

if nargin < 4, nameStr = 'signal'; end
q = Q(idx,:).';

figure('Color','w'); hold on; grid on;
plot(t, real(q), 'LineWidth', 1.8, 'DisplayName', [nameStr ' Re']);
plot(t, imag(q), '--', 'LineWidth', 1.8, 'DisplayName', [nameStr ' Im']);
xlabel('time'); ylabel(sprintf('q(idx=%d)', idx));
title(sprintf('%s at idx=%d', nameStr, idx));
legend('Location','best');
end
