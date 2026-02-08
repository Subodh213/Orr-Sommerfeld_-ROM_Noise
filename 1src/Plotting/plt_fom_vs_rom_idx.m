function plt_fom_vs_rom_idx(t, Qf, Qr, idx, titleStr)
%PLT_FOM_VS_ROM_IDX  Compare FOM vs ROM at idx over time (Re only) + rel error.
% t: (Kx1), Qf/Qr: (nxK), idx: integer

if nargin < 5, titleStr = 'FOM vs ROM (index trace)'; end

qf = Qf(idx,:).';
qr = Qr(idx,:).';

% relative error (magnitude), safe denom
eps0 = 1e-14;
err = abs(qr - qf) ./ max(abs(qf), eps0);

figure('Color','w');
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

nexttile; hold on; grid on;
plot(t, real(qf), 'LineWidth', 1.8, 'DisplayName','FOM Re');
plot(t, real(qr), '--', 'LineWidth', 1.8, 'DisplayName','ROM Re');
ylabel(sprintf('Re(q(idx=%d))', idx));
title(titleStr);
legend('Location','best');

nexttile; hold on; grid on;
plot(t, err, 'LineWidth', 1.6);
xlabel('time'); ylabel('rel error  |ROM-FOM|/max(|FOM|,eps)');
title('Relative error');
end
