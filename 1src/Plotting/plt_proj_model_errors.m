function plt_proj_model_errors(t, Eproj_clean, Emod_clean, Eproj_rom, Emod_rom, sLabel, rhoPhi)
%PLT_PROJ_MODEL_ERRORS  Plot projection and model errors (clean vs ROM).
%
% Inputs:
%   t           : (Kx1) time vector
%   Eproj_clean : (Kx1)
%   Emod_clean  : (Kx1)
%   Eproj_rom   : (Kx1)
%   Emod_rom    : (Kx1)
%   sLabel      : string like 's=1e-6' (or '')
%   rhoPhi      : spectral radius of Phi_r (or [])

if nargin < 6, sLabel = ''; end

figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% Projection error
nexttile; hold on; grid on;
plot(t, Eproj_clean, 'k-', 'LineWidth', 2.0, 'DisplayName','Proj (clean)');
plot(t, Eproj_rom,   'r--','LineWidth', 1.6, 'DisplayName',['Proj (' sLabel ')']);
xlabel('time'); ylabel('projection error');
title('Projection error');
legend('Location','best');

% Model error
nexttile; hold on; grid on;
plot(t, Emod_clean, 'k-', 'LineWidth', 2.0, 'DisplayName','Model (clean)');
plot(t, Emod_rom,   'r--','LineWidth', 1.6, 'DisplayName',['Model (' sLabel ')']);
xlabel('time'); ylabel('model error');
if ~isempty(rhoPhi)
    title(sprintf('Model error (\\rho(Phi_r)=%.4g)', rhoPhi));
else
    title('Model error');
end
legend('Location','best');
end
