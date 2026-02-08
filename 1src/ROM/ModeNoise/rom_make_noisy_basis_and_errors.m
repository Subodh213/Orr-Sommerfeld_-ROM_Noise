function out = rom_make_noisy_basis_and_errors(Qtrue, U_all, Ur_clean, W, W2, Phi, r, k0, sigma)
%ROM_MAKE_NOISY_BASIS_AND_ERRORS
% Adds i.i.d. noise with std=sigma in WEIGHTED space to modes U_all,
% re-orthonormalizes, and computes projection+model errors.

    % noise in weighted space => constant variance
    Uw = W * U_all;
    Uw_noisy = Uw + sigma * randn(size(Uw));
    U_noisy = W \ Uw_noisy;

    % basis from first r columns
    Ur_noisy = W2_orthonormalize(U_noisy(:,1:r), W);

    % errors
    [Eproj_clean, Emod_clean, rho_clean] = rom_errors_proj_model(Qtrue, Ur_clean, W2, Phi, k0);
    [Eproj_noisy, Emod_noisy, rho_noisy] = rom_errors_proj_model(Qtrue, Ur_noisy, W2, Phi, k0);

    out.Ur_noisy    = Ur_noisy;
    out.Eproj_clean = Eproj_clean;
    out.Emod_clean  = Emod_clean;
    out.rho_clean   = rho_clean;
    out.Eproj_noisy = Eproj_noisy;
    out.Emod_noisy  = Emod_noisy;
    out.rho_noisy   = rho_noisy;
end
