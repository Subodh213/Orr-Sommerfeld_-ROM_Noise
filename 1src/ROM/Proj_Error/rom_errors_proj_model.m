function [Eproj, Emod, rhoPhi] = rom_errors_proj_model(Qtrue, Ur, W2, Phi, k0)
    K = size(Qtrue,2);
    wnorm = @(v) sqrt(real(v'*(W2*v)));

    % projection error
    Eproj = zeros(K,1);
    for k = 1:K
        qk = Qtrue(:,k);
        ak = Ur' * W2 * qk;
        qproj = Ur * ak;
        Eproj(k) = wnorm(qk - qproj) / max(wnorm(qk), eps);
    end

    % reduced propagator + stability
    Phi_r = Ur' * W2 * Phi * Ur;
    rhoPhi = max(abs(eig(Phi_r)));

    % model error
    Emod = zeros(K,1);
    a = Ur' * W2 * Qtrue(:,k0);
    for k = 1:K
        qk = Qtrue(:,k);
        qhat = Ur * a;
        Emod(k) = wnorm(qk - qhat) / max(wnorm(qk), eps);
        a = Phi_r * a;
    end
end
