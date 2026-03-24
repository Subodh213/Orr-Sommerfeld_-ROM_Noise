function rom = rom_build_pg(Qtrial, Qtest, W, W2, L, r, alpha)
%ROM_BUILD_PG  Petrov–Galerkin ROM using two datasets:
%   trial space  from Qtrial
%   test space   from Qtest
%
% Enforces biorthogonality:
%   Vr' * W2 * Ur = I
%
% Inputs
%   Qtrial : (n x K) snapshot matrix for TRIAL space
%   Qtest  : (n x K) snapshot matrix for TEST  space
%   W      : diag(sqrt(w))
%   W2     : diag(w)
%   L      : full generator
%   r      : ROM dimension
%   alpha  : scalar for c-plane mapping: c = (1i*lambda)/alpha
%
% Output struct "rom"
%   rom.Utrial   : (n x k) POD modes (unweighted) from Qtrial
%   rom.Vtest    : (n x k) POD modes (unweighted) from Qtest
%   rom.Ur       : (n x r) trial basis (W2-orthonormal)
%   rom.Vr       : (n x r) test  basis (biorthonormal to Ur)
%   rom.Lr       : (r x r) reduced generator Vr' * W2 * L * Ur
%   rom.c        : mapped eigenvalues of Lr
%   rom.Veig     : eigenvectors of Lr
%   rom.c_sort   : eigenvalues sorted by imag(c) desc
%   rom.Veig_sort: eigenvectors sorted consistently
%   rom.orthErrU : ||Ur' W2 Ur - I||_F
%   rom.orthErrV : ||Vr' W2 Vr - I||_F
%   rom.bioErr   : ||Vr' W2 Ur - I||_F

    % -------------------- TRIAL POD --------------------
    Xw_tr = W * Qtrial;
    [Uw_tr, ~, ~] = svd(Xw_tr, 'econ');
    Utrial = W \ Uw_tr;                 % unweighted trial POD modes
    rom.Utrial = Utrial;

    % Build W2-orthonormal trial basis (like your clean ROM)
    Ur = W2_orthonormalize(Utrial(:,1:r), W);
    rom.Ur = Ur;
    rom.orthErrU = norm(Ur' * W2 * Ur - eye(r), 'fro');

    % -------------------- TEST POD --------------------
    Xw_te = W * Qtest;
    [Uw_te, ~, ~] = svd(Xw_te, 'econ');
    Vtest = W \ Uw_te;                  % unweighted test POD modes
    rom.Vtest = Vtest;

    % Take first r test modes, then biorthonormalize to Ur
    V0 = Vtest(:,1:r);

    % Optional: W2-orthonormalize V0 first (helps conditioning, not required)
    V0 = W2_orthonormalize(V0, W);

    % Enforce biorthogonality: Vr' * W2 * Ur = I
    M = V0' * W2 * Ur;                  % (r x r)
    % Vr = V0 * inv(M')  ==> Vr' W2 Ur = I
    Vr = V0 / (M');                     % right-division is stable-ish
    rom.Vr = Vr;

    rom.orthErrV = norm(Vr' * W2 * Vr - eye(r), 'fro');
    rom.bioErr   = norm(Vr' * W2 * Ur - eye(r), 'fro');

    % -------------------- PETROV–GALERKIN REDUCED GENERATOR --------------------
    Lr = Vr' * W2 * L * Ur;
    rom.Lr = Lr;

    % -------------------- Reduced eigendata (mapped to c-plane) --------------------
    [Veig, Dr] = eig(Lr);
    lam = diag(Dr);

    c = (1i * lam) / alpha;
    rom.Veig = Veig;
    rom.c    = c(:);

    [~, ord] = sort(imag(c), 'descend');
    rom.c_sort    = c(ord);
    rom.Veig_sort = Veig(:, ord);
end