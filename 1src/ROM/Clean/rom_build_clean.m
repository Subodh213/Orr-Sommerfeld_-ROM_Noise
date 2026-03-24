function rom = rom_build_clean(Q, W, W2, L, r, alpha)
%ROM_BUILD_CLEAN  Weighted POD basis + reduced generator + eigendata (clean case).
%
% Inputs
%   Q      : (n x K) snapshot matrix (truth states)
%   W      : (n x n) diag(sqrt(w))  (sqrt-weights)
%   W2     : (n x n) diag(w)        (weights)
%   L      : (n x n) full generator (e.g. L = -1i*(M\A))
%   r      : ROM dimension
%   alpha  : scalar for c-plane mapping: c = (1i*lambda)/alpha
%
% Output struct "rom"
%   rom.U      : (n x k) POD modes (state space, unweighted)
%   rom.Ur     : (n x r) reduced basis with Ur' * W2 * Ur = I
%   rom.Lr     : (r x r) reduced generator Ur' * W2 * L * Ur
%   rom.c      : (r x 1) eigenvalues mapped to c-plane
%   rom.Vr     : (r x r) eigenvectors of rom.Lr
%   rom.c_sort : (r x 1) c sorted by imag(c) descending
%   rom.Vr_sort: (r x r) eigenvectors sorted consistently with c_sort
%   rom.orthErr: Fro norm of (Ur' * W2 * Ur - I)

    % --- Weighted POD (Hilbert POD) ---
    Xw = W * Q;                 % weighted snapshots
    [Uw, S, ~] = svd(Xw, 'econ');

    U = W \ Uw;                 % POD modes in state space
    rom.U = U;

    rom.s = diag(S);   % weighted singular values
    rom.S = S;         % optional full diagonal matrix

    % --- Build W2-orthonormal reduced basis ---
    Ur = W2_orthonormalize(U(:,1:r), W);
    rom.Ur = Ur;

    rom.orthErr = norm(Ur' * W2 * Ur - eye(r), 'fro');

    % --- Reduced generator ---
    Lr = Ur' * W2 * L * Ur;
    rom.Lr = Lr;

    % --- Reduced eigendata, mapped to c-plane ---
    [Vr, Dr] = eig(Lr);
    lam = diag(Dr);

    c = (1i * lam) / alpha;
    rom.Vr = Vr;
    rom.c  = c(:);

    % Sort by imag(c) descending (most unstable first)
    [~, ord] = sort(imag(c), 'descend');
    rom.c_sort  = c(ord);
    rom.Vr_sort = Vr(:, ord);
end
