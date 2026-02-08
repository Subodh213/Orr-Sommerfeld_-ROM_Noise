function [M, A] = make_OS_matrices(Re, N, alpha, beta, meanInd, doPlots)
%MAKE_OS_MATRICES  Build Orr–Sommerfeld generalized EVP matrices.
%
%   [M, A] = make_OS_matrices(Re, N, alpha, beta, meanInd, doPlots)
%
%   Outputs:
%     A, M : matrices in the generalized eigenproblem  A v = omega M v
%
%   Inputs:
%     Re      : Reynolds number
%     N       : number of Chebyshev interior points (as used by OS_Op)
%     alpha   : streamwise wavenumber
%     beta    : spanwise wavenumber
%     meanInd : 2 = Poiseuille (parabolic), 1 = Couette (linear)
%     doPlots : 0/1 (optional), whether to plot eigenmode + spectrum
%
%   Requires:
%     OS_Op(alpha,beta,N,Re,meanInd) that returns [A,M,y]

    if nargin < 6 || isempty(doPlots)
        doPlots = 0;
    end

    % Build operator and mass matrices
    [A, M, y] = OS_Op(alpha, beta, N, Re, meanInd);

    if doPlots
        % Solve EVP
        [Evecs, EvalsMat] = eig(A, M);
        Evals = diag(EvalsMat);

        % Sort from least to most stable (largest imag first)
        [~, ind] = sort(imag(Evals), 'descend');
        Evals = Evals(ind);
        Evecs = Evecs(:, ind);

        % Plot least stable eigenmode (pad BCs for plotting)
        figure
        plot(y, abs([0; Evecs(:,1); 0]), 'k', 'LineWidth', 2); hold on
        plot(y, real([0; Evecs(:,1); 0]), 'k--', 'LineWidth', 2);
        xlabel('$y$', 'Interpreter', 'latex', 'FontSize', 24)
        ylabel('Eigenmode')

        % Plot spectrum (phase speed if alpha ~= 0)
        figure
        if alpha ~= 0
            plot(real(Evals)/alpha, imag(Evals)/alpha, 'kx'); hold on
            xlabel('$c_r$', 'Interpreter', 'latex', 'FontSize', 24)
            ylabel('$c_i$', 'Interpreter', 'latex', 'FontSize', 24)
            if meanInd == 2
                axis([0 1 -1 0.05])
            else
                axis([-1 1 -1 0.05])
            end
        else
            plot(real(Evals), imag(Evals), 'kx'); hold on
            xlabel('$\omega_r$', 'Interpreter', 'latex', 'FontSize', 24)
            ylabel('$\omega_i$', 'Interpreter', 'latex', 'FontSize', 24)
            axis([-0.5 0.5 -1 0.05])
        end
        plot([-1, 1], [0, 0], 'k:')
        title([' Orr-Sommerfeld, $\alpha = ', num2str(alpha), ...
               ', \beta = ', num2str(beta), '$, $N = ', num2str(N), '$'], ...
               'Interpreter', 'latex')
    end
end
