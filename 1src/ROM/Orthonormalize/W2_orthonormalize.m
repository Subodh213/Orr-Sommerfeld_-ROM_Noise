function Uo = W2_orthonormalize(U, W)
    % Enforce Uo' * W2 * Uo = I where W2 = W'*W
    Y = W * U;
    [Q,R] = qr(Y,0);
    for jj = 1:size(Q,2)
        if real(R(jj,jj)) < 0
            Q(:,jj) = -Q(:,jj);
            R(jj,:) = -R(jj,:);
        end
    end
    Uo = W \ Q;
end