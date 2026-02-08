function [L_OS,M,y] = OS_Op(kx,kz,N,Re,meanInd)

% Numerical formulation of Orr-Sommerfeld operator
% This formulation uses a Chebyshev polynomials to construct the
% differentiation operators, and uses a trick to implicitly enforce the
% boundary conditions. You don't need to worry too much about the details
% to use and modify the code, however.
        
[~,D4]=cheb4c(N+2); % Incorporates clamped boundary conditions

k2 = kx*kx + kz*kz;
k4 = k2*k2;

I = eye(N);
Z = zeros(N,N);

[y,DM] = chebdif(N+2,2); 

D1 = DM(2:N+1,2:N+1,1); % Incorporate homogeneous BC
D2 = DM(2:N+1,2:N+1,2);

if meanInd == 1
    U0 = y;
elseif meanInd == 2
    U0 = 1-y.^2;
else
    error('meanInd must be 1 or 2')
end

U0y = DM(:,:,1)*U0;
U0yy = DM(:,:,2)*U0;

U0 = U0(2:N+1);
U0y = U0y(2:N+1);
U0yy = U0yy(2:N+1);


Delta = (D2 - k2*I);
Delta2 = (D4 - 2*k2*D2 + k4*I);

L_OS = Delta\(1i*kx*(diag(U0yy) - diag(U0)*Delta) + 1/Re*Delta2);
M = -1i*eye(N);

end


