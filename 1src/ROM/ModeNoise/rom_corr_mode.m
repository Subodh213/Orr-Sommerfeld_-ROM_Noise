function out = rom_corr_mode(AtildeN, trA, sigma)
%ROM_CORR_mode Trace-I correction: Atilde_corr = AtildeN - sigma^2*tr(A)*I
%
% Inputs:
%   AtildeN : (r x r) noisy reduced operator (already computed from noisy basis)
%   trA     : scalar trace(A) of the FULL operator A 
%   sigma   : noise std of entries of N 
%
% Output struct:
%   out.AN    : noisy reduced operator
%   out.bias  : sigma^2*trA*I
%   out.Acorr : corrected reduced operator
%   out.trA   : trace used
%   out.sigma : sigma used

r = size(AtildeN,1);
bias = (sigma^2) * trA * eye(r);

out.AN    = AtildeN;
out.bias  = bias;
out.Acorr = AtildeN - bias;
out.trA   = trA;
out.sigma = sigma;
end
