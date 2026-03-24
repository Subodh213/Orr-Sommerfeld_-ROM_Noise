function Qr = rom_roll(Qf, Ur, W2, Phi, k0)
% ROM rollout using projected propagator

K = size(Qf,2);
Phi_r = Ur' * W2 * Phi * Ur;

a = Ur' * W2 * Qf(:,k0);

Qr = zeros(size(Qf));
for k = 1:K
    Qr(:,k) = Ur * a;
    a = Phi_r * a;
end
end
