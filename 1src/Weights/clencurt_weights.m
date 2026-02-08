function w = clencurt_weights(N)
% w = clencurt_weights(N)
% N+2 = Ninternal nodes + 2 boundary nodes
% Nodes: x_j = cos(pi*j/N), j=0..N
% Weights satisfy: sum(w) = 2  (integral over [-1,1])
N=N+2; % added boundary nodes
theta = (0:N-1)' * pi / (N-1);
w = zeros(N,1);

if mod(N,2)==0
    % N even
    for k = 0:N-1
        s = 1;
        for m = 1:(N/2-1)
            s = s + 2*cos(2*m*theta(k)) / (1 - (2*m)^2);
        end
        s = s + cos(N*theta(k)) / (1 - N^2);   % no /2 here
        w(k+1) = (2/N) * s;
    end
else
    % N odd
    for k = 0:N-1
        s = 1;
        for m = 1:((N-1)/2)
            s = s + 2*cos(2*m*theta(k)) / (1 - (2*m)^2);
        end
        w(k+1) = (2/N) * s;
    end
end

w = w(2:N-1);

end
