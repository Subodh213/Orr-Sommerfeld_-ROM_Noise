
function w = clencurt_WeightsB(N)

Nnodes  = N + 2; %Internal + 2 boundary nodes
N_c = Nnodes - 1; %Chebychev nodes 0,1,2,...,Nnodes-1 (total =Nnodes)

% expected: interior weights are positive and nonuniform
theta = (0:N_c)' * pi / N_c;   % theta = j*pi/N_c

%%

w_full = zeros(Nnodes,1); %weight vector,all nodes included

if mod(N_c,2) == 0 %Nc= 0,1,2,3,4 
    % N_c even
    %i is nodes counter
    %m is basis function counter
    for i = 1:N_c+1  %theta(i) = 0>1>2>3,...>Nc
        s = 1; %first mode T_0 = 1 (constant)
        for m = 1:(N_c/2 - 1) 
            %mode T_1 to T_(Nc/2 -1) (for fixed node)
            %note m->2m, this picks only even modes, since odd number
            %chebychev mode integrate to zero as per the chebychev
            %polynomial property.
            s = s + 2*cos(2*m*theta(i)) / (1 - (2*m)^2); 
        end
        s = s + cos(N_c*theta(i)) / (1 - N_c^2); %Last mode,T_Nc
        w_full(i) = (2/N_c) * s;
    end
else
    %  odd number of nodes
    for i = 1:N_c+1
        s = 1;
        for m = 1:((N_c-1)/2)
            s = s + 2*cos(2*m*theta(i)) / (1 - (2*m)^2);
        end
        w_full(i) = (2/N_c) * s;
    end
end
w = w_full(2:Nnodes-1);

end
