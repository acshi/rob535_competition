%load 66s.mat
%states_ic = X(:,1:6);
%inputs_ic = X(:,7:8);
%inputs_ic(:,2) = 1000*inputs_ic(:,2);

nStates = 3;
states_ic = zeros(3, nStates);
states_ic(1,:) = 0:nStates-1;
inputs_ic = zeros(1, nStates);
lb = -inf;%-0.1;
ub = inf;%0.1;
u_step = 0.01;
[states, inputs] = mk_opt(states_ic, inputs_ic, lb, ub, u_step, @integrate, @objective, @constrain);

function [x1, A, B] = integrate(x0, u0)
x1 = x0 + [cos(x0(3)); sin(x0(3)); u0];
A = eye(3) + [0,0,-sin(x0(3));0,0,cos(x0(3));0,0,0];
B = [0;0;1];%-sin(u0); cos(u0)];
end

function [C, dC] = constrain(state)
C = -(state(1)^2 + (state(2) - 60)^2 - 50^2);
dC = [-2*state(1); -2*(state(2) - 60); 0];
end

function [obj, dobj] = objective(states)
    obj = states(2,size(states,2));
    dobj = zeros(size(states));
    dobj(2,size(states,2)) = 1;
end