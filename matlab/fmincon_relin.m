function [U, X] = fmincon_relin(U_init, X_init, dt, X_ic, TestTrack, obs)
% TODO provide gradients
problem.solver = 'fmincon';
nSteps = size(X_init, 1);
problem.Aeq = [eye(6), zeros(6, 8*nSteps - 6)];
problem.Beq = X_ic;
problem.objective = @(x) sol_time(x, TestTrack, dt);
problem.nonlcon = @nonlcon;
problem.x0 = reshape([X_init, U_init]', [], 1);
problem.lb = -inf(size(problem.x0));
problem.ub = inf(size(problem.x0));
problem.lb(7:8:end) = -.5;
problem.ub(7:8:end) = .5;
problem.lb(8:8:end) = -5000;
problem.ub(8:8:end) = 2500;
problem.options = optimoptions('fmincon', 'MaxFunctionEvaluations', 2);
fmincon(problem);

    function [C, Ceq] = nonlcon(x)
        C = dist_to_obj(x, obs);
        Ceq = euler_cons(x, dt);
    end
end