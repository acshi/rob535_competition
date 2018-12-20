function X = improve_traj(X, TestTrack, dt, obs, tol, start)
horizon = 90;
take = 10;
Fx_scale = 1000;
saved = 0;
for i=start:take:100000
    if i + horizon + 2 >= size(X,1)
        break
    end
    
    
    %X_ic = [X(i:i+horizon/2,:); X(i + horizon/2 + 2:i+horizon + 1,:)];
    X_ic = interp1((0:horizon+1)/(horizon+1), X(i:i+horizon+1,:), (0:horizon)/horizon);
    
    x_start = X(i,1:6);
    x_end = X(i+horizon+1,1:6);
    [X_fh, exitflag, output] = fmincon_improve_traj(X_ic, dt, x_start, x_end, obs, @zero, tol);
    if exitflag == 1
        saved = saved + 1;
    end
    fprintf("%d %d %.2e %d %d\n", i, exitflag, output.constrviolation, output.iterations, saved);
   	if exitflag ~= 1
        %continue
        [X_fh, exitflag,output] = fmincon_improve_traj(X(i:i+horizon+1,:), dt, x_start, x_end, obs, @zero, tol, 1000);
        if exitflag ~= 1
            disp('help help double fail');
            notAVar = notAVar + 1;
            return;
        end
    end
    X = [X(1:i-1,:) ; reshape(X_fh, 8, [])'; X(i+horizon+2:end,:)];
    X(i,1:6) = x_start;
    for j=0:take-1
        controls = X(i+j,7:8);
        controls(2) = Fx_scale*controls(2);
        state = X(i+j,1:6);
        X(i + j + 1,1:6) = rk4_integrate(state', controls', dt, 64);
    end
end

X = optimize_tail(X, TestTrack, dt, obs, tol);

end


function [Z, dZ] = zero(x)
Z = 0;
dZ = zeros(size(x));
end