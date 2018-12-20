function X_ret = optimize_tail(X, TestTrack, dt, obs, tol)
horizon = 90;
x_ic = X(end-horizon, 1:6);
X_ic = X(end-horizon:end,:);
X_ic(:,8) = 2.5;

obj_fun =  @(x) sol_time(x, TestTrack, dt);
X_fh = fmincon_optimize_tail(X_ic, dt, x_ic, obs, obj_fun, tol);


pos = [X_fh(1:8:end-7)'; X_fh(3:8:end - 5)'];
end_line = TestTrack.br(:,end) - TestTrack.bl(:,end);
offset = pos - TestTrack.bl(:,end);
    
crossed = end_line(1)*offset(2,:) - end_line(2)*offset(1,:) > 0;
for i=1:numel(crossed)
    if crossed(i)
        X_fh2 = X_fh(1:8*i);
        break;
    end
end

X_ret = X(1:end-horizon-1,:);
X_ret = [X_ret; reshape(X_fh2, 8, [])'];
end