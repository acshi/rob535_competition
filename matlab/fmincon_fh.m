function X = fmincon_fh(x_ic, dt, TestTrack, XObs, obs)
horizon = 90;
take = 10;
X = zeros(1,8);
% figure;
% hold on;
X_ic = potential_fields_euler_cons(x_ic, horizon, TestTrack, XObs);
obj_fun =  @(x) fmincon_fh_obj(x, TestTrack, obs);

figure;
hold on;
plot(TestTrack.bl(1,:), TestTrack.bl(2,:));
plot(TestTrack.br(1,:), TestTrack.br(2,:));
plot_x = X_ic(:,1);
plot_y = X_ic(:,3);
h = plot(plot_x, plot_y, '.');
h.XDataSource = 'plot_x';
h.YDataSource = 'plot_y';
drawnow;

    
for i=1:take:1000
    disp(i);
    X_fh = fmincon_full_state(X_ic, dt, x_ic, TestTrack, obs, obj_fun);
    for j=0:take-1
        X(i+j,:) = X_fh(8*j+1:8*j+8);
    end
    x_ic = X_fh(8*take + 1:8*take+6)';

    X_ic = zeros(horizon + 1, 8);
    X_ic(1:horizon + 1 - take,:) = reshape(X_fh(8*take+1:end), 8, [])';
    end_controls = X_ic(horizon - take,7:8);
    X_ic(horizon-take + 1:horizon+1,7:8) = repelem(end_controls, 11, 1);
    for j=1:take
        X_ic(horizon-take+j+1,1:6) = X_ic(horizon-take+j,1:6) + dt*nonl_bike(X_ic(horizon-take+j,1:8))';
    end
  
%     plot(X_ic(:,1), X_ic(:,3), '.');
%     plot(X_fh(1:8:end), X_fh(3:8:end), '.');
%     plot(TestTrack.bl(1,:), TestTrack.bl(2,:));
%     plot(TestTrack.br(1,:), TestTrack.br(2,:));
    plot_x = X_fh(1:8:end);
    plot_y = X_fh(3:8:end);
    refreshdata(h, 'caller');
    drawnow;
    
    pos = [X_fh(end-7); X_fh(end - 5)];
    end_line = TestTrack.br(:,end) - TestTrack.bl(:,end);
    offset = pos - TestTrack.bl(:,end);
    
    if end_line(1)*offset(2) - end_line(2)*offset(1) > 0
        for j = 1:horizon
            X(i+j,:) = X_fh(8*j + 1:8*j + 8);
        end
        break;
    end
end
end