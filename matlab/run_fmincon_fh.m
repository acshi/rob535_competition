if ~exist('TestTrack', 'var')
    load TestTrack.mat;
    XObs = generateRandomObstacles(20, TestTrack);
    %[U_pf, X_pf] = potential_fields_euler_cons(TestTrack);
end

obs = obs_seg(TestTrack, []);
if false
    [xs, ys] = meshgrid(210:260, -35:10);
    states = zeros(8*numel(xs), 1);
    states(1:8:end) = xs(:);
    states(3:8:end) = ys(:);
    d = dist_to_obj(states, obs);
    hold on;
    plot(TestTrack.bl(1,:), TestTrack.bl(2,:))
    plot(TestTrack.br(1,:), TestTrack.br(2,:))
    surf(xs, ys, reshape(d, size(xs)));
end

X_ic = [287,5,-176,0,2,0];
dt = 0.1;
tic;
X = fmincon_fh(X_ic, dt, TestTrack, [], obs);
fprintf('Opt took %f\n', toc);
figure;
hold on;
plot(X(:,1), X(:,3));
plot(TestTrack.bl(1,:), TestTrack.bl(2,:));
plot(TestTrack.br(1,:), TestTrack.br(2,:));

figure;
hold on;
plot(X(:,2));
plot(X(:,4));
plot(X(:,6)*10);
