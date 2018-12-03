load TestTrack.mat;
XObs = generateRandomObstacles(20, TestTrack);
[U, X] = potential_fields(TestTrack, XObs);

figure;
hold on;
plot(X(:,2));
plot(U(:,2)/max(U(2,:)));

U = repelem(U, 10, 1);
[Y, T] = forwardIntegrateControlInput(U);
figure;
hold on;
plot(X(:,1), X(:,3), '.');
plot(Y(:,1), Y(:,3), '.');
plot(TestTrack.bl(1,:), TestTrack.bl(2,:))
plot(TestTrack.br(1,:), TestTrack.br(2,:))
for i=1:numel(XObs)
    ob = [XObs{i}; XObs{i}(1,:)];
    plot(ob(:,1),ob(:,2));
end
info = getTrajectoryInfo(Y, U, []);
