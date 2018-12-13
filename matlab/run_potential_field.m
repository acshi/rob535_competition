load TestTrack.mat;
XObs = generateRandomObstacles(20, TestTrack);
tic;
[U_short, X] = potential_fields(TestTrack, XObs);
toc

figure;
hold on;
plot(X(:,2));
plot(U_short(:,2)/max(U_short(2,:)));

U = repelem(U_short, 10, 1);
[Y, T] = forwardIntegrateControlInput2(U);
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

