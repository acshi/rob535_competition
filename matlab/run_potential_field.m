load TestTrack.mat;
XObs = generateRandomObstacles(20, TestTrack);
tic;

obs = zeros(8*numel(XObs), 1);
for i=1:numel(XObs)
    for j=1:4
        obs(8*i + 2*j - 9) = XObs{i}(j,1);
        obs(8*i + 2*j - 8) = XObs{i}(j,2);
    end
end

[U_short] = potential_field(TestTrack.bl, TestTrack.br, TestTrack.cline, TestTrack.theta, obs);
toc

%figure;
%hold on;
%plot(X(:,2));
%plot(U_short(:,2)/max(U_short(2,:)));


U = repelem(U_short', 10, 1);
[Y, T] = forwardIntegrateControlInput2(U);
figure;
hold on;
%plot(X(:,1), X(:,3), '.');
plot(Y(:,1), Y(:,3), '.');
plot(TestTrack.bl(1,:), TestTrack.bl(2,:))
plot(TestTrack.br(1,:), TestTrack.br(2,:))
for i=1:numel(XObs)
    ob = [XObs{i}; XObs{i}(1,:)];
    plot(ob(:,1),ob(:,2));
end
info = getTrajectoryInfo(Y, U, []);
disp(info);

