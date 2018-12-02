load TestTrack.mat;
XObs = generateRandomObstacles(1, TestTrack);
disp(TestTrack.bl(1:5));
U = potential_field(TestTrack.bl, TestTrack.br, TestTrack.cline, TestTrack.theta);
%U = U';
%U = repelem(U, 10, 1);
%disp(TestTrack.bl(1:5));
%[Y, T] = forwardIntegrateControlInput(U);
%hold on
%plot(Y(:,1), Y(:,3), '.');
%plot(TestTrack.bl(1,:), TestTrack.bl(2,:))
%plot(TestTrack.br(1,:), TestTrack.br(2,:))
%info = getTrajectoryInfo(Y, U, []);U