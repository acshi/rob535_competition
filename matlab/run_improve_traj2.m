load 65s43d2.mat
load TestTrack.mat

obs = obs_seg(TestTrack, []);
tic;
X_imp = improve_traj(X, TestTrack, 0.1, obs,0.2,451);
toc
