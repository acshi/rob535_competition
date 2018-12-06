function [res] = obs_seg(TestTrack, XObs)
res = struct;
res.p0 = [TestTrack.br(:,2:end), TestTrack.bl(:,1:end - 1)];
diffs = [TestTrack.br(:,1:end - 1) - TestTrack.br(:,2:end),...
           TestTrack.bl(:,2:end) - TestTrack.bl(:,1:end - 1)];
res.len = sqrt(sum(diffs.^2, 1));
res.dir = normalize(diffs, 1, 'norm');
end
