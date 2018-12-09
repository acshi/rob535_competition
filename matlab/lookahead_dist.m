function [d, grad] = lookahead_dist(X, TestTrack)
pos = [X(end-7); X(end - 5)];
n = size(TestTrack.cline, 2);
dist = sum((pos - TestTrack.cline).^2,1);
[~, idx] = min(dist);
idx = min(idx + 5, n);
v = pos - TestTrack.cline(:,idx);

d = sum(v.^2) + sum(sum((TestTrack.cline(:,idx + 6:end) - TestTrack.cline(:,idx+5:end-1)).^2));

if nargout > 1
    grad = zeros(size(X));
    grad(end - 7) = 2*v(1);
    grad(end - 5) = 2*v(2);
end
end