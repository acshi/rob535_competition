function [d, grad] = lookahead_dist(X, TestTrack)
pos = [X(end-7); X(end - 5)];

end_line = TestTrack.br(:,end) - TestTrack.bl(:,end);
offset = pos - TestTrack.bl(:,end);
if end_line(1)*offset(2) - end_line(2)*offset(1) > 0
    [d, grad] = sol_time(X, TestTrack);
    d = 10*(d - 200);
    grad = 10*grad;
    return;
end

n = size(TestTrack.cline, 2);
dist = sum((pos - TestTrack.cline).^2,1);
[~, idx] = min(dist);
idx = min(idx + 5, n);
if idx == n
    v = pos - (5*(TestTrack.cline(:,end) - TestTrack.cline(:,end - 1)) + TestTrack.cline(:,end));
else
    v = pos - TestTrack.cline(:,idx);
end

d = sum(v.^2) + sum(sum((TestTrack.cline(:,idx + 6:end) - TestTrack.cline(:,idx+5:end-1)).^2));

if nargout > 1
    grad = zeros(size(X));
    grad(end - 7) = 2*v(1);
    grad(end - 5) = 2*v(2);
end
end