function [d, grad_d] = rem_dist(X, TestTrack)
pos = [X(end-7); X(end - 5)];
dist = sum((pos - TestTrack.cline).^2,1);
[~, idx] = min(dist);

for i=[idx idx - 1]
    dir = (TestTrack.cline(:,i + 1) - TestTrack.cline(:,i));
    len = norm(dir);
    s = (pos - TestTrack.cline(:,i))'*dir/len;
    if s < 0
        continue;
    end
    d = (len - s).^2 + sum(sum((TestTrack.cline(:,i + 2:end) - TestTrack.cline(:,i+1:end-1)).^2));
    return;
end
disp('Error: need more robust rem_dist');
d = nan;

if isnan(d)
    if nargout < 2
        return;
    else
        grad_d = nan(size(x));
    end
end


grad_d = zeros(size(x));
v = 2*dir/len;
grad_d(end-7) = v(1);
grad_d(end-5) = v(2);
end