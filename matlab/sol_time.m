function [t, grad] = sol_time(X, TestTrack, dt)
t = 0;
grad = zeros(size(X));
return;
%tic;

if nargin == 2
    dt = 1;
end

pos = [X(1:8:end)'; X(3:8:end)'];
d = sum((pos - TestTrack.cline(:,end)).^2);
[~, idx] = min(d);

for i = [idx,idx - 1]
    if i >= size(pos, 2)
        continue
    end
    p0 = pos(:, i);
    p1 = pos(:, i + 1);
    bl = TestTrack.bl(:, end);
    br = TestTrack.br(:, end);
    q_p = [(p1 - p0) (bl - br)]\(bl - p0);
    
    if q_p(1) > 0 && q_p(1) < norm(p0 - p1) && q_p(2) > 0 && q_p(2) < norm(br - bl)
        t = (i + q_p(1)/norm(p0 - p1))*dt;
        dp0 = [(p1 - p0) (bl - br)]\((q_p(1) - 1)*eye(2));
        grad = sparse(size(X, 1), 1);
        L = norm(p1 - p0);
        grad(8*i - 7) = dt*(dp0(1,1)/L + q_p(1)*(p1(1) - p0(1))/L^3);
        grad(8*i - 5) = dt*(dp0(1,2)/L + q_p(1)*(p1(2) - p0(2))/L^3);
        dp1 = [(p1 - p0) (bl - br)]\(-q_p(1)*eye(2));
        grad(8*(i + 1) - 7) = dt*(dp1(1,1)/L - q_p(1)*(p1(1) - p0(1))/L^3);
        grad(8*(i + 1) - 5) = dt*(dp1(1,2)/L - q_p(1)*(p1(2) - p0(2))/L^3);
        %fprintf('sol_time: %f\n', toc);
        return;
    end
end
disp('Error: need more robust sol_time')
t = nan;
grad = nan(size(X));
end