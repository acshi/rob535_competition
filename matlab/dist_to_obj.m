function [d, dd] = dist_to_obj(X, obs, tol)

if nargin < 3
    tol = 0;
end
%tic;
% Actually distance squared.
d = zeros(size(X, 1)/8, 1);
dd = spalloc(size(X,1)/8, size(X,1), 2*size(X,1)/8);
for i=1:size(X, 1)/8
    pos = [X(8*i - 7); X(8*i - 5)];
    s = sum((pos - obs.p0).*obs.dir, 1);
    s(s > obs.len) = obs.len(s > obs.len);
    s(s < 0) = 0;
    [d(i), idx] = min(sum((s.*obs.dir + obs.p0 - pos).^2, 1));
    offset = pos - obs.p0(:,idx);
    cross = offset(1)*obs.dir(2,idx) - offset(2)*obs.dir(1,idx);
    d(i) = sign(cross)*d(i) - tol*tol;
    ddi = -sign(cross)*2*(s(idx).*obs.dir(:,idx) + obs.p0(:,idx) - pos);
    dd(i, 8*i - 7) = ddi(1);
    dd(i, 8*i - 5) = ddi(2);
end
%fprintf('dist_to_obj: %f\n', toc);
end