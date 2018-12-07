function [d, g] = dist_to_target(X, target)
pos = [X(end-7); X(end - 5)];
v = (pos - target);
d = sum(v.^2);

if nargout == 1
    return;
end

g = zeros(size(X));
g(end - 7) = 2*v(1);
g(end - 5) = 2*v(2);
end