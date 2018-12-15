x_ic = [287,5,-176,0,2,0];
%load fmincon_fh_90s_horizon_rk4_3.mat

figure;
hold on;

%X = reshape(X, 8, [])';
U1 = X(:,7:8);
U1(:,2) = 1000*U1(:,2);

tic;
X11 = zeros(size(U1,1) + 1, 6);
X11(1,:) = x_ic;
for i=1:size(U1,1)
    X11(i + 1,:) = rk4_integrate(X11(i,:)', U1(i,:)', 0.1, 1);
end
fprintf("rk4-1 took %f\n", toc);
plot(X11(:,1), X11(:,3), '.', 'DisplayName', 'rk4-1');
drawnow;

tic;
X4 = zeros(size(U1,1) + 1, 6);
X4(1,:) = x_ic;
for i=1:size(U1,1)
    X4(i + 1,:) = rk4_integrate(X4(i,:)', U1(i,:)', 0.1, 2);
end
fprintf("rk4-2 took %f\n", toc);
legend;
plot(X4(:,1), X4(:,3), '.', 'DisplayName', 'rk4-2');
drawnow;


tic;
X1 = zeros(size(U1,1) + 1, 6);
X1(1,:) = x_ic;
for i=1:size(U1,1)
    X1(i + 1,:) = rk4_integrate(X1(i,:)', U1(i,:)', 0.1, 4);
end
fprintf("rk4-4 took %f\n", toc);
plot(X1(:,1), X1(:,3), '.', 'DisplayName', 'rk4-4');
legend;
drawnow;

tic;
X5 = zeros(size(U1,1) + 1, 6);
X5(1,:) = x_ic;
for i=1:size(U1,1)
    X5(i + 1,:) = rk4_integrate(X5(i,:)', U1(i,:)', 0.1, 8);
end
fprintf("rk4-8 took %f\n", toc);
plot(X5(:,1), X5(:,3), '.', 'DisplayName', 'rk4-8');
legend;
drawnow;


tic;
X13 = zeros(size(U1,1) + 1, 6);
X13(1,:) = x_ic;
for i=1:size(U1,1)
    X13(i + 1,:) = rk4_integrate(X13(i,:)', U1(i,:)', 0.1, 16);
end
fprintf("rk4-16 took %f\n", toc);
plot(X13(:,1), X13(:,3), '.', 'DisplayName', 'rk4-16');
legend;
drawnow;


U2 = repelem(U1, 10, 1);
tic;
[X2, ~] = forwardIntegrateControlInput(U2, x_ic);
fprintf("fi took %f\n", toc);
plot(X2(:,1), X2(:,3), 'x', 'DisplayName', 'fi');
legend;
drawnow;

tic;
[X3, ~] = forwardIntegrateControlInput2(U2, x_ic);
fprintf("fi2 took %f\n", toc);
plot(X3(:,1), X3(:,3), 'x', 'DisplayName', 'fi2');
legend;
drawnow;

tic;
[X7, ~] = forwardIntegrateControlInput2(U2, x_ic, 1e-7);
fprintf("fi2-7 took %f\n", toc);
plot(X7(:,1), X7(:,3), 'x', 'DisplayName', 'fi2-7');
legend;
drawnow;

tic;
[X8, ~] = forwardIntegrateControlInput2(U2, x_ic, 1e-8);
fprintf("fi2-8 took %f\n", toc);
plot(X8(:,1), X8(:,3), 'x', 'DisplayName', 'fi2-8');
legend;
drawnow;

tic;
[X9, ~] = forwardIntegrateControlInput2(U2, x_ic, 1e-9);
fprintf("fi2-9 took %f\n", toc);
plot(X9(:,1), X9(:,3), 'x', 'DisplayName', 'fi2-9');
legend;
drawnow;

tic;
[X10, ~] = forwardIntegrateControlInput2(U2, x_ic, 1e-10);
fprintf("fi2-10 took %f\n", toc);
plot(X10(:,1), X10(:,3), 'x', 'DisplayName', 'fi2-10');
legend;
drawnow;

tic;
[X14, ~] = forwardIntegrateControlInput2(U2, x_ic, 1e-11);
fprintf("fi2-11 took %f\n", toc);
plot(X14(:,1), X14(:,3), 'x', 'DisplayName', 'fi2-11');
legend;
drawnow;

tic;
[X15, ~] = forwardIntegrateControlInput3(U2, x_ic, 1e-6);
fprintf("fi3-6 took %f\n", toc);
plot(X15(:,1), X15(:,3), 'DisplayName', 'fi3-6');
legend;
drawnow;

tic;
[X16, ~] = forwardIntegrateControlInput3(U2, x_ic, 1e-7);
fprintf("fi3-7 took %f\n", toc);
plot(X16(:,1), X16(:,3), 'DisplayName', 'fi3-7');
legend;
drawnow;

tic;
[X17, ~] = forwardIntegrateControlInput3(U2, x_ic, 1e-8);
fprintf("fi3-8 took %f\n", toc);
plot(X17(:,1), X17(:,3), 'DisplayName', 'fi3-8');
legend;
drawnow;

tic;
[X18, ~] = forwardIntegrateControlInput3(U2, x_ic, 1e-9);
fprintf("fi3-9 took %f\n", toc);
plot(X18(:,1), X18(:,3), 'DisplayName', 'fi3-9');
legend;
drawnow;

tic;
[X19, ~] = forwardIntegrateControlInput3(U2, x_ic, 1e-10);
fprintf("fi3-10 took %f\n", toc);
plot(X19(:,1), X19(:,3), 'DisplayName', 'fi3-10');
legend;
drawnow;

tic;
[X20, ~] = forwardIntegrateControlInput3(U2, x_ic, 1e-11);
fprintf("fi3-11 took %f\n", toc);
plot(X20(:,1), X20(:,3), 'DisplayName', 'fi3-11');
legend;
drawnow;

