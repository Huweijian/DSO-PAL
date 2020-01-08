clear;
%% test point-to-line distance
% clear;
% x0 = [1 1 1]';
% u = [1 1 1]' ;
% u = u / norm(u);
% 
% 
% p = [1 1 0]';
% distv = (x0-p) + dot(p-x0, u)*u;
% dist = norm(distv)
% 
% 
% return;

%% test SVD
% clear;
% 
% figN = 1;
% figure(figN); clf(figN);
% 
% 
% p = importdata('data1.txt');
% scatter3(p(:, 1), p(:, 2), p(:, 3))
% 
% mp = mean(p);
% p = p - mp;
% [u, d, v] = svd(p);
% vv = v(:, 1)';
% a = 1;
% 
% mp =  [ -0.264869  0.340067    0.9825];
% vv = [0.251155 -0.963363 0.0940897];
% 
% pfit = [mp-a*vv; mp+a*vv];
% hold on;
% plot3(pfit(:, 1), pfit(:, 2), pfit(:, 3));
% % axis([-Inf Inf -Inf Inf 0 2])

%% test arccos
clear;
theta = deg2rad(50 + randn(100, 1));
x = cos(theta);
y = sin(theta);
g = [x y];
dlmwrite('test.txt', g, ' ');

