clear;
dataset = 'pal_real_large_scale';

ptc = pcread(['result/' dataset '.pcd']);



pcshow(ptc);
view(3)
hold on;
traj = importdata(['result/' dataset '.log']);
traj = traj(:, 2:4);
plot3(traj(:, 1), traj(:, 2), traj(:, 3), 'LineWidth', 3, 'Color', 'r');




% large scale
axis([-50 60 -30 30 -5 30])

% pal hall
% axis([-20 40 -30 30 -5 5])

% dso
% axis([-15 10 -5 20 -5 5])