clear;
dataset = 'pal_real_hall';

ptc = pcread(['result_old/' dataset '.pcd']);
pcrot = ptc;

% hall2 only
% rot = eul2rotm([0, pi/10 * (-1.0), pi/4 * (0.8)]);
% pcrot = pointCloud((rot * ptc.Location')', 'Color', ptc.Color);
% ----------

pcshow(pcrot);
view(3)
hold on;


traj = importdata(['result_old/' dataset '.log']);

% dso hall2 only
% traj(end, :) = [];
% len_t = length(traj);
% traj = reshape(traj', [14 int32(len_t/7) ])';
% traj = traj(:, [1 6 10 14]);
% traj(:, 2:4) = (rot * traj(:, 2:4)')';
% ahahahah

traj = traj(:, 2:4);
plot3(traj(:, 1), traj(:, 2), traj(:, 3), 'LineWidth', 3, 'Color', 'r');

grid off;


% large scale
% axis([-50 60 -30 30 -5 30])

% pal hall
axis([-20 40 -30 30 -5 5])

% dso hall 1 
% axis([-15 10 -5 20 -5 5])



