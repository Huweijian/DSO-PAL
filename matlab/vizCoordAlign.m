% clear;

%% gt and pc 
pcfilename = 's33';

sim3local2global = importdata(['result/sim3_' pcfilename '.log']);
% sim3local2global = eye(4);
pc_raw = pcread(['result/' pcfilename '.pcd']);
pc = [pc_raw.Location'; ones(1, pc_raw.Count)];
pc = sim3local2global * pc;
pcw = pointCloud(pc(1:3, :)', 'Color', pc_raw.Color);
pcshow(pcw);

gt = importdata('result/trajectory_304_306.txt')';
hold on;
plot3(gt(1, :), gt(2, :), gt(3, :), 'LineWidth', 3, 'Color', 'r');
axis off;

%% load sequences
seq = 's30';
traj = importdata(['result/blind_' seq '.log']);
trajw = [traj(:, 2:4)'; ones(1, length(traj))];
sim3 = importdata(['result/sim3_' seq '.log']);
trajw = sim3 * trajw;

hold on;
plot3(trajw(1, :), trajw(2, :), trajw(3, :));


% large scale
% axis([-50 60 -30 30 -5 30])

% pal hall
% axis([-20 40 -30 30 -5 5])

% dso
% axis([-15 10 -5 20 -5 5])

%% vis 3 traj directly
figure(10);
clf(10);
gtinc = [
0.4     0       -3.677  0       8.525   0       3.102   0       -3.102  0       ;
0       -4.758	0       -8.678  0       3.696   0       1.547   0       3.307 
];
gt = cumsum(gtinc, 2);


hold on;
% plot(tra_dso_all(3, :), tra_dso_all(2, :));

plot(tra_dso_all_s33(3, :), tra_dso_all_s33(2, :)); % day
plot(tra_dso_all_s34(3, :), tra_dso_all_s34(2, :)); % night
plot(tra_dso_all_s35(3, :), tra_dso_all_s35(2, :)); % day
plot(tra_dso_all_s36(3, :), tra_dso_all_s36(2, :)); % night
plot(tra_dso_all_s37(3, :), tra_dso_all_s37(2, :)); % night
% plot(tra_dso_all_s38(3, :), tra_dso_all_s38(2, :)); % not good 

% plot(tra_dso_all_s39(3, :), tra_dso_all_s39(2, :));
% plot(tra_dso_all_s40(3, :), tra_dso_all_s40(2, :)); % day
% plot(tra_dso_all_s41(3, :), tra_dso_all_s41(2, :));
% plot(tra_dso_all_s42(3, :), tra_dso_all_s42(2, :)); % day
% plot(gt(1, :), gt(2, :));
axis([-10+4 15-4 -20+4 5-4])

legend('route 1','route 2','route 3','route 4','route 5')



