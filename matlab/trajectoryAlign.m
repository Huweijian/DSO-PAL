clc;
LEN = length(tt_dso);
%% calc and output trajectory
% ------ DLT ----------------
R_d0_mk = nan(3, 3, LEN);
for i=1:LEN
    if isnan(tt_mk(1, i)) || isnan(tt_dso(1, i))
        continue;
    end
    RR_mk_c = squeeze(RR_mk(:, :, i));
    RR_d0_c = squeeze(RR_dso(:, :, i));
    R_d0_mk(:, :, i) = RR_d0_c * inv(RR_mk_c);

end
Rv_d0_mk = rotm2axang(R_d0_mk);
Rv_d_m = mean(Rv_d0_mk, 1);
R_d_m = axang2rotm(Rv_d_m);

tra_dso_noscale = inv(R_d_m) * tt_dso;
A = zeros(3*LEN, 4);
B = zeros(3*LEN, 1);
for i=1:LEN
    ii = (i-1)*3 + 1;
    A(ii:ii+2, 1) = tra_dso_noscale(:, i);
    A(ii:ii+2, 2:4) = eye(3);
    B(ii:ii+2) = tt_mk(:, i);
end
resv = A\B;
s_dlt = resv(1);
t_dlt = resv(2:4);
tra_dso = tra_dso_noscale * s_dlt + t_dlt;
tra_mk = tt_mk ;
% -------------------

tra_dso_all = tt_dso_raw(:, ~isnan(tt_dso_raw(1, :)));
tra_dso_all = inv(R_d_m) * (tra_dso_all*s_dlt) + t_dlt;
tra_start = idx(1);
tra_dso_all = tra_dso_all(:, tra_start:end);

resfileid = fopen([seq_dir '/trajectory.txt'], 'w');
fprintf(resfileid, '%f %f %f\n', tra_dso_all);
fclose(resfileid);

%% visualize trajectory
minn = min(tra_dso, [], 2); minn = min(minn)*2;
maxn = max(tra_dso, [], 2); maxn = max(maxn)*2;

figure(11)
clf(11)
plot3(tra_dso(1, :), tra_dso(2, :), tra_dso(3, :), 'b'); hold on;
plot3(tra_dso(1, 1), tra_dso(2, 1), tra_dso(3, 1), 'bo');
axis([minn maxn minn maxn minn maxn]);

% figure(12)
% clf(12);
plot3(tra_mk(1, :), tra_mk(2, :), tra_mk(3, :), 'r'); hold on;
plot3(tra_mk(1, 1), tra_mk(2, 1), tra_mk(3, 1), 'ro');
axis([minn maxn minn maxn minn maxn]);


figure(13)
tra_err = tra_dso - tra_mk;
tra_err = tra_err(1, :).^2 + tra_err(2, :).^2 + tra_err(3, :).^3;
stem(tra_err);
axis([-Inf Inf 0 0.01]);

% 显示全部矫正后的轨迹
% figure(14)
% clf(14)
% plot3(tra_dso_all(1, :), tra_dso_all(2, :), tra_dso_all(3, :), 'b');
% hold on;
% plot3(tra_dso_all(1, 1), tra_dso_all(2, 1), tra_dso_all(3, 1), 'ro');
% minn_all = min(tra_dso_all, [], 2); minn_all = min(minn_all);
% maxn_all = max(tra_dso_all, [], 2); maxn_all = max(maxn_all);
% axis([minn_all maxn_all minn_all maxn_all minn_all maxn_all]);


