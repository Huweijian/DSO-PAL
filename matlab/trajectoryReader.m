clc;
% clear;
clearvars -except tra_dso_all_s33 tra_dso_all_s34 tra_dso_all_s35 tra_dso_all_s36 tra_dso_all_s38 tra_dso_all_s37

MAX_LEN = 2000; % max len
MARKER_ID = 223;
seq_dir = 'pal_s38';

mkfile = [seq_dir '/hwjcamPoseMarker.log'];
mkfileid = fopen(mkfile);
tt_mk = nan(3, MAX_LEN);
RR_mk = nan(3, 3, MAX_LEN);
id_mk = nan(1, MAX_LEN);
mpid = nan(1, MAX_LEN);
while 1
    a = fscanf(mkfileid, '%d %d %d', 3);
    if isempty(a)
        break;
    end
    mkFrameId = a(1)+1;
    mpid(mkFrameId) = a(3);
        
    t = [nan nan nan]';
    R = nan(3, 3);
    if a(2) ~= -1 
        R = fscanf(mkfileid, '%f', 9);
        R = reshape(R, 3, 3)';
        t = fscanf(mkfileid, '%f', 3);
        RR_mk(:, :, mkFrameId) = R;
        tt_mk(:, mkFrameId) = t;
        id_mk(mkFrameId) = a(2);
    end
end
fclose(mkfileid);


%% 
dsofile = [seq_dir '/hwjcamPoseDso.log'];
dsofileid = fopen(dsofile);
tt_dso_raw = nan(3, MAX_LEN);
RR_dso = nan(3, 3, MAX_LEN);
id_dso = nan(1, MAX_LEN);
while 1
    a = fscanf(dsofileid, '%d %d', 2); 
    dsoFrameId = a(1) + 1;
    
    % remove fail init result
    if isnan(id_dso(dsoFrameId-1))
        id_dso(1:mkFrameId-1) = nan;
        RR_dso(:, :, 1:mkFrameId-1) = nan;
        tt_dso_raw(:, 1:mkFrameId-1) = nan;
    end
    
    Rt = fscanf(dsofileid, '%f', 12);
    if isempty(Rt)
        break;
    end
        
    Rt = reshape(Rt,4, 3)';
    R = Rt(:, 1:3);
    t = Rt(:, 4);
    id_dso(dsoFrameId) = a(2);
    RR_dso(:, :, dsoFrameId) = R;
    tt_dso_raw(:, dsoFrameId) = t;
end
fclose(dsofileid);

idx = 1:MAX_LEN;
idvalid = (id_dso == MARKER_ID) & (id_mk == MARKER_ID) ;

idx = idx(idvalid);
tt_dso = tt_dso_raw(:, idvalid);
tt_mk = tt_mk(:, idvalid);
RR_dso = RR_dso(:, :, idvalid);
RR_mk = RR_mk(:, :, idvalid);

% inverse pose MK
for i=1:length(RR_mk)
    if(isnan(tt_mk(1, i)))
        continue ;
    end
    T = inv([RR_mk(:, :, i) tt_mk(:, i); [0 0 0 1]]);
    RR_mk(:, :, i) = T(1:3, 1:3);
    tt_mk(:, i) = T(1:3, 4);
end


%% calse
trajectoryAlign;
%% 可视化原始位移
% figure(1)
% plot(idx, tt_dso', '.');
% figure(2)
% plot(idx, tt_mk', '.');

%% 可视化原始旋转数据
% ST = 1;
% RR_mk_d = nan(size(RR_mk));
% RR_dso_d = nan(size(RR_dso));
% RR_mk_dso = nan(size(RR_dso));
% for i=ST:length(RR_mk)
%     if(isnan(RR_mk(1, 1, i)))
%         continue ;
%     end
%     RR_mk_d(:, :, i) = RR_mk(:, :, i) * inv(RR_mk(:, :, ST));
%     RR_dso_d(:, :, i) = inv(RR_dso(: ,:, i) * inv(RR_dso(:, :, ST)));
%     RR_mk_dso(:, :, i) = RR_dso(:, :, i) * inv(RR_mk(: ,:, i));
% end
% 
% Rv_dso = rotm2axang(RR_dso_d);
% Rv_mk = rotm2axang(RR_mk_d);
% Rv_mk_dso = rotm2axang(RR_mk_dso);
% 
% figure(5);
% plot(Rv_dso);
% figure(6);
% plot(Rv_mk);
% figure(7);
% plot(Rv_mk_dso);


%% test 
% R = axang2rotm([0, 1, 0, pi/4]);
% t = [ 1 2 3 ]';
% T = [R , t; zeros(1, 3), 1];
% Tinv = inv(T);
% Rvinv = rotm2axang(Tinv(1:3, 1:3));
% Rinv = Tinv(1:3, 1:3);
% tinv = Tinv(1:3, 4);
% 
% p1 = [1 2 3 1]';
% p2 = T * p1;
% p2t = R*p1(1:3) + t;
% p1tt = Rinv * p2t - t;
% p1t = Tinv * p2;









