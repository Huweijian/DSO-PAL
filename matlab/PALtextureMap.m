% clear;
% addpath(ocam_path) and load calib mat file
cam = calib_data.ocam_model;

% read image
szp = 0.5;
sz = 720*szp;
img = imread('9.png');
img = imresize(img, [sz, sz]);

% sphere points
[X,Y,Z] = sphere(sz-1);
P = zeros([size(X)  3]);
P(:, :, 1) = Y;
P(:, :, 2) = X;
P(:, :, 3) = -Z;
P = reshape(P, sz*sz, 3)';

% image points
uv = world2cam(P, cam) *szp;
u = uv(1, :);
v = uv(2, :);
uv(:, u<1 | u >sz) = 1;
uv(:, v<1 | v >sz) = 1;
uv_int = int32(round(uv));

% color map
Color = zeros([sz sz 3], 'uint8');
for i=1:sz
    for j=1:sz
        idx = (i-1)*sz+j;
        Color(j, i, :) = img(uv_int(1, idx), uv_int(2, idx), :);
    end
end

% sphere -> cube
Pabs = abs(P);
Pmed = median(Pabs);
[Pmax, maxIdx] = max(Pabs, [], 1);
P = P./Pmax;

% cube -> long cube
topCenterPoint = abs(P(1,:))<0.5 & abs(P(2, :))<0.5 & P(3, :) == -1;
topNonCenterPoint = (abs(P(1,:))>=0.5 | abs(P(2, :))>=0.5) & P(3, :) == -1;
Pmax2 = max(abs(P(1:2, topNonCenterPoint)), [], 1);
P(:, topNonCenterPoint) = P(:, topNonCenterPoint) ./Pmax2;
P(:, topCenterPoint) = P(:, topCenterPoint) * max(abs(P(3, :)));

% show 
P = P';
Y = reshape(P(:, 1), sz, sz);
X = reshape(P(:, 2), sz, sz);
Z = -reshape(P(:, 3), sz, sz);
surface(X,Y,Z, Color, ...
    'FaceColor','texturemap',...
    'EdgeColor','none',...
    'CDataMapping','direct');
view(3)
axis([-1 1 -1 1 -1 1]*3)


