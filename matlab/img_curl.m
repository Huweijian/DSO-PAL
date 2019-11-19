clear;
img_raw = imread('../1.png');
% img = img_raw(72:82, 300:310);
img = img_raw;

figure(4)
imshow(img);
[gx, gy] = gradient(double(img));
modgrad = sqrt(gx.^2 + gy.^2);
angle_grad = rad2deg(atan2(gx, -gy));

% 梯度角度
figure(1)
clf(1);
imagesc(angle_grad)
colormap hsv
colorbar 

% 梯度的散度
figure(3)
div = divergence(gx, gy);
div_check = (div<5);
% div(div_check) = 0;
imagesc(div);
colormap jet
colorbar

% 梯度角度的梯度角度
[gx2, gy2] = gradient(angle_grad);
angle_grad2 = rad2deg(atan2(gx2, -gy2));
modgrad2 = sqrt(gx2.^2 + gy2.^2);
% angle_grad2(div_check) = 0;
figure(2)
clf(2);
imagesc(modgrad2)
% colormap hsv
colorbar 


y = 1:size(img, 1);
x = 1:size(img, 2);


figure(5)
contour(x, y, img);
set(gca, 'ydir', 'reverse')
hold on;
quiver(x, y, gx, gy)
