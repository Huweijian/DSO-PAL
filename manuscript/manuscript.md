1. Introduction  
盲人需要定位  
视觉定位的特点
PAL的特点,pal为什么适合这个应用  

2. Related works  
omnidirection camera applications  
blind localization articles

3. Method  
- structure
- camera model : pin unity  
- tracking: epolar mathching, triangluation 
- *hole fixing

4. evaluation 
- localization on pulic dataset(H) [compare different model]
- blind-folded field experiment 

5. conclusion

## 必要性  
- 为什么朝上看:向前看动态物体太多,给视觉定位带来额外的负担
- 为什么用大市场相机:视场大,适合向上看
- 为什么用pal:相比鱼眼,同视角下体积更小,适合可穿戴设备.  

## 创新点
#### 在PAL模型上实现了定位算法  
1. 直接法:求导
2. tracking: 极线匹配
3. *误差推导:边缘市场对梯度求导的步长需要优化（z变化一点点，实际的距离会变化很多），因此，对于边缘市场，z在优化过程中的step应该小一些。
4. *针对空洞进行优化 
5. *在另外两种模型上实现了pal算法:单针孔，分段针孔
6. *相机模型自适应切换：对于屋顶面积较大的地方，使用单一针孔模型，对于周围特征点较多的地方，使用分段针孔模型。对于其他场景，使用球面模型。

#### 完成了一套盲人应用系统
工作流程：  
1. 明眼人建立地图
2. 检测marker判断入口
3. *根据房间平面图选择合适的相机模型

## 实验
#### 上述改进的有效性  
1. 大视场的好处
2. *相机模型在各自场景中的好处

#### 和别人对比
1. 和其他针孔slam在实际场景中的表现
2. *在公开数据集上的表现(TUM omni数据集)

#### *实际盲人实验
1. 蒙眼人定位效果 