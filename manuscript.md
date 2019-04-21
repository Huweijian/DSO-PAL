## 必要性  
- 为什么朝上看:向前看动态物体太多,给视觉定位带来额外的负担
- 为什么用大市场相机:视场大,适合向上看
- 为什么用pal:相比鱼眼,同视角下体积更小,适合可穿戴设备.  

## 创新点
#### 在统一模型上实现了定位算法  
1. 直接法:求导
2. tracking: 极线匹配
3. *误差推导
#### 针对pal顶部视觉,对dso做了特殊优化  
1. 点的权重,点的选择 (TODO)
2. *针对空洞进行优化 

## 实验
#### 上述改进的有效性  
1. 大视场的好处
2. 点选择的好处

#### 和别人对比
1. 和其他针孔slam在实际场景中的表现
2. *在公开数据集上的表现(TUM omni数据集)

#### *实际盲人实验
1. 蒙眼人定位效果 