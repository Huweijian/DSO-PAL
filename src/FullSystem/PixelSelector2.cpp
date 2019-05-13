/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#include "FullSystem/PixelSelector2.h"

#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"
#include "util/pal_interface.h"

namespace dso
{


PixelSelector::PixelSelector(int w, int h)
{
	randomPattern = new unsigned char[w*h];
	std::srand(3141592);	// want to be deterministic.
	for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF;
	currentPotential=3;


	gradHist = new int[100*(1+w/32)*(1+h/32)];
	ths = new float[(w/32)*(h/32+1)+100];
	thsSmoothed = new float[(w/32+1)*(h/32+1)+100];

	memset(thsSmoothed, 0, sizeof(thsSmoothed));
	memset(ths, 0, sizeof(ths));

	allowFast=false;
	gradHistFrame=0;
}

PixelSelector::~PixelSelector()
{
	delete[] randomPattern;
	delete[] gradHist;
	delete[] ths;
	delete[] thsSmoothed;
}
// 计算直方图below%的点位于直方图的位置
int computeHistQuantil(int* hist, float below)
{
	int th = hist[0]*below+0.5f;
	for(int i=0;i<90;i++)
	{
		th -= hist[i+1];
		if(th<0) return i;
	}
	return 90;
}

// 根据梯度直方图计算每块小区域的梯度阈值
void PixelSelector::makeHists(const FrameHessian* const fh)
{
	gradHistFrame = fh;
	float * mapmax0 = fh->absSquaredGrad[0]; //梯度图

	int w = wG[0];
	int h = hG[0];

	int w32 = w/32;
	int h32 = h/32;
	thsStep = w32;
	// 把图像分为32×32个小块，计算每块的梯度直方图，并计算每个小块的梯度阈值(7+梯度中值)
	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			float* map0 = mapmax0+32*x+32*y*w;
			int* hist0 = gradHist;// + 50*(x+y*w32);
			memset(hist0,0,sizeof(int)*50);

			for(int j=0;j<32;j++) 
				for(int i=0;i<32;i++)
				{
					int it = i+32*x;
					int jt = j+32*y;

					// 位于图像边缘的点不加入梯度直方图
					if(USE_PAL == 1 || USE_PAL == 2){
						if(!pal_check_in_range_g(it, jt, 1, 0))
							continue;	
					}
					else
					{
						if(it>w-2 || jt>h-2 || it<1 || jt<1) 
							continue;
					}
					int g = sqrtf(map0[i+j*w]);
					if(g>48) g=48;
					hist0[g+1]++;
					hist0[0]++;
				}

			ths[x+y*w32] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd;

			if(USE_PAL == 1 || USE_PAL == 2){// mask以外的不选择点
				if(!pal_check_in_range_g(32*x+16, 32*y+16, 1, 0))
					ths[x+y*w32] *= 3;
			}		

		}

	// 3*3滑动窗口平滑一下,再平方
	for(int y=0;y<h32;y++)
	for(int x=0;x<w32;x++)
	{
		float sum=0,num=0;
		if(x>0)
		{
			if(y>0) 	{num++; 	sum+=ths[x-1+(y-1)*w32];}
			if(y<h32-1) {num++; 	sum+=ths[x-1+(y+1)*w32];}
			num++; sum+=ths[x-1+(y)*w32];
		}

		if(x<w32-1)
		{
			if(y>0) 	{num++; 	sum+=ths[x+1+(y-1)*w32];}
			if(y<h32-1) {num++; 	sum+=ths[x+1+(y+1)*w32];}
			num++; sum+=ths[x+1+(y)*w32];
		}

		if(y>0) 	{num++; 	sum+=ths[x+(y-1)*w32];}
		if(y<h32-1) {num++; 	sum+=ths[x+(y+1)*w32];}
		num++; sum+=ths[x+y*w32];

		thsSmoothed[x+y*w32] = (sum/num) * (sum/num);

		// PAL按照视场范围修改阈值,并且不使用滑动窗口
		// if(ENH_PAL){ //pal点选择阈值,不使用滑动平均
		// 	float w = pal_get_weight(Vec2f(x*32, y*32));
		// 	w = 0.0 + 1*((1-w));
		// 	thsSmoothed[x+y*w32] = w * ths[x+y*w32] * ths[x+y*w32]; //(阈值高,)
		// 	// thsSmoothed[x+y*w32] = w * ths[x+y*w32]; //(阈值低,)
		// 	// thsSmoothed[x+y*w32] = (sum/num) * (sum/num);
		// }

	}

	// hwjdebug
	{
		using namespace cv;

		Mat thsMat = Mat(h32, w32, CV_8UC1);
		Mat thsSmoMat = Mat(h32, w32, CV_8UC1);
		auto grad = IOWrap::getOCVImg_tem(mapmax0, w, h);

		for(int y=0;y<h32;y++){

			for(int x=0;x<w32;x++){
				circle(grad, Point(x*32, y*32), 3, 255);

				thsMat.at<uchar>(y, x) = ths[y*w32+x]>255? 255: ths[y*w32+x];
				// printf("%.1f ", ths[y*w32+x]);
				thsSmoMat.at<uchar>(y, x) = thsSmoothed[y*w32+x]>255? 255:thsSmoothed[y*w32+x];
				grad.at<uchar>(y*32, x*32) = thsSmoMat.at<uchar>(y, x);

				// grad.at<uchar>(y*32, x*32) = thsSmoothed[y*w32+x];
				// grad.at<uchar>(y*32, x*32+1) = ths[y*w32+x];
			}
			// printf("\n");
		}


		resize(thsMat, thsMat, Size(), 32, 32, INTER_NEAREST);
		resize(thsSmoMat, thsSmoMat, Size(), 32, 32, INTER_NEAREST);

		// imshow("ths", thsMat);
		// imshow("thsSmo", thsSmoMat);

		// imshow("grad", grad);
		// moveWindow("grad", 50+w+50, 50);
		// waitKey(1);
	}


}

// density: 希望采集xx个点  recursionLeft：递归采集剩余次数（默认=1表示允许递归采集1次） thFactor:thresholdFactor 梯度阈值因子(默认=1)
// 返回成功采集的点
int PixelSelector::makeMaps(
		const FrameHessian* const fh,
		float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
	float numHave=0;
	float numWant=density;
	float quotia;
	int idealPotential = currentPotential;

	{

		// the number of selected pixels behaves approximately as
		// K / (pot+1)^2, where K is a scene-dependent constant.
		// we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.

		// 分成32*32个小块，利用梯度直方图计算每个小块的梯度阈值
		if(fh != gradHistFrame) {
			printf("\t Point selection, want=%.1f\n", density);
			makeHists(fh);
		}

		// select!
		// 根据阈值选择点
		Eigen::Vector3i n = this->select(fh, map_out,currentPotential, thFactor);
		printf("\t - it %d, get %d points, pot=%d\n", 1-recursionsLeft, n[0] + n[1] + n[2], currentPotential);
		
		// sub-select!
		// 根据本次采集的点数重新估计网格大小(Potential 可以理解为采点的网格大小)
		numHave = n[0]+n[1]+n[2];
		quotia = numWant / numHave; 
		// by default we want to over-sample by 40% just to be sure.
		float K = numHave * (currentPotential+1) * (currentPotential+1);
		idealPotential = sqrtf(K/numWant)-1;	// round down.
		if(idealPotential<1) idealPotential=1;

		// 如果点数过多或者过少，都重新采集
		if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
		{
			//re-sample to get more points!
			// potential needs to be smaller
			if(idealPotential>=currentPotential)
				idealPotential = currentPotential-1;

			// printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
			// 		100*numHave/(float)(wG[0]*hG[0]),
			// 		100*numWant/(float)(wG[0]*hG[0]),
			// 		currentPotential,
			// 		idealPotential);

			printf("\t - (点太少)currPot change from %d to %d\n", currentPotential, idealPotential);
			currentPotential = idealPotential;
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);
		}
		else if(recursionsLeft>0 && quotia < 0.25)
		{
			// re-sample to get less points!

			if(idealPotential<=currentPotential)
				idealPotential = currentPotential+1;

			// printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
			// 		100*numHave/(float)(wG[0]*hG[0]),
			// 		100*numWant/(float)(wG[0]*hG[0]),
			// 		currentPotential,
			// 		idealPotential);

			printf("\t - (点太多)currPot change from %d to %d\n", currentPotential, idealPotential);
			currentPotential = idealPotential;
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);

		}
	}

	// 如果采集的点还是略多，随机去掉一些点
	int numHaveSub = numHave;
	if(quotia < 0.95)
	{
		int wh=wG[0]*hG[0];
		int rn=0;
		unsigned char charTH = 255*quotia;
		for(int i=0;i<wh;i++)
		{
			if(map_out[i] != 0)
			{
				if(randomPattern[rn] > charTH )
				{
					map_out[i]=0;
					numHaveSub--;
				}
				rn++;
			}
		}
	}

//	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
//			100*numHave/(float)(wG[0]*hG[0]),
//			100*numWant/(float)(wG[0]*hG[0]),
//			currentPotential,
//			idealPotential,
//			100*numHaveSub/(float)(wG[0]*hG[0]));


	if(plot)
	{
		int w = wG[0];
		int h = hG[0];


		MinimalImageB3 img(w,h);

		for(int i=0;i<w*h;i++)
		{
			float c = fh->dI[i][0]*0.7;
			if(c>255) c=255;
			img.at(i) = Vec3b(c,c,c);
		}

		// IOWrap::displayImage("Selector Image", &img);

		for(int y=0; y<h;y++)
			for(int x=0;x<w;x++)
			{
				int i=x+y*w;
				if(map_out[i] == 1)
					img.setPixel1(x,y,Vec3b(0,255,0));
				else if(map_out[i] == 2)
					img.setPixel1(x,y,Vec3b(255,0,0));
				else if(map_out[i] == 4)
					img.setPixel1(x,y,Vec3b(0,0,255));
			}
		
		// hwjdebug---------------------
		cv::Mat cvimg = cv::Mat(img.h, img.w, CV_8UC3, img.data);

		// if(currentPotential > 1){
		// 	for(int x=0; x<img.w; x+=currentPotential){
		// 		for(int y=0; y<img.h; y+=currentPotential){
		// 			cvimg.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(255, 255, 0);
		// 		}
		// 	}
		// }
		cv::imshow("Selector Pixels", cvimg);
		// -----------------------------
		// IOWrap::displayImage("Selector Pixels", &img);
		cv::moveWindow("Selector Pixels", 100+img.w, 50);
		cv::waitKey(0);
	}

	printf("\t - (点正常)currPot change from %d to %d\n", currentPotential, idealPotential);
	currentPotential = idealPotential;

	return numHaveSub;
}


// 选中的点,map_out > 0 (值代表在x层网格中被选中) 没选中的点，map_out = 0.0
// pot,每个pot*pot的像素块中只会选一个点
Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
		float* map_out, int pot, float thFactor)
{

	Eigen::Vector3f const * const map0 = fh->dI;

	float * mapmax0 = fh->absSquaredGrad[0];
	float * mapmax1 = fh->absSquaredGrad[1];
	float * mapmax2 = fh->absSquaredGrad[2];


	int w = wG[0];
	int w1 = wG[1];
	int w2 = wG[2];
	int h = hG[0];


	const Vec2f directions[16] = {
	         Vec2f(0,    1.0000),
	         Vec2f(0.3827,    0.9239),
	         Vec2f(0.1951,    0.9808),
	         Vec2f(0.9239,    0.3827),
	         Vec2f(0.7071,    0.7071),
	         Vec2f(0.3827,   -0.9239),
	         Vec2f(0.8315,    0.5556),
	         Vec2f(0.8315,   -0.5556),
	         Vec2f(0.5556,   -0.8315),
	         Vec2f(0.9808,    0.1951),
	         Vec2f(0.9239,   -0.3827),
	         Vec2f(0.7071,   -0.7071),
	         Vec2f(0.5556,    0.8315),
	         Vec2f(0.9808,   -0.1951),
	         Vec2f(1.0000,    0.0000),
	         Vec2f(0.1951,   -0.9808)};

	memset(map_out,0,w*h*sizeof(PixelSelectorStatus));



	float dw1 = setting_gradDownweightPerLevel;
	float dw2 = dw1*dw1;


	int n3=0, n2=0, n4=0;
	// 4倍pot枚举枚举网格
	for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot))
	{
		int my3 = std::min((4*pot), h-y4);
		int mx3 = std::min((4*pot), w-x4);
		int bestIdx4=-1; float bestVal4=0;
		Vec2f dir4 = directions[randomPattern[n2] & 0xF];

		// 枚举每个4倍网格中的2倍网格
		for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
		{
			int x34 = x3+x4;
			int y34 = y3+y4;
			int my2 = std::min((2*pot), h-y34);
			int mx2 = std::min((2*pot), w-x34);
			int bestIdx3=-1; float bestVal3=0;
			Vec2f dir3 = directions[randomPattern[n2] & 0xF];

			// 枚举每个2倍网格中的1倍网格
			for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)
			{
				int x234 = x2+x34;
				int y234 = y2+y34;
				int my1 = std::min(pot, h-y234);
				int mx1 = std::min(pot, w-x234);
				int bestIdx2=-1; float bestVal2=0;
				Vec2f dir2 = directions[randomPattern[n2] & 0xF];

				// 枚举每个网格中的每个像素
				for(int y1=0;y1<my1;y1+=1) for(int x1=0;x1<mx1;x1+=1)
				{
					assert(x1+x234 < w);
					assert(y1+y234 < h);
					int idx = x1+x234 + w*(y1+y234);
					int xf = x1+x234;
					int yf = y1+y234;

					if(USE_PAL == 1 || USE_PAL == 2){
						if(!pal_check_in_range_g(xf, yf, 4)) continue;
					}
					else{
						if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;
					}

					// 三个梯度阈值(一个比一个低0.75)
					float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep] > 6 ? thsSmoothed[(xf>>5) + (yf>>5) * thsStep] : 100000;
					float pixelTH1 = pixelTH0*dw1;
					float pixelTH2 = pixelTH1*dw2;

					// 寻找梯度足够大的点，优先在金字塔底寻找
					float ag0 = mapmax0[idx];
					if(ag0 > pixelTH0*thFactor) // 如果梯度大于第一个阈值
					{
						Vec2f ag0d = map0[idx].tail<2>(); //vector的后两个元素，也就是dx和dy
						float dirNorm = fabsf((float)(ag0d.dot(dir2))); // 梯度在某个随机方向上的投影
						if(!setting_selectDirectionDistribution) dirNorm = ag0;

						// 保存梯度最大的点,之后就不用在高层金字塔上选点了
						if(dirNorm > bestVal2)
						{ bestVal2 = dirNorm; bestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
					}
					if(bestIdx3==-2) continue;

					// 如果这一点对应第1层金字塔梯度超过阈值,那么就选当前点
					float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1];
					if(ag1 > pixelTH1*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir3)));
						if(!setting_selectDirectionDistribution) dirNorm = ag1;

						if(dirNorm > bestVal3)
						{ bestVal3 = dirNorm; bestIdx3 = idx; bestIdx4 = -2;}
					}
					if(bestIdx4==-2) continue;

					float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2];
					if(ag2 > pixelTH2*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir4)));
						if(!setting_selectDirectionDistribution) dirNorm = ag2;

						if(dirNorm > bestVal4)
						{ bestVal4 = dirNorm; bestIdx4 = idx; }
					}
				}

				if(bestIdx2>0)
				{
					map_out[bestIdx2] = 1;
					bestVal3 = 1e10;
					n2++;
				}
			}

			if(bestIdx3>0)
			{
				map_out[bestIdx3] = 2;
				bestVal4 = 1e10;
				n3++;
			}
		}

		if(bestIdx4>0)
		{
			map_out[bestIdx4] = 4;
			n4++;
		}
	}


	return Eigen::Vector3i(n2,n3,n4);
}


}

