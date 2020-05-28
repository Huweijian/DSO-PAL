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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "IOWrapper/ImageRW.h"
#include "util/pal_interface.h"
#include "line/line_init.h"

#include <algorithm>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


template<int b, typename T>
T* allocAligned(int size, std::vector<T*> &rawPtrVec)
{
    const int padT = 1 + ((1 << b)/sizeof(T));
    T* ptr = new T[size + padT];
    rawPtrVec.push_back(ptr);
    T* alignedPtr = (T*)(( ((uintptr_t)(ptr+padT)) >> b) << b);
    return alignedPtr;
}


CoarseTracker::CoarseTracker(int ww, int hh) : lastRef_aff_g2l(0,0)
{
	// make coarse tracking templates.
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = ww>>lvl;
        int hl = hh>>lvl;

        idepth[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        weightSums[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        weightSums_bak[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);

        pc_u[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_v[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_idepth[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_color[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);

	}

	// warped buffers
    buf_warped_idepth = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_u = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_v = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_dx = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_dy = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_residual = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_weight = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_refColor = allocAligned<4,float>(ww*hh, ptrToDelete);

	newFrame = 0;
	lastRef = 0;
	debugPlot = debugPrint = true;
	w[0]=h[0]=0;
	refFrameID=-1;
}

CoarseTracker::~CoarseTracker()
{
    for(float* ptr : ptrToDelete)
        delete[] ptr;
    ptrToDelete.clear();
}

void CoarseTracker::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		if(USE_PAL == 1){ // 0 1
// #ifdef PAL
			fx[level] = 1;
			fy[level] = 1;
			cx[level] = 0;
			cy[level] = 0;
		}
		else{
// #else
			fx[level] = fx[level-1] * 0.5;
			fy[level] = fy[level-1] * 0.5;
			cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
			cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
		}
// #endif
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}

// 设置深度图
void CoarseTracker::makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians)
{
	// make coarse tracking templates for latstRef.
	memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);
	memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);

	// 枚举关键帧内的所有的ph，如果还是内点，就
	for(FrameHessian* fh : frameHessians)
	{
		for(PointHessian* ph : fh->pointHessians)
		{
			if(ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN)
			{
				// 获取ph到最新关键帧的pfr
				PointFrameResidual* r = ph->lastResiduals[0].first;
				assert(r->efResidual->isActive() && r->target == lastRef);
				// ph投影到最新关键帧的坐标
				int u = r->centerProjectedTo[0] + 0.5f;
				int v = r->centerProjectedTo[1] + 0.5f;
				float new_idepth = r->centerProjectedTo[2];
				float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12));

				// 保存到深度图和权重图
				idepth[0][u+w[0]*v] += new_idepth * weight;
				weightSums[0][u+w[0]*v] += weight;
			}
		}
	}

	// 枚举高层金字塔,计算深度图和误差图
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		float* idepth_l = idepth[lvl];
		float* weightSums_l = weightSums[lvl];

		float* idepth_lm = idepth[lvlm1];
		float* weightSums_lm = weightSums[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
			{
				int bidx = 2*x   + 2*y*wlm1;
				idepth_l[x + y*wl] = 		idepth_lm[bidx] +
											idepth_lm[bidx+1] +
											idepth_lm[bidx+wlm1] +
											idepth_lm[bidx+wlm1+1];

				weightSums_l[x + y*wl] = 	weightSums_lm[bidx] +
											weightSums_lm[bidx+1] +
											weightSums_lm[bidx+wlm1] +
											weightSums_lm[bidx+wlm1+1];
			}
	}

	// 对0-2层金字塔膨胀一次权重图
	// 下面这两段好像没用
    // dilate idepth by 1.
	for(int lvl=0; lvl<2; lvl++)
	{
		int numIts = 1;

		for(int it=0;it<numIts;it++)
		{
			int wh = w[lvl]*h[lvl]-w[lvl];
			int wl = w[lvl];
			float* weightSumsl = weightSums[lvl];
			float* weightSumsl_bak = weightSums_bak[lvl];
			memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
			float* idepthl = idepth[lvl];	// dotnt need to make a temp copy of depth, since I only
											// read values with weightSumsl>0, and write ones with weightSumsl<=0.

			// 枚举所有weightSumsl_bak点
			for(int i=w[lvl];i<wh;i++)
			{
				// 如果没有赋值过，用周围值的平均值填充这一点
				if(weightSumsl_bak[i] <= 0)
				{
					float sum=0, num=0, numn=0;
					if(weightSumsl_bak[i+1+wl] > 0) { sum += idepthl[i+1+wl]; num+=weightSumsl_bak[i+1+wl]; numn++;}
					if(weightSumsl_bak[i-1-wl] > 0) { sum += idepthl[i-1-wl]; num+=weightSumsl_bak[i-1-wl]; numn++;}
					if(weightSumsl_bak[i+wl-1] > 0) { sum += idepthl[i+wl-1]; num+=weightSumsl_bak[i+wl-1]; numn++;}
					if(weightSumsl_bak[i-wl+1] > 0) { sum += idepthl[i-wl+1]; num+=weightSumsl_bak[i-wl+1]; numn++;}
					if(numn>0) {
						idepthl[i] = sum/numn;
						weightSumsl[i] = num/numn;
					}
				}
			}
		}
	}

	// 对金字塔2层以上的深度图膨胀
	// dilate idepth by 1 (2 on lower levels).
	for(int lvl=2; lvl<pyrLevelsUsed; lvl++)
	{
		int wh = w[lvl]*h[lvl]-w[lvl];
		int wl = w[lvl];
		float* weightSumsl = weightSums[lvl];
		float* weightSumsl_bak = weightSums_bak[lvl];
		memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
		float* idepthl = idepth[lvl];	// dotnt need to make a temp copy of depth, since I only
										// read values with weightSumsl>0, and write ones with weightSumsl<=0.
		for(int i=w[lvl];i<wh;i++)
		{
			if(weightSumsl_bak[i] <= 0)
			{
				float sum=0, num=0, numn=0;
				if(weightSumsl_bak[i+1] > 0) { sum += idepthl[i+1]; num+=weightSumsl_bak[i+1]; numn++;}
				if(weightSumsl_bak[i-1] > 0) { sum += idepthl[i-1]; num+=weightSumsl_bak[i-1]; numn++;}
				if(weightSumsl_bak[i+wl] > 0) { sum += idepthl[i+wl]; num+=weightSumsl_bak[i+wl]; numn++;}
				if(weightSumsl_bak[i-wl] > 0) { sum += idepthl[i-wl]; num+=weightSumsl_bak[i-wl]; numn++;}
				if(numn>0) 
				{
					idepthl[i] = sum/numn; 
					weightSumsl[i] = num/numn;
				}
			}
		}
	}

	// normalize idepths and weights.
	// 归一化深度
	// 枚举每一层，每一个点，idep /= weight(之前深度是加权深度，这里再除以权重，变成真正的深度)
	// 把结果赋值导pc_xxx 变量
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		float* weightSumsl = weightSums[lvl];
		float* idepthl = idepth[lvl];
		Eigen::Vector3f* dIRefl = lastRef->dIp[lvl];

		int wl = w[lvl], hl = h[lvl];

		int lpc_n=0;
		float* lpc_u = pc_u[lvl];
		float* lpc_v = pc_v[lvl];
		float* lpc_idepth = pc_idepth[lvl];
		float* lpc_color = pc_color[lvl];

		for(int y=2;y<hl-2;y++)
			for(int x=2;x<wl-2;x++)
			{
				int i = x+y*wl;

				if(weightSumsl[i] > 0)
				{
					idepthl[i] /= weightSumsl[i];
					lpc_u[lpc_n] = x;
					lpc_v[lpc_n] = y;
					lpc_idepth[lpc_n] = idepthl[i];
					lpc_color[lpc_n] = dIRefl[i][0];

					if(!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i]>0))
					{
						idepthl[i] = -1;
						continue;	// just skip if something is wrong.
					}
					lpc_n++;
				}
				else
					idepthl[i] = -1;

				weightSumsl[i] = 1;
			}

		pc_n[lvl] = lpc_n;
	}

}

// SSE计算梯度
void CoarseTracker::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l)
{
	using namespace std;

	// 准备SSE常数
	__m128 fxl = _mm_set1_ps(fx[lvl]);
	__m128 fyl = _mm_set1_ps(fy[lvl]);
	__m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
	__m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]));
	__m128 one = _mm_set1_ps(1);
	__m128 minusOne = _mm_set1_ps(-1);
	__m128 zero = _mm_set1_ps(0);
	int n = buf_warped_n;
	assert(n%4==0);

	// 累加H矩阵和b
	acc.initialize();
	for(int i=0;i<n;i+=4)
	{
		if(USE_PAL == 1){ // 0 1
			float buf_drdSE3[6][4] = {0};
			for(int k=0; k<4; k++){
				if(buf_warped_idepth[i+k] != 0){	// 反深度为0会导致pt为nan

					Vec3f pt = Vec3f(buf_warped_u[i+k], buf_warped_v[i+k], 1) / buf_warped_idepth[i+k];
					Eigen::Matrix<float, 2, 6> dx2dSE;
					Eigen::Matrix<float, 2, 3> duv2dxyz;
					pal_model_g->jacobian_xyz2uv(pt, dx2dSE, duv2dxyz);
					Vec2f drdx2 = Vec2f(buf_warped_dx[i+k], buf_warped_dy[i+k]);
					Vec6f drdSE3 = drdx2.transpose() * dx2dSE;

					// hwjdebug -------------

					// if(i+k % 100 == 0){
					// 	printf("(i=%d lvl=%d) p = %.2f %.2f %.2f\n", i+k, lvl, buf_warped_u[i+k], buf_warped_v[i+k], buf_warped_idepth[i+k]);
					// 	cout << "pt = " << pt.transpose() << endl;
					// 	cout << "drdx2 = " << drdx2.transpose() << std::endl;
					// 	cout << dx2dSE << endl;
					// 	cout << "drdSE3 = " <<drdSE3.transpose() << endl; 
					// 	cout << "-----------" << endl;
					// }

					// -------------------

					for(int idx=0; idx<6; idx++){
						buf_drdSE3[idx][k] = drdSE3[idx];	
					}
				}
			}

			// 对SE的导数就是基本的直接法导数
			acc.updateSSE_weighted(
				_mm_load_ps(buf_drdSE3[0]),
				_mm_load_ps(buf_drdSE3[1]),
				_mm_load_ps(buf_drdSE3[2]),
				_mm_load_ps(buf_drdSE3[3]),
				_mm_load_ps(buf_drdSE3[4]),
				_mm_load_ps(buf_drdSE3[5]),
				_mm_mul_ps(a,_mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor+i))),					// 光度a a * (b - color)
				minusOne,																			// 光度b -1 
				_mm_load_ps(buf_warped_residual+i),													// res
				_mm_load_ps(buf_warped_weight+i)													// weight
			);
			// printf("i = %d, H(0,0) = [%.2f %.2f %.2f %.2f]\n", i, acc.SSEData[0], acc.SSEData[1], acc.SSEData[2], acc.SSEData[3]);
		}
		else{
			__m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx+i), fxl);
			__m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy+i), fyl);
			__m128 u = _mm_load_ps(buf_warped_u+i);
			__m128 v = _mm_load_ps(buf_warped_v+i);
			__m128 id = _mm_load_ps(buf_warped_idepth+i);

			// 对SE的导数就是基本的直接法导数
			acc.updateSSE_weighted(
					_mm_mul_ps(id,dx),																	// SE0 idep * gx * fx 
					_mm_mul_ps(id,dy),																	// SE1 idep * gy * fy
					_mm_sub_ps(zero, _mm_mul_ps(id,_mm_add_ps(_mm_mul_ps(u,dx), _mm_mul_ps(v,dy)))), 	// SE2 -idepth * (u*gx*fx + v*gy*fy)
					_mm_sub_ps(zero, _mm_add_ps(														// SE3 -(u*v*gx*fx + gy*fy*(1+v^2))	
							_mm_mul_ps(_mm_mul_ps(u,v),dx),
							_mm_mul_ps(dy,_mm_add_ps(one, _mm_mul_ps(v,v))))),
					_mm_add_ps(																			// SE4 u*v*gy*fy + gx*fx*(1+u^2)
							_mm_mul_ps(_mm_mul_ps(u,v),dy),
							_mm_mul_ps(dx,_mm_add_ps(one, _mm_mul_ps(u,u)))),
					_mm_sub_ps(_mm_mul_ps(u,dy), _mm_mul_ps(v,dx)),										// SE5 u*gy*fy - v*gx*fx
					_mm_mul_ps(a,_mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor+i))),					// 光度a a * (b - color)
					minusOne,																			// 光度b -1 
					_mm_load_ps(buf_warped_residual+i),													// res
					_mm_load_ps(buf_warped_weight+i));													// weight
		}
	}

	// 分解H和b
	acc.finish();
	H_out = acc.H.topLeftCorner<8,8>().cast<double>() * (1.0f/n);
	b_out = acc.H.topRightCorner<8,1>().cast<double>() * (1.0f/n);

	// 乘以权重
	H_out.block<8,3>(0,0) *= SCALE_XI_ROT;
	H_out.block<8,3>(0,3) *= SCALE_XI_TRANS;
	H_out.block<8,1>(0,6) *= SCALE_A;
	H_out.block<8,1>(0,7) *= SCALE_B;
	H_out.block<3,8>(0,0) *= SCALE_XI_ROT;
	H_out.block<3,8>(3,0) *= SCALE_XI_TRANS;
	H_out.block<1,8>(6,0) *= SCALE_A;
	H_out.block<1,8>(7,0) *= SCALE_B;
	b_out.segment<3>(0) *= SCALE_XI_ROT;
	b_out.segment<3>(3) *= SCALE_XI_TRANS;
	b_out.segment<1>(6) *= SCALE_A;
	b_out.segment<1>(7) *= SCALE_B;

	// hwjdebug ----------------
	// printf(" ! AFTER  finish H(0 0) = %.2f(%.2f + %.2f + %.2f + %.2f)\n", acc.H(0, 0), 
	//	acc.SSEData1m[0], acc.SSEData1m[1], acc.SSEData1m[2], acc.SSEData1m[3]);
	// cout << "H = \n" << acc.H << endl;
	// --------------------------
	
	// TODO
	// 思考如何对直线误差求导(可以用武大那篇文章的思路,用梯度和直线的方向差)
	// 先用ceres的自动求导,实验一下 这样做是否可行
}



// 返回值： 0：总能量 1：能量的数目 2,3,4:纯旋转和旋转位移下的像素平移量 5:残差大于阈值的百分比
Vec6 CoarseTracker::calcRes(
		int lvl, 
		const SE3 &refToNew, 	// 位姿
		AffLight aff_g2l, 		// ab	
		float cutoffTH)			// 误差的截至阈值(超过此阈值, 误差停止累计)
{
	using namespace cv;
	using namespace std;
	
	float E = 0;
	int numTermsInE = 0;
	int numTermsInWarped = 0;
	int numSaturated=0;

	int wl = w[lvl];
	int hl = h[lvl];
	Eigen::Vector3f* dINewl = newFrame->dIp[lvl];
	
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];

	// 准备变量
	Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
	Vec3f t = (refToNew.translation()).cast<float>();
	Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l).cast<float>();

	float sumSquaredShiftT=0;
	float sumSquaredShiftRT=0;
	float sumSquaredShiftNum=0;

	// maxEnergy = 2*huberTH*cutoffTH - huberTH^2
	float maxEnergy = 2*setting_huberTH*cutoffTH - setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.

	// debug图像
    MinimalImageB3* resImage = 0;
	if(debugPlot)
	{
		resImage = new MinimalImageB3(wl,hl);
		resImage->setConst(Vec3b(255,255,255));
	}




	// hwjdebug----------------------
		// Mat ref_depth = Mat::zeros(hl, wl, CV_8UC1);
		// Mat fra_depth = Mat::zeros(hl, wl, CV_8UC1);
	// ---------------------------

	int nl = pc_n[lvl];
	float* lpc_u = pc_u[lvl];
	float* lpc_v = pc_v[lvl];
	float* lpc_idepth = pc_idepth[lvl];
	float* lpc_color = pc_color[lvl];

	// 枚举最新关键帧的所有点云
	for(int i=0;i<nl;i++)
	{
		float id = lpc_idepth[i];
		float x = lpc_u[i];
		float y = lpc_v[i];

		Vec3f pt;
		float u ;
		float v ;
		float Ku;
		float Kv;

		// 投影KP到当前帧, 计算当前帧的坐标和反深度
		if(USE_PAL == 1){ // 0 1
			pt = RKi * pal_model_g->cam2world(x, y, lvl) + t*id;
			u = pt[0] / pt[2];
			v = pt[1] / pt[2];
			Vec2f Kpt = pal_model_g->world2cam(pt, lvl);
			Ku = Kpt[0];
			Kv = Kpt[1];
		}
		else{
			pt = RKi * Vec3f(x, y, 1) + t*id;
			u = pt[0] / pt[2];
			v = pt[1] / pt[2];
			Ku = fxl * u + cxl;
			Kv = fyl * v + cyl;
		}
		float new_idepth = id/pt[2];
		
		// hwjdebug ---------------
		// printf("ref [%.2f %.2f %.2f] -> new [%.2f %.2f %.2f]\n", x, y, 1.0/id, Ku, Kv, 1.0/new_idepth);
		// ref_depth.at<uchar>(y, x) = abs(1.0/id*100 - 100) ;
		// fra_depth.at<uchar>(Kv, Ku) = (1.0/new_idepth * 100); 
		// ---------------------

		// 对于第0层的某些点(i%32==0) 计算光流
		if(lvl==0 && i%32==0)
		{
			float uT, vT, KuT, KvT; 
			float uT2, vT2, KuT2, KvT2; 
			float u3, v3, Ku3, Kv3; 

			if(USE_PAL == 1){ // 0 1
				// translation only (positive)
				pal_project(x, y, id, Mat33f::Identity(), t, uT, vT, KuT, KvT);

				// translation only (negative)
				pal_project(x, y, id, Mat33f::Identity(), -t, uT2, vT2, KuT2, KvT2);

				//translation and rotation (negative)
				pal_project(x, y, id, RKi, -t, u3, v3, Ku3, Kv3);
			}
			else{
				// translation only (positive)
				Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t*id;
				uT = ptT[0] / ptT[2];
				vT = ptT[1] / ptT[2];
				KuT = fxl * uT + cxl;
				KvT = fyl * vT + cyl;

				// translation only (negative)
				Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t*id;
				uT2 = ptT2[0] / ptT2[2];
				vT2 = ptT2[1] / ptT2[2];
				KuT2 = fxl * uT2 + cxl;
				KvT2 = fyl * vT2 + cyl;

				//translation and rotation (negative)
				Vec3f pt3 = RKi * Vec3f(x, y, 1) - t*id;
				u3 = pt3[0] / pt3[2];
				v3 = pt3[1] / pt3[2];
				Ku3 = fxl * u3 + cxl;
				Kv3 = fyl * v3 + cyl;
			}

			//translation and rotation (positive)
			//already have it.
			// 假设纯位移，点的移动值
			sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
			sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
			// 旋转+位移 点的移动值
			sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
			sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
			sumSquaredShiftNum+=2;
		}

		// 检查越界
		if(USE_PAL == 1 || USE_PAL == 2){
			if(!(pal_check_in_range_g(Ku, Kv, 3, lvl) && new_idepth > 0))
				continue;
		}
		else{
			if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0))
				continue;
		}


		// 亮度差
		float refColor = lpc_color[i];
        Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
        if(!std::isfinite((float)hitColor[0])) 
			continue;
        float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);


		// 误差太大了只累加(最大能量)
		if(fabs(residual) > cutoffTH)
		{
			E += maxEnergy;
			numTermsInE++;
			numSaturated++;

			if(debugPlot) 
				resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0,0,255));
		}
		// 误差还行，进一步缓存一些东西
		else
		{
			E += hw *residual*residual*(2-hw);
			numTermsInE++;

			buf_warped_idepth[numTermsInWarped] = new_idepth;
			buf_warped_u[numTermsInWarped] = u;
			buf_warped_v[numTermsInWarped] = v;
			buf_warped_dx[numTermsInWarped] = hitColor[1];
			buf_warped_dy[numTermsInWarped] = hitColor[2];
			buf_warped_residual[numTermsInWarped] = residual;
			buf_warped_weight[numTermsInWarped] = hw;
			buf_warped_refColor[numTermsInWarped] = lpc_color[i];
			numTermsInWarped++;

			if(debugPlot) 
				resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(residual+128,residual+128,residual+128));
		}
	}

	// 凑够4的倍数
	while(numTermsInWarped%4!=0)
	{
		buf_warped_idepth[numTermsInWarped] = 0;
		buf_warped_u[numTermsInWarped] = 0;
		buf_warped_v[numTermsInWarped] = 0;
		buf_warped_dx[numTermsInWarped] = 0;
		buf_warped_dy[numTermsInWarped] = 0;
		buf_warped_residual[numTermsInWarped] = 0;
		buf_warped_weight[numTermsInWarped] = 0;
		buf_warped_refColor[numTermsInWarped] = 0;
		numTermsInWarped++;
	}
	buf_warped_n = numTermsInWarped;

	if(debugPlot)
	{
		IOWrap::displayImage("RES", resImage, true);
		IOWrap::waitKey(0);
		delete resImage;
	}

	// hwjdebug----------------------------

		// Mat ref_depth_show, fra_depth_show;
		// resize(ref_depth, ref_depth_show, Size(w[0], h[0]));
		// resize(fra_depth, fra_depth_show, Size(w[0], h[0]));
		// imshow("frameDepth", fra_depth_show);
		// imshow("refDepth", ref_depth_show);
		// moveWindow("refDepth", 50, 50);
		// moveWindow("frameDepth", 50+w[0]+50, 50);
		// waitKey();

	// -----------------------

	// 直线误差
	float E_line = 0;
	if(lvl == 0 && init_method_g == "line")
	{
		int line_num = lastRef->line_u.size();
		for(int i=0; i<line_num; i++){
			E_line += 0;
		}
	}

	Vec6 rs;
	rs[0] = E; 						//总能量
	rs[1] = numTermsInE; 			//残差数目

	rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);	//位移导致的平均每个点的光流
	rs[3] = E_line;										// 直线误差// TODO
	rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);	// RT导致的平均光流

	rs[5] = numSaturated / (float)numTermsInE;			// 超大残差占总残差项的比例

	return rs;
}



void CoarseTracker::setCoarseTrackingRef(
		std::vector<FrameHessian*> frameHessians)
{
	assert(frameHessians.size()>0);
	// 设置最新关键帧为Ref
	lastRef = frameHessians.back();

	// 构造深度图
	makeCoarseDepthL0(frameHessians);

	// 设置参考ID
	refFrameID = lastRef->shell->id;

	// 获取光度ab
	lastRef_aff_g2l = lastRef->aff_g2l();

	firstCoarseRMSE=-1;
}

bool CoarseTracker::trackNewestCoarse(
		FrameHessian* newFrameHessian,				// [输入] 当前帧
		SE3 &lastToNew_out, AffLight &aff_g2l_out, 	// [输入/出] 初始位姿 初始光度ab
		int coarsestLvl,							// 最粗的金字塔等级
		Vec5 minResForAbort,						// 达到这个阈值就可以停止了
		IOWrap::Output3DWrapper* wrap)				
{

	debugPlot = setting_render_displayCoarseTrackingFull;
	debugPrint = false;

	// hwjdebug--------------
	using namespace std;
	// debugPlot = true;
	// debugPrint = true;
	// --------------------

	assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

	// 初始化
	lastResiduals.setConstant(NAN);
	lastFlowIndicators.setConstant(1000);
	newFrame = newFrameHessian;
	int maxIterations[] = {10,20,50,50,50};
	float lambdaExtrapolationLimit = 0.001;

	SE3 refToNew_current = lastToNew_out;
	AffLight aff_g2l_current = aff_g2l_out;

	bool haveRepeated = false;

	// 从金字塔最高层向下主层追踪	
	for(int lvl=coarsestLvl; lvl>=0; lvl--)
	{
		Mat88 H; Vec8 b;
		float levelCutoffRepeat=1;
		// 计算当前位姿的残差
		Vec6 resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH*levelCutoffRepeat);

		// hwjdebug--------------------------

		// printf("\n - LVL %d start \n", lvl);
		// SE3 pose_test = SE3::exp(Vec6::Zero());
		// for(int i=0; i<100; i++){
		// 	resOld = calcRes(lvl, pose_test, aff_g2l_current, setting_coarseCutoffTH*levelCutoffRepeat);
		// 	Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_current).cast<float>();
		// 	printf(" res = %.3f \t ",
		// 			resOld[0] / resOld[1]);
		// 	std::cout << pose_test.log().transpose() << endl;
		// 	Vec6 inc_se3_test;
		// 	inc_se3_test << -0.001, 0, 0, 0, 0, 0;
		// 	pose_test = SE3::exp(inc_se3_test) * pose_test;
		// }
		// continue;

		// -----------------------------------

		// 如果大残差数目过多，放宽一点残差阈值再算一次（不能超过50）
		while(resOld[5] > 0.6 && levelCutoffRepeat < 50)
		{
			levelCutoffRepeat*=2;
			resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH*levelCutoffRepeat);

            if(debugPrint)
                printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
		}

		// 计算正规方程的H和b(SSE)
		// SE3基本没用，亮度有一点点用
		calcGSSSE(lvl, H, b, refToNew_current, aff_g2l_current);

		float lambda = 0.01;

		if(debugPrint)
		{
			printf("\n - LVL %d start \n", lvl);
			Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_current).cast<float>();
			printf("lvl %d, it %d (l=%.4f / %.4f) %s: [%.3f->%.3f] (%d -> %d) (|inc| = %f)! \n\t",
					lvl, -1, lambda, 1.0f,
					"INITIA",
					resOld[0] / resOld[1],
					resOld[0] / resOld[1],
					 0,(int)resOld[1],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF:" << aff_g2l_current.vec().transpose() <<" (rel " << relAff.transpose() << ")\n";
		}


		for(int iteration=0; iteration < maxIterations[lvl]; iteration++)
		{
			// 求解增量
			Mat88 Hl = H;
			for(int i=0;i<8;i++) 
				Hl(i,i) *= (1+lambda);
			Vec8 inc = Hl.ldlt().solve(-b);

			// 根据不同光度模式，单独设置光度增量
			if(setting_affineOptModeA < 0 && setting_affineOptModeB < 0)	// fix a, b
			{
				inc.head<6>() = Hl.topLeftCorner<6,6>().ldlt().solve(-b.head<6>());
			 	inc.tail<2>().setZero();
			}
			if(!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0)	// fix b
			{
				inc.head<7>() = Hl.topLeftCorner<7,7>().ldlt().solve(-b.head<7>());
			 	inc.tail<1>().setZero();
			}
			if(setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0))	// fix a
			{
				Mat88 HlStitch = Hl;
				Vec8 bStitch = b;
				HlStitch.col(6) = HlStitch.col(7);
				HlStitch.row(6) = HlStitch.row(7);
				bStitch[6] = bStitch[7];
				Vec7 incStitch = HlStitch.topLeftCorner<7,7>().ldlt().solve(-bStitch.head<7>());
				inc.setZero();
				inc.head<6>() = incStitch.head<6>();
				inc[6] = 0;
				inc[7] = incStitch[6];
			}

			// 如果lambda太小，步长扩大一点
			float extrapFac = 1;
			if(lambda < lambdaExtrapolationLimit) 
				extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
			inc *= extrapFac;

			// 带系数的增量
			Vec8 incScaled = inc;
			incScaled.segment<3>(0) *= SCALE_XI_ROT;
			incScaled.segment<3>(3) *= SCALE_XI_TRANS;
			incScaled.segment<1>(6) *= SCALE_A;
			incScaled.segment<1>(7) *= SCALE_B;

            if(!std::isfinite(incScaled.sum())) 
				incScaled.setZero();

			//应用增量，新的状态变量存储在xxx_new变量中
			SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
			AffLight aff_g2l_new = aff_g2l_current;
			aff_g2l_new.a += incScaled[6];
			aff_g2l_new.b += incScaled[7];
			
			// 重新计算res
			Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH*levelCutoffRepeat);

			// 误差是否下降
			bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

			if(debugPrint)
			{
				Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_new).cast<float>();
				printf("lvl %d, it %d (l=%.4f / %.4f) %s: [%.3f->%.3f] (%d -> %d) (|inc| = %f)! \n",
						lvl, iteration, lambda,
						extrapFac,
						(accept ? "ACCEPT" : "REJECT"),
						resOld[0] / resOld[1],
						resNew[0] / resNew[1],
						(int)resOld[1], (int)resNew[1],
						inc.norm());
				cout << "\tinc = " << incScaled.head<6>().transpose() << endl;
				std::cout << "\t" <<refToNew_new.log().transpose() << " AFF " << aff_g2l_new.vec().transpose() <<" (rel " << relAff.transpose() << ")\n";
			}

			// 接受增量，重新计算H和b，lambda增加
			if(accept)
			{
				calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
				resOld = resNew;
				aff_g2l_current = aff_g2l_new;
				refToNew_current = refToNew_new;
				lambda *= 0.5;
			}
			// 拒绝增量，lambda增加
			else
			{
				lambda *= 4;
				if(lambda < lambdaExtrapolationLimit) 
					lambda = lambdaExtrapolationLimit;
			}

			// 增量过小，退出
			if(!(inc.norm() > 1e-3)) 
			{
				if(debugPrint)
					printf("inc(%f) too small, break!\n", inc.norm());
				break;
			}
		}

		// set last residual for that level, as well as flow indicators.
		// 保存一下最终的残差
		lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
		lastFlowIndicators = resOld.segment<3>(2); // resOle[2 3 4] 像素坐标系的位移量
		//这个变量似乎没用，恒等于NaN
		if(lastResiduals[lvl] > 1.5*minResForAbort[lvl]) 
			return false;

		// 如果截至阈值大于1,那么就再来重复一遍
		if(levelCutoffRepeat > 1 && !haveRepeated)
		{
			lvl++;
			haveRepeated=true;
			printf("REPEAT LEVEL!\n");
		}
	} // 金字塔level

	if(init_method_g == "line"){

		// test line -------------
		const int ip_can[] = {1,2,3,4,5};
		for(int ip=0; ip<sizeof(ip_can)/sizeof(int); ip++)
		for(float delta = 0.04; delta >=-0.04; delta -= 0.001)
		// float delta = 0;	
		{

			SE3 pose_test = refToNew_current;
			printf("[Test Pose] delta pose[%d]=%.4f  -->  \n", ip_can[ip], delta);
			// printf("%d %.4f\t", ip_can[ip], delta);
			Vec6 pose = pose_test.log();
			pose[ip_can[ip]] += delta;
			pose_test = SE3::exp(pose);
			refinePose(pose_test);
		}
		// ------------------

		// refinePose(refToNew_current);
	}


	// set!
	// 保存更新结果
	lastToNew_out = refToNew_current;
	aff_g2l_out = aff_g2l_current;

	// 如果光度变化太大，返回false
	if((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2))
	|| (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
		return false;

	Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_out).cast<float>();

	if((setting_affineOptModeA == 0 && (fabsf(logf((float)relAff[0])) > 1.5))
	|| (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200)))
		return false;

	if(setting_affineOptModeA < 0) 
		aff_g2l_out.a=0;
	if(setting_affineOptModeB < 0) 
		aff_g2l_out.b=0;

	return true;
}



void CoarseTracker::debugPlotIDepthMap(float* minID_pt, float* maxID_pt, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;


	int lvl = 0;

	{
		std::vector<float> allID;
		for(int i=0;i<h[lvl]*w[lvl];i++)
		{
			if(idepth[lvl][i] > 0)
				allID.push_back(idepth[lvl][i]);
		}
		std::sort(allID.begin(), allID.end());
		int n = allID.size()-1;

		float minID_new = allID[(int)(n*0.05)];
		float maxID_new = allID[(int)(n*0.95)];

		float minID, maxID;
		minID = minID_new;
		maxID = maxID_new;
		if(minID_pt!=0 && maxID_pt!=0)
		{
			if(*minID_pt < 0 || *maxID_pt < 0)
			{
				*maxID_pt = maxID;
				*minID_pt = minID;
			}
			else
			{

				// slowly adapt: change by maximum 10% of old span.
				float maxChange = 0.3*(*maxID_pt - *minID_pt);

				if(minID < *minID_pt - maxChange)
					minID = *minID_pt - maxChange;
				if(minID > *minID_pt + maxChange)
					minID = *minID_pt + maxChange;


				if(maxID < *maxID_pt - maxChange)
					maxID = *maxID_pt - maxChange;
				if(maxID > *maxID_pt + maxChange)
					maxID = *maxID_pt + maxChange;

				*maxID_pt = maxID;
				*minID_pt = minID;
			}
		}


		MinimalImageB3 mf(w[lvl], h[lvl]);
		mf.setBlack();
		for(int i=0;i<h[lvl]*w[lvl];i++)
		{
			int c = lastRef->dIp[lvl][i][0]*0.9f;
			if(c>255) c=255;
			mf.at(i) = Vec3b(c,c,c);
		}
		int wl = w[lvl];
		for(int y=3;y<h[lvl]-3;y++)
			for(int x=3;x<wl-3;x++)
			{
				int idx=x+y*wl;
				float sid=0, nid=0;
				float* bp = idepth[lvl]+idx;

				if(bp[0] > 0) {sid+=bp[0]; nid++;}
				if(bp[1] > 0) {sid+=bp[1]; nid++;}
				if(bp[-1] > 0) {sid+=bp[-1]; nid++;}
				if(bp[wl] > 0) {sid+=bp[wl]; nid++;}
				if(bp[-wl] > 0) {sid+=bp[-wl]; nid++;}

				if(bp[0] > 0 || nid >= 3)
				{
					float id = ((sid / nid)-minID) / ((maxID-minID));
					mf.setPixelCirc(x,y,makeJet3B(id));
					//mf.at(idx) = makeJet3B(id);
				}
			}
        //IOWrap::displayImage("coarseDepth LVL0", &mf, false);


        for(IOWrap::Output3DWrapper* ow : wraps)
            ow->pushDepthImage(&mf);

		if(debugSaveImages)
		{
			char buf[1000];
			snprintf(buf, 1000, "images_out/predicted_%05d_%05d.png", lastRef->shell->id, refFrameID);
			IOWrap::writeImage(buf,&mf);
		}

	}
}



void CoarseTracker::debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;
    int lvl = 0;
    MinimalImageF mim(w[lvl], h[lvl], idepth[lvl]);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImageFloat(&mim, lastRef);
}











CoarseDistanceMap::CoarseDistanceMap(int ww, int hh)
{
	fwdWarpedIDDistFinal = new float[ww*hh/4];

	bfsList1 = new Eigen::Vector2i[ww*hh/4];
	bfsList2 = new Eigen::Vector2i[ww*hh/4];

	int fac = 1 << (pyrLevelsUsed-1);


	coarseProjectionGrid = new PointFrameResidual*[2048*(ww*hh/(fac*fac))];
	coarseProjectionGridNum = new int[ww*hh/(fac*fac)];

	w[0]=h[0]=0;
}
CoarseDistanceMap::~CoarseDistanceMap()
{
	delete[] fwdWarpedIDDistFinal;
	delete[] bfsList1;
	delete[] bfsList2;
	delete[] coarseProjectionGrid;
	delete[] coarseProjectionGridNum;
}




//把关键帧的点投影到当前帧，计算距离图
void CoarseDistanceMap::makeDistanceMap(
		std::vector<FrameHessian*> frameHessians,
		FrameHessian* frame)
{
	//!! 第1层金字塔
	int w1 = w[1];
	int h1 = h[1];
	int wh1 = w1*h1;
	for(int i=0;i<wh1;i++)
		fwdWarpedIDDistFinal[i] = 1000;


	// make coarse tracking templates for latstRef.
	int numItems = 0;

	// 枚举所有除了frame的帧
	for(FrameHessian* fh : frameHessians)
	{
		if(frame == fh) 
			continue;

		// 获取位姿
		SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
		Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
		Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

		// 把所有老帧上的点投影到frame上
		for(PointHessian* ph : fh->pointHessians)
		{
			assert(ph->status == PointHessian::ACTIVE);

			int u, v;
			if(USE_PAL == 1){ // 0 1 2
				Vec3f ptp_pal = KRKi * pal_model_g->cam2world(ph->u, ph->v, 1) + Kt * ph->idepth_scaled;
				Vec2f ptp_pal2D = pal_model_g->world2cam(ptp_pal, 1);
				u = ptp_pal2D[0];
				v = ptp_pal2D[1];
				if(!pal_check_in_range_g(u, v, 1, 1))
					continue;
			}
			else{
				Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*ph->idepth_scaled;
				u = ptp[0] / ptp[2] + 0.5f;
				v = ptp[1] / ptp[2] + 0.5f;
				if(USE_PAL == 0){
					if(!(u > 0 && v > 0 && u < w[1] && v < h[1])) 
						continue;
				}
				else if(USE_PAL == 2){
					if(!pal_check_in_range_g(u, v, 1, 1))
						continue;
				}
			}
			fwdWarpedIDDistFinal[u+w1*v]=0;
			bfsList1[numItems] = Eigen::Vector2i(u,v);
			numItems++;
		}
	}

	growDistBFS(numItems);
}




void CoarseDistanceMap::makeInlierVotes(std::vector<FrameHessian*> frameHessians)
{

}


// 利用BFS计算所有点到最近的有深度点的距离
void CoarseDistanceMap::growDistBFS(int bfsNum)
{
	assert(w[0] != 0);
	int w1 = w[1], h1 = h[1];
	for(int k=1;k<40;k++)
	{
		int bfsNum2 = bfsNum;
		std::swap<Eigen::Vector2i*>(bfsList1,bfsList2);
		bfsNum=0;

		if(k%2==0)
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) 
					continue;
				int idx = x + y * w1;

				if(fwdWarpedIDDistFinal[idx+1] > k)
				{
					fwdWarpedIDDistFinal[idx+1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}
			}
		}
		else
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) 
					continue;
				int idx = x + y * w1;

				if(fwdWarpedIDDistFinal[idx+1] > k)
				{
					fwdWarpedIDDistFinal[idx+1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}

				if(fwdWarpedIDDistFinal[idx+1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y-1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y-1); bfsNum++;
				}
			}
		}
	}
}

// 在距离图中添加一个新的点
void CoarseDistanceMap::addIntoDistFinal(int u, int v)
{
	if(w[0] == 0) 
		return;
	bfsList1[0] = Eigen::Vector2i(u,v);
	fwdWarpedIDDistFinal[u+w[1]*v] = 0;
	growDistBFS(1);
}



void CoarseDistanceMap::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;

		if(USE_PAL == 1){ // 0 1
// #ifdef PAL
			fx[level] = 1;
			fy[level] = 1;
			cx[level] = 0;
			cy[level] = 0;
		}
		else{
// #else
			fx[level] = fx[level-1] * 0.5;
			fy[level] = fy[level-1] * 0.5;
			cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
			cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
		}
// #endif

	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}

}
