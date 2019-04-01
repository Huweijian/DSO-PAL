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

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"
#include "util/pal_model.h"
#include "util/pal_interface.h"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0,0), thisToNext(SE3())
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	JbBuffer = new Vec10f[ww*hh];
	JbBuffer_new = new Vec10f[ww*hh];


	frameID=-1;
	fixAffine=true;
	printDebug=true;

	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		if(points[lvl] != 0) delete[] points[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}
// 初始化的跟踪
// 返回值：是否跟踪成功
bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
	newFrame = newFrameHessian;

    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushLiveFrame(newFrameHessian);

	printf("\n - COARSE TRACK Frame %d \n", newFrame->idx);

	int maxIterations[] = {5,5,10,30,50};

	alphaK = 2.5*2.5;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
	regWeight = 0.8;//*freeDebugParam4;
	couplingWeight = 1;//*freeDebugParam5;

	// 如果上一帧没有跟踪成功，重置所有层的所有点的idpeth等于1
	if(!snapped)
	{
		thisToNext.translation().setZero();
		for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
		{
			int npts = numPoints[lvl];
			Pnt* ptsl = points[lvl];
			for(int i=0;i<npts;i++)
			{
				ptsl[i].iR = 1;
				ptsl[i].idepth_new = 1;
				ptsl[i].lastHessian = 0;
			}
		}
	}

	// 初始化亮度仿射变换
	AffLight refToNew_aff_current = thisToNext_aff;
	if(firstFrame->ab_exposure>0 && newFrame->ab_exposure>0)
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure),0); // coarse approximation.

	SE3 refToNew_current = thisToNext;
	Vec3f latestRes = Vec3f::Zero();
	// 从高到低枚举金字塔
	for(int lvl=pyrLevelsUsed-1; lvl>=0; lvl--)
	{
		// 上一层的向下传播，初始化本层的
		if(lvl<pyrLevelsUsed-1)
			propagateDown(lvl+1);

		// 重置点的idepth_new和误差
		resetPoints(lvl);

		// 计算初始残差和海森
		Mat88f H,Hsc; 
		Vec8f b,bsc;
		Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
		// 应用计算的结果
		applyStep(lvl);

		float lambda = 0.1;
		float eps = 1e-4;
		int fails=0;

		if(printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: [%.3f->%.3f] %.3f+%.5f -> %.3f+%.5f  (|inc| = %f)! \n\t",
					lvl, 0, lambda,
					"INITIA",
					(resOld[0]+resOld[1]) / resOld[2],
					(resOld[0]+resOld[1]) / resOld[2],
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
			debugPlot(lvl, wraps, true); 
		}

		int iteration=0;
		while(true)
		{
			Mat88f Hl = H;
			for(int i=0;i<8;i++) 
				Hl(i,i) *= (1+lambda);
			Hl -= Hsc*(1/(1+lambda));
			Vec8f bl = b - bsc*(1/(1+lambda));

			Hl = wM * Hl * wM * (0.01f/(w[lvl]*h[lvl]));
			bl = wM * bl * (0.01f/(w[lvl]*h[lvl]));

			// 求解增量
			Vec8f inc;
			if(fixAffine)
			{
				inc.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6,6>() * (Hl.topLeftCorner<6,6>().ldlt().solve(bl.head<6>())));
				inc.tail<2>().setZero();
			}
			else
				inc = - (wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b.

			// 应用增量(SE3 AFF idepth)
			SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			doStep(lvl, lambda, inc);

			// 重新计算残差和海森
			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
			Vec3f regEnergy = calcEC(lvl); // 正则化导致的误差

			// 累计总的能量
			float eTotalNew = (resNew[0]+resNew[1]+regEnergy[1]);
			float eTotalOld = (resOld[0]+resOld[1]+regEnergy[0]);


			bool accept = eTotalOld > eTotalNew;

			if(printDebug)
			{
				printf("lvl %d, it %d (l=%.5f) %s: [%.2f->%.2f] (%.3f + %.3f + %.3f -> %.3f + %.3f + %.3f) (|inc| = %f)! \n\t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
				debugPlot(lvl, wraps, true);
			}

			if(accept)
			{

				if(resNew[1] == alphaK*numPoints[lvl]){

					snapped = true;
				}
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				applyStep(lvl);
				optReg(lvl);
				lambda *= 0.5;
				fails=0;
				if(lambda < 0.0001) lambda = 0.0001;
			}
			else
			{
				fails++;
				lambda *= 4;
				if(lambda > 10000) lambda = 10000;
			}

			// 退出优化的条件：连续拒绝2次 或 增量过小 或 迭代次数过多
			bool quitOpt = false;
			if(!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
			{
				Mat88f H,Hsc; Vec8f b,bsc;
				quitOpt = true;
			}

			if(quitOpt) 
				break;
			iteration++;
		}
		latestRes = resOld;
	}


	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	// 从更准确的底层金字塔向上传播，更新上层金字塔的iR和idepth
	for(int i=0;i<pyrLevelsUsed-1;i++)
		propagateUp(i);

	frameID++;
	
	if(!snapped) 
		snappedAt=0;

	if(snapped && snappedAt==0){
		snappedAt = frameID;
	}
	if(snapped){
		printf("!!!!初始化成功(at %d)!!!! now %d\n", snappedAt, frameID);
	}

	// 输出深度图
	// HWJDEBUG
    // debugPlot(0,wraps);
	// cv::waitKey(1000);

	// 打断后连续估计5帧，初始化成功
	return snapped && frameID > snappedAt+5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps, bool onlyCVImg)
{
	// 在不传入wraps的情况下强制输出到OpenCV Mat
	if(wraps.size() != 0){
		bool needCall = false;
		for(IOWrap::Output3DWrapper* ow : wraps)
			needCall = needCall || ow->needPushDepthImage();
		if(!needCall) return;
	}
	else{
		onlyCVImg = true;
	}


	int wl= w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

	MinimalImageB3 iRImg(wl,hl);
	// 取出图像的所有像素
	for(int i=0;i<wl*hl;i++)
		iRImg.at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);


	int npts = numPoints[lvl];

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(point->isGood)
		{
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;


	int goodCnt = 0, badCnt = 0;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;

		if(!point->isGood){
			badCnt ++;
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));
		}

		else{
			goodCnt ++;
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
		}
	}

	IOWrap::displayImage("idepth-R", &iRImg, true);

	if(onlyCVImg)
		return ;	

	for(IOWrap::Output3DWrapper* ow : wraps)
		ow->pushDepthImage(&iRImg);
}

// 计算某一层的残差，H 
// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
// 返回值：能量，alpha能量，点数
Vec3f CoarseInitializer::calcResAndGS(
		int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff,
		bool plot)
{
	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

	Mat33f R = refToNew.rotationMatrix().cast<float>();
	Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
	Vec3f t = refToNew.translation().cast<float>();
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

#ifndef PAL
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];
#endif

	Accumulator11 E;
	acc9.initialize();
	E.initialize();

	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	// 枚举这一层的所有点
	for(int i=0;i<npts;i++){

		Pnt* point = ptsl+i;
		point->maxstep = 1e10;
		// 如果这个点不好
		if(!point->isGood)
		{
			// 那就直接把这个点的能量加进去
			E.updateSingle((float)(point->energy[0]));
			point->energy_new = point->energy;
			point->isGood_new = false;
			continue;
		}

		// 如果这个点挺好，计算梯度
        VecNRf dp0;
        VecNRf dp1;
        VecNRf dp2;
        VecNRf dp3;
        VecNRf dp4;
        VecNRf dp5;
        VecNRf dp6;
        VecNRf dp7;
        VecNRf dd;
        VecNRf r;
		JbBuffer_new[i].setZero();

		// sum over all residuals.
		// 求所有pattern点的能量和(如果有一个pattern点无效，那么整个点gg)
		bool isGood = true;
		float energy=0;
		for(int idx=0;idx<patternNum;idx++)
		{
			int dx = patternP[idx][0];
			int dy = patternP[idx][1];

			// pattern点变换到新帧的相机坐标系下(u, v) -> (Ku, Kv)
#ifdef PAL
			Vec3f pt = R * pal_model_g->cam2world(point->u+dx, point->v+dy, lvl) + t*point->idepth_new;
			Vec2f p2_pal = pal_model_g->world2cam(pt, lvl);
			float u = pt[0]/pt[2];	// u v 归一化平面坐标
			float v = pt[1]/pt[2];	
			float Ku = p2_pal[0];	// Ku Kv 像素平面坐标
			float Kv = p2_pal[1];
			float new_idepth = point->idepth_new/pt[2];
			// 如果新的点出界或者深度异常，那么点设置为无效
			if(!(pal_check_in_range_g(Ku, Kv, 2, lvl) && new_idepth > 0))
			{
				isGood = false;

				// PAL debug output 
				// printf("lvl %d point %d pattern %d fail \t", lvl, i, idx);
				// if(new_idepth < 0)
				// 	printf("idep = %f\n", new_idepth);
				// else
				// 	printf("[%d, %d] u=%.2f v=%.2f\n", wl, hl, Ku, Kv);

				break;
			}

			// printf("  success\n");

#else
			Vec3f pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl;
			float Kv = fyl * v + cyl;
			float new_idepth = point->idepth_new/pt[2];
			// 如果新的点出界或者深度异常，那么点设置为无效
			if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0)){
				isGood = false;
				break;
			}
#endif

			// 亚像素梯度和亮度
			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);  // 新帧的 [亮度 dx dy]
			float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl); // 参考帧亮度

			// 如果亮度无效，gg
			if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])){
				isGood = false;
				break;
			}

			// 计算亮度差(考虑仿射变换) 累计到energy变量中
			float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw *residual*residual*(2-hw);

			// 计算一系列导数，并储存
			if(hw < 1) hw = sqrtf(hw);
#ifdef PAL
			float dxdd = (t[0]-t[2]*u); // \rho_2 / \rho1 * (tx - u'_2 * tz)
			float dydd = (t[1]-t[2]*v); // \rho_2 / \rho1 * (ty - v'_2 * tz)
			Vec2f dr2duv2(hitColor[1]*hw, hitColor[2]*hw);
			Eigen::Matrix<float, 2, 6> dx2dSE;
			Eigen::Matrix<float, 2, 3> duv2dxyz;
			pal_model_g->jacobian_xyz2uv(pt/point->idepth_new, dx2dSE, duv2dxyz);
			Vec6f dr2dSE = dr2duv2.transpose() * dx2dSE;	
			dp0[idx] = dr2dSE[0];
			dp1[idx] = dr2dSE[1];
			dp2[idx] = dr2dSE[2];
			dp3[idx] = dr2dSE[3];
			dp4[idx] = dr2dSE[4];
			dp5[idx] = dr2dSE[5];
			Vec3f dxyzdd = Vec3f(dxdd, dydd, 0);

			auto dd_scalar = dr2duv2.transpose() * duv2dxyz * dxyzdd; 
			dd[idx] = dd_scalar[0];
			// FIXME: possible bug
			float maxstep = 1.0f/ dxyzdd.norm();

#else
			float dxdd = (t[0]-t[2]*u)/pt[2]; // \rho_2 / \rho1 * (tx - u'_2 * tz)
			float dydd = (t[1]-t[2]*v)/pt[2]; // \rho_2 / \rho1 * (ty - v'_2 * tz)
			float dxInterp = hw*hitColor[1]*fxl;
			float dyInterp = hw*hitColor[2]*fyl;
			dp0[idx] = new_idepth*dxInterp;						// dE/d(SE1) = gx*fx/Z
			dp1[idx] = new_idepth*dyInterp;						// dE/d(SE2) = gy*fy/Z
			dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp);	// dE/d(SE3)
			dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;		// ...d(SE4)
			dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp;			// ...d(SE5)
			dp5[idx] = -v*dxInterp + u*dyInterp;				// ...d(SE6)
			dd[idx] = dxInterp * dxdd  + dyInterp * dydd;		// dE/d(depth)
			float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();
#endif

			dp6[idx] = - hw*r2new_aff[0] * rlR;					// dE/d(e^(aj))
			dp7[idx] = - hw*1;									// dE/d(bj)	
			r[idx] = hw*residual;								// res

			if(maxstep < point->maxstep) 
				point->maxstep = maxstep;

			// immediately compute dp*dd1' and dd*dd' in JbBuffer1.
			// pattern的所有点当作一个点来计算梯度
			JbBuffer_new[i][0] += dp0[idx]*dd[idx];
			JbBuffer_new[i][1] += dp1[idx]*dd[idx];
			JbBuffer_new[i][2] += dp2[idx]*dd[idx];
			JbBuffer_new[i][3] += dp3[idx]*dd[idx];
			JbBuffer_new[i][4] += dp4[idx]*dd[idx];
			JbBuffer_new[i][5] += dp5[idx]*dd[idx];
			JbBuffer_new[i][6] += dp6[idx]*dd[idx];
			JbBuffer_new[i][7] += dp7[idx]*dd[idx];
			JbBuffer_new[i][8] += r[idx]*dd[idx];
			JbBuffer_new[i][9] += dd[idx]*dd[idx];
		} //枚举一个点的所有pattern

		// 如果跟踪点挂了，或者误差过大
		if(!isGood || energy > point->outlierTH*20)
		{
			// 单纯累计误差，不继续算了
			E.updateSingle((float)(point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}

		// 计算完成pattern 8个点的误差，累计能量E，累计8个梯度到acc9中
		// add into energy.
		E.updateSingle(energy);
		point->isGood_new = true;
		point->energy_new[0] = energy;

		// update Hessian matrix.
		// 尽可能SSE，剩余的单独加入
		for(int i=0;i+3<patternNum;i+=4)
			acc9.updateSSE(
					_mm_load_ps(((float*)(&dp0))+i),
					_mm_load_ps(((float*)(&dp1))+i),
					_mm_load_ps(((float*)(&dp2))+i),
					_mm_load_ps(((float*)(&dp3))+i),
					_mm_load_ps(((float*)(&dp4))+i),
					_mm_load_ps(((float*)(&dp5))+i),
					_mm_load_ps(((float*)(&dp6))+i),
					_mm_load_ps(((float*)(&dp7))+i),
					_mm_load_ps(((float*)(&r))+i));
		for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			acc9.updateSingle(
					(float)dp0[i],(float)dp1[i],(float)dp2[i],(float)dp3[i],
					(float)dp4[i],(float)dp5[i],(float)dp6[i],(float)dp7[i],
					(float)r[i]);
	} // 枚举所有点

	E.finish();
	acc9.finish();


	// 累加energy[1]到E中
	// calculate alpha energy, and decide if we cap it.
	Accumulator11 EAlpha;
	EAlpha.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		// 点不好，用旧的能量
		if(!point->isGood_new)
		{
			E.updateSingle((float)(point->energy[1]));
		}
		// 点好，重新计算能量，是反深度的平方
		else
		{
			point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1);
			E.updateSingle((float)(point->energy_new[1]));
		}
	}
	EAlpha.finish();

	// 这个Ealpha似乎没用，恒等于0，
	// 计算alphaEnergy = alphaW * norm(t) * npts 
	// AlphaEnergy用来表征当前估计的质量（位移越大，点数越多，质量越高）
	// 	等于位移乘以点数
	float alphaEnergy = alphaW*(EAlpha.A + refToNew.translation().squaredNorm() * npts);

	// compute alpha opt.
	float alphaOpt;
	// 如果位移乘以点数足够大，也就是位移足够多(和点数无关)
	if(alphaEnergy > alphaK*npts)
	{
		// 那么不进行alpha操作
		alphaOpt = 0;
		// alpha能量等于阈值
		alphaEnergy = alphaK*npts;
	}
	else
	{
		// 位移不够大，需要进行alpha操作，值等于alphaW(150*150)
		alphaOpt = alphaW;
	}

	// 更新每个点的lastH，更新JBuffer[8 9]
	// 累加新的JBuffer
	acc9SC.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
			continue;

		point->lastHessian_new = JbBuffer_new[i][9];

		// 更新JBuffer8和9
		// 如果alphaOpt不等于0 （位移不够大）那么增加一点噪声(150*150)
		JbBuffer_new[i][8] += alphaOpt*(point->idepth_new - 1);
		JbBuffer_new[i][9] += alphaOpt;

		// 如果alphaOpt=0 (位移足够大) 也增加一点耦合噪声 couplingWeight=1
		if(alphaOpt==0)
		{
			JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}

		JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();

	// 分块取出海森矩阵
	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	H_out = acc9.H.topLeftCorner<8,8>();// / acc9.num;
	b_out = acc9.H.topRightCorner<8,1>();// / acc9.num;
	H_out_sc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num;
	b_out_sc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;

	// 更新H和b的一些值 
	H_out(0,0) += alphaOpt*npts;
	H_out(1,1) += alphaOpt*npts;
	H_out(2,2) += alphaOpt*npts;

	Vec3f tlog = refToNew.log().head<3>().cast<float>();
	b_out[0] += tlog[0]*alphaOpt*npts;
	b_out[1] += tlog[1]*alphaOpt*npts;
	b_out[2] += tlog[2]*alphaOpt*npts;

	return Vec3f(E.A, alphaEnergy ,E.num);
}

float CoarseInitializer::rescale()
{
	float factor = 20*thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

	return factor;
}


Vec3f CoarseInitializer::calcEC(int lvl)
{
	if(!snapped) 
		return Vec3f(0,0,numPoints[lvl]);

	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(!point->isGood_new) 
			continue;
		float rOld = (point->idepth - point->iR);
		float rNew = (point->idepth_new - point->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}

// 平滑lvl层的iR。
// 如果打断了，利用knn初始化，
// 如果没打断，初始化为1
void CoarseInitializer::optReg(int lvl)
{
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];

	// 如果没有打断，那么初始化iR=1
	if(!snapped)
	{
		for(int i=0;i<npts;i++)
			ptsl[i].iR = 1;
		return;
	}

	// 如果打断了，利用每个点的knn优化当前点的iR
	for(int i=0;i<npts;i++)
	{
		// 如果点有效
		Pnt* point = ptsl+i;
		if(!point->isGood) 
			continue;

		float idnn[10];
		int nnn=0;
		// 枚举所有knn，保存还行的点的iR到idnn
		for(int j=0;j<10;j++)
		{
			if(point->neighbours[j] == -1) 
				continue;
			Pnt* other = ptsl+point->neighbours[j];
			if(!other->isGood) 
				continue;
			idnn[nnn] = other->iR;
			nnn++;
		}
		
		// 如果knn还行的点大于2个，
		if(nnn > 2)
		{
			// 求idnn的中值(保存在idnn[nnn/2])
			std::nth_element(idnn,idnn+nnn/2,idnn+nnn);
			// 把knn中值的iR融合到当前点的idepth中
			point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
		}
	}

}



// 从底层金字塔向上传播，更新上层金字塔的iR和idepth
void CoarseInitializer::propagateUp(int srcLvl)
{
	assert(srcLvl+1<pyrLevelsUsed);
	// set idepth of target

	int nptss= numPoints[srcLvl];
	int nptst= numPoints[srcLvl+1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl+1];

	// set to zero.
	// 初始化初始化高一层为0
	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		parent->iR=0;
		parent->iRSumNum=0;
	}

	// 枚举当前层
	for(int i=0;i<nptss;i++)
	{
		Pnt* point = ptss+i;
		if(!point->isGood) 
			continue;
		Pnt* parent = ptst + point->parent;
		// 爸爸的iR等于孩子的点×海森
		parent->iR += point->iR * point->lastHessian;
		// 爸爸的RSumNum累加
		parent->iRSumNum += point->lastHessian;
	}

	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		// 如果爸爸点之前更新过了
		if(parent->iRSumNum > 0)
		{
			// 更新爸爸点的idepth和iR
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
			parent->isGood = true;
		}
	}

	// 对爸爸层执行平滑
	optReg(srcLvl+1);
}

// 从srcLvl层向下传播一层，初始化isGood iR idepth
void CoarseInitializer::propagateDown(int srcLvl)
{
	assert(srcLvl>0);
	// set idepth of target

	Pnt* ptss = points[srcLvl];
	int nptst= numPoints[srcLvl-1];
	Pnt* ptst = points[srcLvl-1];

	// 枚举下一层的所有点
	for(int i=0;i<nptst;i++)
	{
		Pnt* point = ptst+i;
		Pnt* parent = ptss+point->parent;

		// 如果爸爸不太行，那就算了
		if(!parent->isGood || parent->lastHessian < 0.1) 
			continue;

		// 如果当前点不不好
		if(!point->isGood)
		{
			// 用爸爸的iR 初始化当前的iR idepth idepth_new
			point->iR = point->idepth = point->idepth_new = parent->iR;
			point->isGood=true;
			point->lastHessian=0;
		}
		// 如果当前点还不错
		else
		{
			// 和爸爸的iR融合
			// iR = (iR*H*2 + iR*H) / (H*2 * H)
			float newiR = (point->iR*point->lastHessian*2 + parent->iR*parent->lastHessian) / (point->lastHessian*2+parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}

	// 初始化下一层的iR
	optReg(srcLvl-1);
}


void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
				dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1][0] +
													dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
			dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
		}
	}
}

// 设置初始化的第一帧图像
void CoarseInitializer::setFirst(CalibHessian* HCalib, FrameHessian* newFrameHessian)
{

	// 本地化相机参数
	makeK(HCalib);
	firstFrame = newFrameHessian;

	PixelSelector sel(w[0],h[0]);

	float* statusMap = new float[w[0]*h[0]];
	bool* statusMapB = new bool[w[0]*h[0]];

	float densities[] = {0.03,0.05,0.15,0.5,1};
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
#ifdef PAL //修改点的密度
		int r0 = pal_model_g->mask_radius[0];
		int r1 = pal_model_g->mask_radius[1];
		densities[lvl] *= 3.14*(r1*r1 - r0*r0) / w[0]*h[0];
#endif
		sel.currentPotential = 3;
		int npts;
		if(lvl == 0)
			// 对于原始图像，选择一些梯度较大的点
			npts = sel.makeMaps(firstFrame, statusMap, densities[lvl]*w[0]*h[0],1,false,2);
		else
			// 对于其他金字塔图像
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);

		if(points[lvl] != 0) delete[] points[lvl];
		points[lvl] = new Pnt[npts];

		// set idepth map to initially 1 everywhere.
		int wl = w[lvl], hl = h[lvl];
		Pnt* pl = points[lvl];
		int nl = 0;
		for(int y=patternPadding+1;y<hl-patternPadding-2;y++){
			for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
			{
				// 如果这个点被选中了
				if((lvl!=0 && statusMapB[x+y*wl]) || (lvl==0 && statusMap[x+y*wl] != 0))
				{
#ifdef PAL // 排除外部的点
					if(!pal_check_in_range_g(x, y, patternPadding+1, lvl)){
						continue;
					}
#endif
					// 初始化这个点的信息
					pl[nl].u = x+0.1;
					pl[nl].v = y+0.1;
					pl[nl].idepth = 1;
					pl[nl].iR = 1;
					pl[nl].isGood=true;
					pl[nl].energy.setZero();
					pl[nl].lastHessian=0;
					pl[nl].lastHessian_new=0;
					pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl]; //高层金字塔，type=1,底层金字塔，type= 1 ?????
					pl[nl].outlierTH = patternNum*setting_outlierTH;

					Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
					float sumGrad2=0;// 所有pattern点的梯度和
					for(int idx=0;idx<patternNum;idx++)
					{
						int dx = patternP[idx][0];
						int dy = patternP[idx][1];
						float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm(); //pattern点的梯度绝对值
						sumGrad2 += absgrad;
					}

					nl++;
					assert(nl <= npts);
				}
			}
		}

		numPoints[lvl]=nl;
	}

	delete[] statusMap;
	delete[] statusMapB;

	// 计算每个点的KNN及在上层的爸爸点
	makeNN();

	// 初始化一些变量
	thisToNext=SE3();
	snapped = false;
	frameID = snappedAt = 0;

	// 初始化dGrads为0向量
	for(int i=0;i<pyrLevelsUsed;i++)
		dGrads[i].setZero();

	// 可视化 pal 点选择结果 
	// std::vector<IOWrap::Output3DWrapper*> place_holder;
	// debugPlot(0, place_holder, true);
	// cv::waitKey();
}

// 重置某层点的误差 and idepth_new，
void CoarseInitializer::resetPoints(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{

		pts[i].energy.setZero();
		pts[i].idepth_new = pts[i].idepth;

		// 如果是最高层，而且点不好
		if(lvl==pyrLevelsUsed-1 && !pts[i].isGood)
		{
			// 用knn的iR平均值初始化当前点
			float snd=0, sn=0;
			for(int n = 0;n<10;n++)
			{
				if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) 
					continue;
				
				snd += pts[pts[i].neighbours[n]].iR;
				sn += 1;
			}

			if(sn > 0)
			{
				pts[i].isGood=true;
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
			}
			// 实在不行就算了，不初始化了
		}
	}
}

// 更新idepth，控制在步长以内
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{
	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood) 
			continue;

		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float step = - b * JbBuffer[i][9] / (1+lambda);


		float maxstep = maxPixelStep*pts[i].maxstep;
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;

		float newIdepth = pts[i].idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

}

// 将energy isgood idepth lastHessian JB等变量从new中拷贝到正常变量中
void CoarseInitializer::applyStep(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}
// 从相机参数中本地化信息
void CoarseInitializer::makeK(CalibHessian* HCalib)
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
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
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

// 搜索每个点的KNN点，记录点的idx和距离
// 搜索每个点在上层金字塔的对应点，记为爸爸并 记录爸爸距离
void CoarseInitializer::makeNN()
{
	const float NNDistFactor=0.05;

	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	KDTree* indexes[PYR_LEVELS];
	FLANNPointcloud pcs[PYR_LEVELS];
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		// 每一层创建一个FLANN点云和KD树
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	const int nn=10;

	// find NN & parents
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for(int i=0;i<npts;i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u,pts[i].v);
			// 搜索N邻居，结果存储再resultSet中，pt是待搜索的点
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx=0;
			float sumDF = 0;
			for(int k=0;k<nn;k++)
			{
				// 记录每个点的邻居idx，距离=exp(-dist*factor)/sumDF*10
				pts[i].neighbours[myidx]=ret_index[k];
				float df = expf(-ret_dist[k]*NNDistFactor);
				sumDF += df;
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts);
				myidx++;
			}
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF;


			if(lvl < pyrLevelsUsed-1 )
			{
				resultSet1.init(ret_index, ret_dist);
				pt = pt*0.5f-Vec2f(0.25f,0.25f);// 点在上层金字塔的位置
				//查找点在上层金字塔中最接近的点
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());
				// 保存为这个点的爸爸和爸爸距离
				pts[i].parent = ret_index[0];
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor);

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
			}
			else
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}

	// done.

	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];
}
}

