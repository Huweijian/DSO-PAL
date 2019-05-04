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



#include "FullSystem/ImmaturePoint.h"
#include "util/FrameShell.h"
#include "FullSystem/ResidualProjections.h"
#include "util/pal_interface.h"

namespace dso
{
//计算总梯度海森，每个pattern点的权重，计算能量阈值
ImmaturePoint::ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib)
: u(u_), v(v_), host(host_), my_type(type), idepth_min(0), idepth_max(NAN), lastTraceStatus(IPS_UNINITIALIZED)
{
	// 梯度.T * 梯度
	gradH.setZero();
	grad.setZero();

	//枚举每个pattern点
	for(int idx=0;idx<patternNum;idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		//插值的点
        Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u+dx, v+dy,wG[0]);

		color[idx] = ptc[0];
		if(!std::isfinite(color[idx])){
			energyTH=NAN; 
			return;
		}
		grad += ptc.tail<2>();
		gradH += ptc.tail<2>()  * ptc.tail<2>().transpose();

		weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
	}
	grad = grad / patternNum;

	// 能量阈值(和pattern点数有关)
	energyTH = patternNum*setting_outlierTH;
	energyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;

	idepth_GT=0;
	quality=10000;
}

ImmaturePoint::~ImmaturePoint()
{
}



/*
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 */
ImmaturePointStatus ImmaturePoint::traceOn(FrameHessian* frame,const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f& hostToFrame_affine, CalibHessian* HCalib, bool debugPrint)
{
	using namespace cv;
	using namespace std;
	if(lastTraceStatus == ImmaturePointStatus::IPS_OOB) 
		return lastTraceStatus;

	debugPrint = false;//rand()%100==0;

	// hwjdebug ----------------
	Mat img_now;
	if(frame->shell->incoming_id >= 8848){
		debugPrint = true;
		img_now = IOWrap::getOCVImg(frame->dI,wG[0], hG[0]);	
		cvtColor(img_now, img_now, COLOR_GRAY2BGR);

	}
	// -------------------

	// // hwjdebug 显示参考帧---------------

	if(debugPrint){

		Mat img_ref = IOWrap::getOCVImg(host->dI, wG[0], hG[0]);	
		circle(img_ref, Point(u, v), 5, 255);
		imshow("img_ref", img_ref);
		moveWindow("img_ref", 50, 50);
	}
	// // -------------------

	// 最大极线长度
	float maxPixSearch = (wG[0]+hG[0])*setting_maxPixSearch;

	if(debugPrint)
		printf("trace pt(%.1f %.1f) frame(%d-%d) idepth(%.4f-%.4f). t = [%.4f %.4f %.4f]!\n",
				u,v,
				host->shell->id, frame->shell->id,
				idepth_min, idepth_max,
				hostToFrame_Kt[0],hostToFrame_Kt[1],hostToFrame_Kt[2]);

//	const float stepsize = 1.0;				// stepsize for initial discrete search.
//	const int GNIterations = 3;				// max # GN iterations
//	const float GNThreshold = 0.1;				// GN stop after this stepsize.
//	const float extraSlackOnTH = 1.2;			// for energy-based outlier check, be slightly more relaxed by this factor.
//	const float slackInterval = 0.8;			// if pixel-interval is smaller than this, leave it be.
//	const float minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.
	// ============== project min and max. return if one of them is OOB ===================
	Vec3f pr ;
	Vec3f ptpMin; //对于PAL,ptpMIN是最小反深度的3D坐标,对于pin ptpMIN是临时变量,无意义
	float uMin;
	float vMin;
	// PAL 极线搜索Min点确定
	if(USE_PAL == 1){ // 0 1 2 
		pr = hostToFrame_KRKi * pal_model_g->cam2world(u, v);
		ptpMin = pr + hostToFrame_Kt*idepth_min;
		Vec2f ptpMin2D = pal_model_g->world2cam(ptpMin);
		uMin = ptpMin2D[0];
		vMin = ptpMin2D[1];

		if(!(pal_check_in_range_g(uMin, vMin, 5, 0))){
			if(debugPrint) {
				printf(" $ OOB uMin %f %f - %f %f %f (id %f-%f)! ",
					u,v,uMin, vMin,  ptpMin[2], idepth_min, idepth_max);
				cout << ptpMin.transpose() << endl;
			}
			lastTraceUV = Vec2f(-1,-1);
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}

	}
	else{
		// 注意:这里uv是像素坐标,DSO直接在像素坐标系上执行的旋转平移操作
		// 因此 pr的xy是像素坐标,z是深度值
		pr = hostToFrame_KRKi * Vec3f(u, v, 1);
		ptpMin = pr + hostToFrame_Kt*idepth_min;
		uMin = ptpMin[0] / ptpMin[2];
		vMin = ptpMin[1] / ptpMin[2];

		bool is_oob = false;
		if(USE_PAL == 0)
			if(!(uMin > 4 && vMin > 4 && uMin < wG[0]-5 && vMin < hG[0]-5)){
				is_oob = true;
			}
		else if(USE_PAL == 2){
			if(!(pal_check_in_range_g(uMin, vMin, 5, 0))){
				is_oob = true;
			}
		}

		if(is_oob){
			if(debugPrint) printf(" $ OOB uMin %f %f - %f %f %f (id %f-%f)!\n",
					u,v,uMin, vMin,  ptpMin[2], idepth_min, idepth_max);
			lastTraceUV = Vec2f(-1,-1);
			lastTracePixelInterval=0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}

	}

	float dist;
	float uMax;
	float vMax;
	Vec3f ptpMax;	//对于PAL,ptpMax是最大反深度的3D坐标,对于pin ptpMax是临时变量,无意义;
	// max深度是有效的
	if(std::isfinite(idepth_max))
	{
		ptpMax = pr + hostToFrame_Kt*idepth_max;

		// PAL极线搜索max点确定
		if(USE_PAL == 1){ // 0 1 2
			Vec2f ptpMax2D = pal_model_g->world2cam(ptpMax);
			uMax = ptpMax2D[0];
			vMax = ptpMax2D[1];

			if(!(pal_check_in_range_g(uMax, vMax, 5, 0))){
				if(debugPrint) 
					printf("OOB uMax  %f %f - %f %f!\n",u,v, uMax, vMax);
				lastTraceUV = Vec2f(-1,-1);
				lastTracePixelInterval=0;
				return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
			}
		}
		else{
			uMax = ptpMax[0] / ptpMax[2];
			vMax = ptpMax[1] / ptpMax[2];

			bool is_oob = false;
			if(USE_PAL == 0){
				if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
					is_oob = true;
			}
			else if (USE_PAL == 2){
				if(!(pal_check_in_range_g(uMax, vMax, 5, 0)))
					is_oob = true;
			}
				
			if(is_oob){
				if(debugPrint) 
					printf("OOB uMax  %f %f - %f %f!\n",u,v, uMax, vMax);
				lastTraceUV = Vec2f(-1,-1);
				lastTracePixelInterval=0;
				return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
			}	

		}

	

		// ============== check their distance. everything below 2px is OK (-> skip). ===================

		// 极线长度
		dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax);
		dist = sqrtf(dist);
		// 极线长度过小
		float minPolarDist = setting_trace_slackInterval;
		if(ENH_PAL){ // PAL极线匹配阈值降低,即,即使极线没那么长,也试着匹配一下
			minPolarDist = pal_get_weight(Vec2f(u, v));
			minPolarDist = 0.5*setting_trace_slackInterval;
		}
		if(dist < minPolarDist)
		{
			if(debugPrint){
				// hwjdebug -----------
				line(img_now, Point(uMax, vMax), Point(uMin, vMin), {255, 255, 255});
				imshow("img_now", img_now);
				waitKey();
				// ------------------
				printf(" $ skip, TOO CERTAIN ALREADY (dist %f)!\n", dist);
			}

			lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
			lastTracePixelInterval=dist;
			return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
		}
		assert(dist>0);
	}

	// idepth_max 无效,需要手动定义深度 
	else	
	{
		dist = maxPixSearch;

		// project to arbitrary depth to get direction.
		// 在idepthmax为 无穷的情况下确定PAL极线搜索范围
		if(USE_PAL == 1){ // 0 1 2 

			float dist_pal = 0, dist_pal_try = 0;
			float idepth_max_pal = 0.01;
			float uMax_try = 0, vMax_try = 0;
			Vec3f ptpMax_try;
			do{
				uMax = uMax_try;
				vMax = vMax_try;
				dist_pal = dist_pal_try;
				ptpMax = ptpMax_try;

				idepth_max_pal *= 3;
				ptpMax_try = pr + hostToFrame_Kt*idepth_max_pal;
				Vec2f ptpMax2D = pal_model_g->world2cam(ptpMax_try);
				uMax_try = ptpMax2D[0];
				vMax_try = ptpMax2D[1];
				float dx_pal = uMax_try - uMin;
				float dy_pal = vMax_try - vMin;
				dist_pal_try = sqrt(dx_pal*dx_pal + dy_pal*dy_pal);	
			}while(dist_pal_try < dist);
			dist = dist_pal;

			if(!(pal_check_in_range_g(uMax, vMax, 5, 0)))
			{
				if(debugPrint) 
					printf(" $ OOB uMax-coarse %f %f %f!\n", uMax, vMax,  ptpMax[2]);
				lastTraceUV = Vec2f(-1,-1);
				lastTracePixelInterval=0;
				return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
			}

		}
		else{

			ptpMax = pr + hostToFrame_Kt*0.01;
			uMax = ptpMax[0] / ptpMax[2];
			vMax = ptpMax[1] / ptpMax[2];

			// direction.
			float dx = uMax-uMin;
			float dy = vMax-vMin;
			float d = 1.0f / sqrtf(dx*dx+dy*dy);

			// set to [setting_maxPixSearch].
			uMax = uMin + dist*dx*d;
			vMax = vMin + dist*dy*d;

			// may still be out!
			bool is_oob = false;
			if(USE_PAL == 0){
				if(!(uMax > 4 && vMax > 4 && uMax < wG[0]-5 && vMax < hG[0]-5))
					is_oob = true;
			}
			else if (USE_PAL == 2){
				if(!(pal_check_in_range_g(uMax, vMax, 5, 0)))
					is_oob = true;
			}
				
			if(is_oob){
				if(debugPrint) 
					printf("OOB uMax  %f %f - %f %f!\n",u,v, uMax, vMax);
				lastTraceUV = Vec2f(-1,-1);
				lastTracePixelInterval=0;
				return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
			}	
			assert(dist>0);

		}
	} // if(std::isfinite(idepth_max))

	// set OOB if scale change too big.
	// 括号内条件是继续运行的条件: 最大深度小于0 || 最大深度变化不大[0.75 1.5](尺度变化不大)
	if(!(idepth_min<0 || (ptpMin[2]>0.75 && ptpMin[2]<1.5)))
	{
		if(debugPrint) 
			printf(" $ OOB SCALE %f %f %f!\n", uMax, vMax,  ptpMin[2]);
		lastTraceUV = Vec2f(-1,-1);
		lastTracePixelInterval=0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}




	// ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
	float dx = setting_trace_stepsize*(uMax-uMin);
	float dy = setting_trace_stepsize*(vMax-vMin);

	// a:极线在梯度方向上投影的平方
	// b:垂直极线的方向在梯度上投影的平方
	// 如果梯度和极线方向相同,是非常理想的情况,此时a比较大,b接近0,此时errInPxl接近最小值 -> 0.4
	// 反之,是不理想的情况,此时a接近0,b比较大, errInPxl接近最大值 -> +Inf 
	float a = (Vec2f(dx,dy).transpose() * gradH * Vec2f(dx,dy));
	float b = (Vec2f(dy,-dx).transpose() * gradH * Vec2f(dy,-dx));
	// 好坑啊,(a+b)/a 化简之后就是1/(cos(theta)^2) theta是极线和梯度的夹角,这个表达式和梯度,极线的大小无关,之和夹角有关
	float errorInPixel = 0.2f + 0.2f * (a+b) / a;

	// 误差比极线还长,没意义了
	if(errorInPixel*setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max))
	{
		if(debugPrint){

			printf(" $ NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);

			// hwjdebug -----------
			line(img_now, Point(uMax, vMax), Point(uMin, vMin), {0, 0, 255});
			imshow("img_now", img_now);
			waitKey();
			// hwjdebug ------------------
		}

		lastTraceUV = Vec2f(uMax+uMin, vMax+vMin)*0.5;
		lastTracePixelInterval=dist;
		return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
	}

	if(errorInPixel >10) 
		errorInPixel=10;



	// ============== do the discrete search ===================
	dx /= dist;
	dy /= dist;

	if(debugPrint)
		printf("\tmin(%.1f %.1f) -> max(%.1f %.1f)! ErrorInPixel %.1f!\n",
				uMin, vMin,
				uMax, vMax,
				errorInPixel
				);


	if(dist>maxPixSearch)
	{
		uMax = uMin + maxPixSearch*dx;
		vMax = vMin + maxPixSearch*dy;
		dist = maxPixSearch;
	}

	int numSteps = 1.9999f + dist / setting_trace_stepsize;
	Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2,2>();

	// 随机偏移一点极线的起始点
	float randShift = uMin*1000-floorf(uMin*1000);
	float ptx = uMin-randShift*dx;
	float pty = vMin-randShift*dy;

	// 把pattern进行旋转
	Vec2f rotatetPattern[MAX_RES_PER_POINT];
	for(int idx=0;idx<patternNum;idx++)
		rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

	if(!std::isfinite(dx) || !std::isfinite(dy))
	{
		//printf("COUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);
		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	float errors[100]; // 累计极线上每个搜索点的误差
	float bestU=0, bestV=0, bestEnergy=1e10;
	int bestIdx=-1;
	if(numSteps >= 100) 
		numSteps = 99;

	Vec3f d_pal;
	if(USE_PAL == 1){ // 1
		ptpMin = ptpMin / ptpMin.norm();
		ptpMax = ptpMax / ptpMax.norm();
		d_pal = (ptpMax - ptpMin) / numSteps;
	}


	// 沿着极线搜索，把误差保存到error中
	for(int i=0;i<numSteps;i++)
	{
		float energy=0;
		// 累计每个pattern的残差
		for(int idx=0;idx<patternNum;idx++)
		{
			float hitColor = getInterpolatedElement31(frame->dI,
										(float)(ptx+rotatetPattern[idx][0]),
										(float)(pty+rotatetPattern[idx][1]),
										wG[0]);

			if(!std::isfinite(hitColor)) 
				{energy+=1e5; continue;}
			float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw *residual*residual*(2-hw);
		}

		if(debugPrint)
			printf("\tpt[%d/%d](%.1f %.1f) energy = %f!\n", i, numSteps, ptx, pty, energy);

		errors[i] = energy;
		if(energy < bestEnergy){
			bestU = ptx; bestV = pty; bestEnergy = energy; bestIdx = i;
		}

		// PAL极线搜索点计算
		if(USE_PAL == 1){ // 0 1
			Vec2f pt_pal = pal_model_g->world2cam(ptpMin + d_pal*(i+1) );
			ptx = pt_pal[0];
			pty = pt_pal[1];
		}
		else{
			ptx+=dx;
			pty+=dy;
		}



	}
	
	// 根据best匹配点重新计算 dx dy，作为优化方向
	if(USE_PAL == 1){ // 1
		Vec2f best_pt_pal1 = pal_model_g->world2cam(ptpMin + d_pal*bestIdx);
		Vec2f best_pt_pal2 = pal_model_g->world2cam(ptpMin + d_pal*(bestIdx+1));
		Vec2f best_pt_diff = best_pt_pal2 - best_pt_pal1;
		best_pt_diff.normalize();
		dx = best_pt_diff[0];
		dy = best_pt_diff[1];
	}

	// find best score outside a +-2px radius.
	float secondBest=1e10;
	for(int i=0;i<numSteps;i++)
	{
		if((i < bestIdx-setting_minTraceTestRadius || i > bestIdx+setting_minTraceTestRadius) && errors[i] < secondBest)
			secondBest = errors[i];
	}
	float newQuality = secondBest / bestEnergy;
	if(newQuality < quality || numSteps > 10) 
		quality = newQuality;


	// ============== do GN optimization ===================
	float uBak=bestU, vBak=bestV, gnstepsize=1, stepBack=0;
	if(setting_trace_GNIterations>0) 
		bestEnergy = 1e5;
	int gnStepsGood=0, gnStepsBad=0;
	for(int it=0;it<setting_trace_GNIterations;it++)
	{
		float H = 1, b=0, energy=0;
		// 累计所有pattern的能量
		for(int idx=0;idx<patternNum;idx++)
		{
			Vec3f hitColor = getInterpolatedElement33(frame->dI,
					(float)(bestU+rotatetPattern[idx][0]),
					(float)(bestV+rotatetPattern[idx][1]),wG[0]);

			if(!std::isfinite((float)hitColor[0])) 
				{energy+=1e5; continue;}
			float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			// 极线方向和梯度方向的点积
			float dResdDist = dx*hitColor[1] + dy*hitColor[2];
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			H += hw*dResdDist*dResdDist;
			b += hw*residual*dResdDist;
			energy += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
		}


		if(energy > bestEnergy)
		{
			gnStepsBad++;

			// do a smaller step from old point.
			stepBack*=0.5;
			bestU = uBak + stepBack*dx;
			bestV = vBak + stepBack*dy;
			if(debugPrint)
				printf("\t   - GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
						it, energy, H, b, stepBack,
						uBak, vBak, bestU, bestV);
		}
		else
		{
			gnStepsGood++;

			float step = -gnstepsize*b/H;
			if(step < -0.5) step = -0.5;
			else if(step > 0.5) step=0.5;

			if(!std::isfinite(step)) step=0;

			uBak=bestU;
			vBak=bestV;
			stepBack=step;

			bestU += step*dx;
			bestV += step*dy;
			bestEnergy = energy;

			if(debugPrint)
				printf("\t   - GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
						it, energy, H, b, step,
						uBak, vBak, bestU, bestV);
		}

		if(fabsf(stepBack) < setting_trace_GNThreshold) 
			break;
	}

	// ============== detect energy-based outlier. ===================
	if(!(bestEnergy < energyTH*setting_trace_extraSlackOnTH)){
		if(debugPrint)
			printf("\tOUTLIER!\n");

		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		if(lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		else
			return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	// ============== set new interval ===================

	// pin的三角化在像素坐标系下进行,PAL需要在相机坐标系下进行
	// 对于Pr变量对于pin,pr的xy是旋转后的参考帧像素坐标,z是反深度； 对于PAL, pr的前两位是相机坐标,z是反深度
	if(USE_PAL == 1){ // 0 1
		// 把bestUV转换到相机坐标系,和Pr变量统一坐标系
		Vec3f bestUVmin_pal = pal_model_g->cam2world(bestU-errorInPixel*dx, bestV-errorInPixel*dy);
		Vec3f bestUVmax_pal = pal_model_g->cam2world(bestU+errorInPixel*dx, bestV+errorInPixel*dy);

		if(dx*dx>dy*dy)
		{
			idepth_min = (pr[2]*(bestUVmin_pal[0]) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestUVmin_pal[0]));
			idepth_max = (pr[2]*(bestUVmax_pal[0]) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestUVmax_pal[0]));
		}
		else
		{
			idepth_min = (pr[2]*(bestUVmin_pal[1]) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestUVmin_pal[1]));	
			idepth_max = (pr[2]*(bestUVmax_pal[1]) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestUVmax_pal[1]));
		}
	}
	else{
		// bestUV和Pr都是像素坐标系
		if(dx*dx>dy*dy)
		{
			idepth_min = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU-errorInPixel*dx));
			idepth_max = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2]*(bestU+errorInPixel*dx));
		}
		else
		{
			// ( d0 * v1 - v0 ) / ( t2 - t3*v1 )
			idepth_min = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV-errorInPixel*dy));	
			idepth_max = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2]*(bestV+errorInPixel*dy));
		}
	}

	if(idepth_min > idepth_max) 
		std::swap<float>(idepth_min, idepth_max);

	// hwjdebug 显示极线和最佳匹配----------
	if(debugPrint){

		// hwjdebug -----------
		line(img_now, Point(uMax, vMax), Point(uMin, vMin), {0, 255, 0});
		circle(img_now, Point(bestU, bestV), 5, {0, 255, 0});
		line(img_now, Point(bestU-grad[0], bestV-grad[1]), Point(bestU+grad[0], bestV+grad[1]), {255, 0, 0});
		float dot = grad[0]*dx + grad[1]*dy;
		float det = grad[0]*dy - grad[1]*dx;
		float angle = abs(atan2(det, dot))/3.14*180;
		if(angle > 90)
			angle = 180 - angle;		

		static int veryGoodPoints = 0;
		static int allPoints = 0;
		allPoints ++;
		if(angle < 45){
			veryGoodPoints ++;
		}

		printf("\t   * new idmin: %.2f idmax: %.2f angle(%.2f) verygood(%d/%d)\n", idepth_min, idepth_max, angle, veryGoodPoints, allPoints);
		imshow("img_now", img_now);
		moveWindow("img_now", 50+wG[0]+50, 50);
		waitKey();
	}
	// -------------------

	//如果ideph 是无限大，或者小于0,那么设置为外点
	if(!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max<0))
	{
		//printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);
		lastTracePixelInterval=0;
		lastTraceUV = Vec2f(-1,-1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	lastTracePixelInterval=2*errorInPixel;
	// if(ENH_PAL) //极线搜索误差根据pal模型增大一些
	// 	lastTracePixelInterval *= 2;

	lastTraceUV = Vec2f(bestU, bestV);
	return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
}


float ImmaturePoint::getdPixdd(
		CalibHessian *  HCalib,
		ImmaturePointTemporaryResidual* tmpRes,
		float idepth)
{
	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);
	const Vec3f &PRE_tTll = precalc->PRE_tTll;
	float drescale, u=0, v=0, new_idepth;
	float Ku, Kv;
	Vec3f KliP;

	projectPoint(this->u,this->v, idepth, 0, 0,HCalib,
			precalc->PRE_RTll,PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth);

	float dxdd = (PRE_tTll[0]-PRE_tTll[2]*u)*HCalib->fxl();
	float dydd = (PRE_tTll[1]-PRE_tTll[2]*v)*HCalib->fyl();
	return drescale*sqrtf(dxdd*dxdd + dydd*dydd);
}


float ImmaturePoint::calcResidual(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float idepth)
{
	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

	float energyLeft=0;
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
	Vec2f affLL = precalc->PRE_aff_mode;

	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;
		if(!projectPoint(this->u+patternP[idx][0], this->v+patternP[idx][1], idepth, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{return 1e10;}

		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
		if(!std::isfinite((float)hitColor[0])) {return 1e10;}
		//if(benchmarkSpecialOption==5) hitColor = (getInterpolatedElement13BiCub(tmpRes->target->I, Ku, Kv, wG[0]));

		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);
	}

	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
	}
	return energyLeft;
}


// 计算未熟点和某帧的残差，帧的信息存储再tmpRes中
// tmpRes 最终残差输出的地方
// outlierTHSlack 外点阈值
double ImmaturePoint::linearizeResidual(
		CalibHessian *  HCalib, const float outlierTHSlack,
		ImmaturePointTemporaryResidual* tmpRes,
		float &Hdd, float &bd,
		float idepth)
{
	// 如果残差是OOB类型，那么就直接返回
	if(tmpRes->state_state == ResState::OOB)
	{ 
		tmpRes->state_NewState = ResState::OOB; 
		// hwjdebugsb
		// printf("\t\t FAIL(OOB)\n");
		return tmpRes->state_energy; 
	}

	// 获取未熟点所在帧和当前要计算误差的帧的precalc值
	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

	// check OOB due to scale angle change.

	float energyLeft=0;
	Eigen::Vector3f* dIl = tmpRes->target->dI; // 目标帧的图像
	const Mat33f &PRE_RTll = precalc->PRE_RTll;
	const Vec3f &PRE_tTll = precalc->PRE_tTll;
	//const float * const Il = tmpRes->target->I;

	Vec2f affLL = precalc->PRE_aff_mode;

	// hwjdebug -----------------
	// Mat fra = IOWrap::getOCVImg(dIl, wG[0], hG[0]);
	// circle(fra, Point(Ku, Kv), 3, 255);
	// imshow("ref", ref);
	// imshow("frame", fra);
	// moveWindow("ref", 50, 50);
	// moveWindow("frame", 50+wG[0]+50, 50);
	// waitKey(0);
	// printf("\t\t");
	// --------------------



	// 对于每个pattern点，投影到关键帧，计算亮度误差
	for(int idx=0;idx<patternNum;idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		if(!projectPoint(this->u,this->v, idepth, dx, dy,HCalib,PRE_RTll,PRE_tTll,  // 输入
			drescale, u, v, Ku, Kv, KliP, new_idepth))	// 输出
			{
			tmpRes->state_NewState = ResState::OOB; 
			// hwjdebugsb
			// printf("\t\tFAIL(Project OUT)\n");
			return tmpRes->state_energy;
		}



		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

		if(!std::isfinite((float)hitColor[0])) 
		{
			tmpRes->state_NewState = ResState::OOB;
			// hwjdebugsb
			// printf("\t\tFAIL(INF COLOR)\n");
			return tmpRes->state_energy;
		}

		// 计算亮度误差，累计能量
		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);
		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx]*weights[idx]*hw *residual*residual*(2-hw);

		// depth derivatives.
		// 对点的深度求导
		float d_idepth;
		if(USE_PAL == 1){ // 0 1

			float gx = hitColor[1];
			float gy = hitColor[2];
			d_idepth = derive_idepth_pal(PRE_tTll, u, v, idepth, gx, gy, drescale);
			// printf("\t\t - p = (%.2f, %.2f, %.2f), dd = %.2f  ", u, v, idepth, d_idepth);
		}
		else{
			float dxInterp = hitColor[1]*HCalib->fxl();
			float dyInterp = hitColor[2]*HCalib->fyl();
			d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);
		}

		hw *= weights[idx]*weights[idx];

		Hdd += (hw*d_idepth)*d_idepth;
		bd += (hw*residual)*d_idepth;
		// printf("hw(%.2f) dd(%.2f) H(%.2f) b(%.2f)\n",hw, d_idepth, (hw*d_idepth)*d_idepth, (hw*residual)*d_idepth);
	}


	// 误差还行，记为内点，否则记为外点
	if(energyLeft > energyTH*outlierTHSlack)
	{
		energyLeft = energyTH*outlierTHSlack;
		tmpRes->state_NewState = ResState::OUTLIER;
		// hwjdebugsb
		// printf("\t\t FAIL(BAD RES) E = %.2f, Eth = %.2f\n", energyLeft, energyTH*outlierTHSlack);
	}
	else
	{
		tmpRes->state_NewState = ResState::IN;
		// hwjdebugsb
		// printf("\t\t GOOD!\n");
	}

	// hwjdebug-----------------
	// using namespace cv;
	// Mat ref = IOWrap::getOCVImg(host->dI, wG[0], hG[0]);
	// circle(ref, Point(this->u, this->v), 3, 255);
	// -------------------------

	tmpRes->state_NewEnergy = energyLeft;
	return energyLeft;
}



}
