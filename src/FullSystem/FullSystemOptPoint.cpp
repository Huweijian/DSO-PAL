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

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ImmaturePoint.h"
#include "math.h"

namespace dso
{


// 尝试优化未熟点point，优化成功返回ph，否则返回0
PointHessian* FullSystem::optimizeImmaturePoint(
		ImmaturePoint* point, int minObs,
		ImmaturePointTemporaryResidual* residuals)
{
	int nres = 0;
	// 枚举关键帧，初始化关键帧到当前未熟点的残差对象
	for(FrameHessian* fh : frameHessians)
	{
		if(fh != point->host)
		{
			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
			residuals[nres].state_NewState = ResState::OUTLIER;
			residuals[nres].state_state = ResState::IN;
			residuals[nres].target = fh;
			nres++;
		}
	}
	assert(nres == ((int)frameHessians.size())-1);

	// bool print = false;//rand()%50==0;
	bool print = true;

	float lastEnergy = 0;
	float lastHdd=0;
	float lastbd=0;
	float currentIdepth = (point->idepth_max+point->idepth_min)*0.5f;

	// 计算点投影到每一帧之后的误差和雅克比，并累计总能量
	for(int i=0;i<nres;i++)
	{
		lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i,lastHdd, lastbd, currentIdepth);
		// 更新残差状态
		residuals[i].state_state = residuals[i].state_NewState;
		residuals[i].state_energy = residuals[i].state_NewEnergy;
	}

	// 如果总能量炸了，或者Hdd不足(??? Hdd什么意思)，成熟失败 gg
	if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
	{
		if(print)
			printf("\tOptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
				nres, lastHdd, lastEnergy);
		return 0;
	}

	// 批准继续成熟
	if(print) 
		printf("\tPoint. %d residuals. Hdd=%.2f. Initial Energy: %.2f. Initial Id=%.2f\n" ,
			nres, lastHdd, lastEnergy, currentIdepth);

	// 优化深度
	// TODO: 这里优化有问题,误差下降不明显(需要进一步可视化+debug)
	float lambda = 0.1;
	for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++)
	{
		float H = lastHdd;
		H *= 1+lambda;
		// idepth前进步长
		float step = (1.0/H) * lastbd;
		// 应用步长
		float newIdepth = currentIdepth - step;

		float newHdd=0; float newbd=0; float newEnergy=0;
		for(int i=0;i<nres;i++)
			newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i,newHdd, newbd, newIdepth);

		if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
		{
			if(print) printf("\tOptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres,
					newHdd,
					lastEnergy);
			return 0;
		}

		if(print) printf("\t - %s %d (L %.2f) %s: %.4f -> %.4f (idepth %.4f)!\n",
				(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
				iteration,
				(lambda),
				"",
				lastEnergy, newEnergy, newIdepth);

		// 前进成功
		if(newEnergy < lastEnergy)
		{
			currentIdepth = newIdepth;
			lastHdd = newHdd;
			lastbd = newbd;
			lastEnergy = newEnergy;
			for(int i=0;i<nres;i++)
			{
				residuals[i].state_state = residuals[i].state_NewState;
				residuals[i].state_energy = residuals[i].state_NewEnergy;
			}

			lambda *= 0.5;
		}
		else
		{
			lambda *= 5;
		}

		if(fabsf(step) < 0.0001*currentIdepth)
			break;
	}

	// 重大bug反馈
	if(!std::isfinite(currentIdepth))
	{
		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

	// 统计成熟一个点产生了几个有效的残差
	int numGoodRes=0;
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) 
			numGoodRes++;

	// 有效残差数目不足,可惜了,gg
	if(numGoodRes < minObs)
	{
		if(print) 
			printf("\tOptPoint: OUTLIER!\n"); 
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

	// 你已经几乎是一个成熟的点了,可惜energyTH大了点,gg
	PointHessian* p = new PointHessian(point, &Hcalib);
	if(!std::isfinite(p->energyTH)){
		delete p; 
		return (PointHessian*)((long)(-1));
	}

	// 初始化一堆成熟点应有的样子
	p->lastResiduals[0].first = 0;
	p->lastResiduals[0].second = ResState::OOB;
	p->lastResiduals[1].first = 0;
	p->lastResiduals[1].second = ResState::OOB;
	p->setIdepthZero(currentIdepth);
	p->setIdepth(currentIdepth);
	p->setPointStatus(PointHessian::ACTIVE);

	// 初始化点帧残差
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN)
		{
			PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
			r->state_NewEnergy = r->state_energy = 0;
			r->state_NewState = ResState::OUTLIER;
			r->setState(ResState::IN);
			// 残差加入点的队列
			p->residuals.push_back(r);

			if(r->target == frameHessians.back())
			{
				p->lastResiduals[0].first = r;
				p->lastResiduals[0].second = ResState::IN;
			}
			else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2]))
			{
				p->lastResiduals[1].first = r;
				p->lastResiduals[1].second = ResState::IN;
			}
		}

	// 成熟了蛤
	if(print) 
		printf("\tpoint activated!\n");

	statistics_numActivatedPoints++;
	return p;
}



}
