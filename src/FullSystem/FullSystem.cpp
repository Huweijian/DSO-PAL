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
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"

#include "util/ImageAndExposure.h"
#include "util/pal_interface.h"
#include <cmath>

namespace dso
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;



FullSystem::FullSystem()
{

	int retstat =0;
	if(setting_logStuff)
	{

		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);


		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847);



	selectionMap = new float[wG[0]*hG[0]];

	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	coarseTracker = new CoarseTracker(wG[0], hG[0]);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100);

	currentMinActDist=2;
	initialized=false;


	ef = new EnergyFunctional();
	ef->red = &this->treadReduce;

	isLost=false;
	initFailed=false;


	needNewKFAfter = -1;

	linearizeOperation=true;
	runMapping=true;
	mappingThread = boost::thread(&FullSystem::mappingLoop, this);
	lastRefStopID=0;



	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	if(setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}

	delete[] selectionMap;

	for(FrameShell* s : allFrameHistory)
		delete s;
	for(FrameHessian* fh : unmappedTrackedFrames)
		delete fh;

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{

}

void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


	// invert.
	for(int i=1;i<255;i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for(int s=1;s<255;s++)
		{
			if(BInv[s] <= i && BInv[s+1] >= i)
			{
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}



void FullSystem::printResult(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);

	for(FrameShell* s : allFrameHistory)
	{
		if(!s->poseValid) continue;

		if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;

		myfile << s->timestamp <<
			" " << s->camToWorld.translation().transpose()<<
			" " << s->camToWorld.so3().unit_quaternion().x()<<
			" " << s->camToWorld.so3().unit_quaternion().y()<<
			" " << s->camToWorld.so3().unit_quaternion().z()<<
			" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
	}
	myfile.close();
}

// 在tracker->lastRef的基础上跟踪fh
// 返回值：0 跟踪误差   1-3 光流
Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
{
	assert(allFrameHistory.size() > 0);
	// set pose initialization.

	// 输出图像
    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);

	// 获取最近一帧关键帧
	FrameHessian* lastF = coarseTracker->lastRef;

	// 亮度
	AffLight aff_last_2_l = AffLight(0,0);
	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;

	if(allFrameHistory.size() == 2){
		// 这个不存在的吧？不可能只有2帧就开始track了
		assert(0);
		for(unsigned int i=0;i<lastF_2_fh_tries.size();i++) 
			lastF_2_fh_tries.push_back(SE3());
	}
	else
	{
		// 拿到最新帧之前两帧之间的位移和亮度，并本地化
		FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
		FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];
		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	// lock on global pose consistency!
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
			lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
			aff_last_2_l = slast->aff_g2l;
		}

		// 假设一些可能的位移
		SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.
		// get last delta-movement.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
		lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
		lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
		lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

		// just try a TON of different initializations (all rotations). In the end,
		// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
		// also, if tracking rails here we loose, so we really, really want to avoid that.
		// 尝试不同的位姿
		// 这个循环只会执行一次
		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
		{
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		}
		
		// 如果有的位姿无效
		if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}

	Vec3 flowVecs = Vec3(100,100,100);
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);

	// as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	// I'll keep track of the so-far best achieved residual for each level in achievedRes.
	// If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

	Vec5 achievedRes = Vec5::Constant(NAN);

	bool haveOneGood = false;
	int tryIterations=0;
	// 尝试上面的全部位姿
	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{
		// 光度ab沿用之前的值作为初始值
		AffLight aff_g2l_this = aff_last_2_l;
		// 位姿进行不同的尝试
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

		// 在从最高层向下逐层tarck
		bool trackingIsGood = coarseTracker->trackNewestCoarse(
				fh, lastF_2_fh_this, aff_g2l_this,
				pyrLevelsUsed-1,
				achievedRes);	// in each level has to be at least as good as the last try.
		tryIterations++;

		if(i != 0)
		{
			printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
					i,
					i, pyrLevelsUsed-1,
					aff_g2l_this.a,aff_g2l_this.b,
					achievedRes[0],
					achievedRes[1],
					achievedRes[2],
					achievedRes[3],
					achievedRes[4],
					coarseTracker->lastResiduals[0],
					coarseTracker->lastResiduals[1],
					coarseTracker->lastResiduals[2],
					coarseTracker->lastResiduals[3],
					coarseTracker->lastResiduals[4]);
		}

		// do we have a new winner?
		// 如果这个位姿跟踪还不错(误差小)
		if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			//printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
			// 保存一些结果（光度ab，位姿，flow)	
			flowVecs = coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			haveOneGood = true;
		}

		// take over achieved res (always).
		if(haveOneGood)
		{
			for(int i=0;i<5;i++)
			{
				// 如果第i个achieveRes无效或者比最新的残差大，那么就保存一下这个残差分量
				// 相当于保存可以达到的最小的残差
				// take over if achievedRes is either bigger or NAN.
				if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	
					achievedRes[i] = coarseTracker->lastResiduals[i];
			}
		}

		// 如果比上次的好，就可以退出了，不用继续尝试了
        if(haveOneGood && achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
            break;

	} // 尝试各种可能位姿

	// 如果所有尝试都挂了，那么全部赋值为0，装死
	if(!haveOneGood)
	{
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
		flowVecs = Vec3(0,0,0);
		aff_g2l = aff_last_2_l;
		lastF_2_fh = lastF_2_fh_tries[0];
	}

	// 保存上次可以达到的误差
	lastCoarseRMSE = achievedRes;

	// 更新shell
	// no lock required, as fh is not used anywhere yet.
	fh->shell->camToTrackingRef = lastF_2_fh.inverse();
	fh->shell->trackingRef = lastF->shell;
	fh->shell->aff_g2l = aff_g2l;
	fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;


	if(coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];

    // if(!setting_debugout_runquiet)
    //     printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);
	
	if(setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
						<< fh->shell->id << " "
						<< fh->shell->timestamp << " "
						<< fh->ab_exposure << " "
						<< fh->shell->camToWorld.log().transpose() << " "
						<< aff_g2l.a << " "
						<< aff_g2l.b << " "
						<< achievedRes[0] << " "
						<< tryIterations << "\n";
	}

	// hwjdebug ----------------
    if(!setting_debugout_runquiet)
		std::cout << std::setprecision(4)	//之前是16
						// << "Coarse Tracker: "
						<< "跟踪帧 " << fh->shell->id << " "
						<< fh->shell->timestamp << " "
						<< fh->ab_exposure << "  pose: "
						<< fh->shell->camToWorld.log().transpose() << " "
						// << "aff: " 
						// << aff_g2l.a << " "
						// << aff_g2l.b << " "
						<< "res:"
						<< achievedRes[0] << " it:"
						<< tryIterations << "\n";

	// 返回误差和光流
	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

// 新帧和所有关键帧的未成熟点进行极线匹配，
void FullSystem::traceNewCoarse(FrameHessian* fh)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();
	
	if(USE_PAL == 0 || USE_PAL == 2){
		K(0,0) = Hcalib.fxl();
		K(1,1) = Hcalib.fyl();
		K(0,2) = Hcalib.cxl();
		K(1,2) = Hcalib.cyl();
	}

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		// 当前帧到历史帧的位姿
		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		Vec3f Kt = K * hostToNew.translation().cast<float>();
		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		// 枚举所有关键帧的所有未熟点
		for(ImmaturePoint* ph : host->immaturePoints)
		{
			// 利用fh的信息提高ph的精度
			ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );

			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}

	printf(" $ TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
			trace_total,
			trace_good, 100*trace_good/(float)trace_total,
			trace_skip, 100*trace_skip/(float)trace_total,
			trace_badcondition, 100*trace_badcondition/(float)trace_total,
			trace_oob, 100*trace_oob/(float)trace_total,
			trace_out, 100*trace_out/(float)trace_total,
			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);

	return ;
}



// 激活未熟点
void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian*>* optimized, 
		std::vector<ImmaturePoint*>* toOptimize,
		int min, int max, Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
	for(int k=min;k<max;k++)
	{
		//尝试对第k个点进行激活
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	}
	delete[] tr;
}


// 多线程激活未熟点
// 1. 根据距离图选择要激活的未熟点
// 2. 尝试多线程激活(投影到所有关键帧，计算残差，如果残差还行的话 再进一步优化反深度，最终成熟)
// 3. 删除激活失败的未熟点
void FullSystem::activatePointsMT()
{

	// 点数不足或者过多，就适当减小MinActDist
	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf(" - SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);


	// 取出最新的帧
	FrameHessian* newestHs = frameHessians.back();

	// make dist map.
	// 建立距离图（到成熟点的距离）
	coarseDistanceMap->makeK(&Hcalib);
	coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

	std::vector<ImmaturePoint*> toOptimize; 
	toOptimize.reserve(20000);

	// hwjdebug ------------------
	int immature_deleted = 0, immature_notReady = 0, immature_needMarg = 0, immature_out = 0, immature_all = 0;

	// ------------------

	//枚举老帧上的未熟点， 如果这些未熟点还行(条件见下面 canActivate )，并且投影到老帧上距离其他成熟点较远，那么加入toOptimize队列，准备优化
	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		if(host == newestHs) 
			continue;

		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]); // DSO可太坏了,K和Ki分别是不同lvl的
		Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

		for(unsigned int i=0;i<host->immaturePoints.size();i+=1)
		{
			immature_all ++;
			ImmaturePoint* ph = host->immaturePoints[i];
			ph->idxInImmaturePoints = i;

			// delete points that have never been traced successfully, or that are outlier on the last trace.
			if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
			{
				// remove point.
				immature_deleted ++ ;
				delete ph;
				host->immaturePoints[i]=0;
				continue;
			}

			//激活未熟点的条件：上次跟踪还行 & 上次跟踪极线误差较小(和梯度方向在极线方向的投影有关) & 上次跟踪质量还行（最佳匹配/二佳匹配）& 上次跟踪大于零
			// can activate only if this is true.
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD
					|| ph->lastTraceStatus == IPS_SKIPPED
					|| ph->lastTraceStatus == IPS_BADCONDITION
					|| ph->lastTraceStatus == IPS_OOB )
							&& ph->lastTracePixelInterval < 8
							&& ph->quality > setting_minTraceQuality
							&& (ph->idepth_max+ph->idepth_min) > 0;


			// if I cannot activate the point, skip it. Maybe also delete it.
			if(!canActivate)
			{
				// if point will be out afterwards, delete it instead.
				if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
				{
					immature_needMarg ++ ;
					delete ph;
					host->immaturePoints[i]=0;
					continue ;
				}

				immature_notReady ++ ;
				// hwjdebug ----------------------
					// printf("\t IM Pt not Ready (%d): status(%d), lastTraceErr(%.2f), quanlity(%.2f), minid(%.2f)\n", 
					// 	i, ph->lastTraceStatus, ph->lastTracePixelInterval, ph->quality, 0.5*(ph->idepth_max+ph->idepth_min));
				// -------------
				continue;
			}


			// see if we need to activate point due to distance map.
			// 未成熟点投影到某一老关键帧上
			Vec3f ptp;
			int u, v;
			
			if(USE_PAL == 1){ // 0 1
				
				ptp = KRKi * pal_model_g->cam2world(ph->u, ph->v, 0) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
				Vec2f ptp_pal2D = pal_model_g->world2cam(ptp, 1);
				u = ptp_pal2D[0];
				v = ptp_pal2D[1];
			}
			else{

				ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
				u = ptp[0] / ptp[2] + 0.5f;
				v = ptp[1] / ptp[2] + 0.5f;
			}

			bool inRange = false;
			if(USE_PAL == 1 || USE_PAL == 2){
				if(pal_check_in_range_g(u, v, 1, 1))
					inRange = true;
			}
			else{
				if((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
					inRange = true;
			}

			if(inRange)
			{

				// 距离附近成熟点的距离
				float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

				// type是点在被选择时候的顺序，分别为1 2 4 
				// currentMinActDist 是和点的数目相关的变量，初始等于2, 点数越少，变量越小
				// dist 是当前点uv投影到老帧之后 到最近特征点的距离，如果这个距离比较大，
				if(dist >= currentMinActDist * ph->my_type)
				{
					// 扩展距离图
					coarseDistanceMap->addIntoDistFinal(u,v);
					toOptimize.push_back(ph);
				}
				
				// hwjdebug ----------------------
					// printf("\t IMP IN (%d): [lvl 1] (%.2f, %.2f) -[%.2f %.2f %.2f]> (%d, %d)\n", i, ph->u, ph->v, ptp[0], ptp[1], ptp[2], u, v);
				// -------------
			}
			else
			{
				// hwjdebug ----------------------
					// printf("\t IMP OUT (%d): [lvl 1] (%.2f, %.2f) -> (%d, %d)\n", i, ph->u, ph->v, u, v);
				// -------------
				immature_out ++ ;
				delete ph;
				host->immaturePoints[i]=0;
			}
		}
	}

	// hwjdebug -------------
    if(!setting_debugout_runquiet)
		printf(" - ACTIVATE: (to active %d/%d, del %d, notReady %d, marg %d, out %d)\n",
				(int)toOptimize.size(), immature_all, immature_deleted, immature_notReady, immature_needMarg, immature_out);
	// -------------------

	// 开始多线程激活未熟点
	std::vector<PointHessian*> optimized; 
	optimized.resize(toOptimize.size());


	// hwjdebug-------
	// multiThreading = false;
	int optiFail = 0, optiBadPoint = 0, optiGood = 0;
	// --------

	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);
	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);


	for(unsigned k=0;k<toOptimize.size();k++)
	{
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		// 如果点已经激活成功，ph被成功初始化
		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			// 你已经长大了，从未熟点队列移出，转移到ph队列中
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);
			// 插入ef
			ef->insertPoint(newpoint);
			for(PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);
			assert(newpoint->efPoint != 0);
			// 彻底摆脱不熟的自己
			delete ph;
			optiGood ++;
		}
		else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
		{
			delete ph;
			ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
			optiFail ++;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
			optiBadPoint ++;
		}
	}

	// hwjdebug -------------
    if(!setting_debugout_runquiet)
		printf(" - ACTIVE RESULT: good(%d), fail(%d), bad(%d)\n", optiGood, optiFail, optiBadPoint);
	// ---------------


	// 排除帧中的所有处理过的未熟点
	for(FrameHessian* host : frameHessians)
	{
		for(int i=0;i<(int)host->immaturePoints.size();i++)
		{
			if(host->immaturePoints[i]==0)
			{
				host->immaturePoints[i] = host->immaturePoints.back();
				host->immaturePoints.pop_back();
				i--;
			}
		}
	}
}






void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

	// 从队列中倒数帧(根本不会执行！！)
	for(int i=((int)frameHessians.size())-1; i>=0 && i >= ((int)frameHessians.size());i--){
		printf("here i = %d\n", i);
		if(!frameHessians[i]->flaggedForMarginalization) 
			fhsToKeepPoints.push_back(frameHessians[i]);
	}
	// hwj add 认为这个没卵用，恒等于0
	assert(fhsToKeepPoints.size() == 0);

	// 把需要mag的帧存入变量
	for(int i=0; i< (int)frameHessians.size();i++)
		if(frameHessians[i]->flaggedForMarginalization) 
			fhsToMargPoints.push_back(frameHessians[i]);


	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;
	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		for(unsigned int i=0;i<host->pointHessians.size();i++)
		{
			PointHessian* ph = host->pointHessians[i];
			if(ph==0) 
				continue;

			// 如果点的深度是负数，或者点没有有效的残差，标记为drop
			if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
			{
				// 算作out点，标记为drop
				host->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				host->pointHessians[i]=0;
				flag_nores++;
			}
			// 否则如果出界了，或者宿主被mag掉了
			else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
			{
				flag_oob++;
				// 点依然是内点，重置状态
				if(ph->isInlierNew())
				{
					flag_in++;
					int ngoodRes=0;
					// 枚举pfr
					for(PointFrameResidual* r : ph->residuals)
					{
						// 重置OOB的状态	
						r->resetOOB();
						// 重新线性化
						r->linearize(&Hcalib);
						// 后面的有点晕了
						r->efResidual->isLinearized = false;
						r->applyRes(true);
						if(r->efResidual->isActive())
						{
							r->efResidual->fixLinearizationF(ef);
							ngoodRes++;
						}
					}
                    if(ph->idepth_hessian > setting_minIdepthH_marg)
					{
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
					}
					else
					{
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
					}


				}
				// 真的要out了
				else
				{
					host->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;


					//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
				}

				host->pointHessians[i]=0;
			}
		}

		// 把标记为0的点删掉
		for(int i=0;i<(int)host->pointHessians.size();i++)
		{
			if(host->pointHessians[i]==0)
			{
				host->pointHessians[i] = host->pointHessians.back();
				host->pointHessians.pop_back();
				i--;
			}
		}
	}

}

// slam 入口，图像从这里输入
void FullSystem::addActiveFrame( ImageAndExposure* image, int id )
{
    if(isLost) 
		return;
	// cv::waitKey();
	boost::unique_lock<boost::mutex> lock(trackMutex);

	// 初始化图像类型
	// =========================== add into allFrameHistory =========================
	FrameHessian* fh = new FrameHessian();
	FrameShell* shell = new FrameShell();
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
	shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id;
	shell->marker_id = image->marker_id;
	fh->shell = shell;
	allFrameHistory.push_back(shell);

	// 计算金字塔和梯度
	// =========================== make Images / derivatives etc. =========================
	fh->ab_exposure = image->exposure_time;
    fh->makeImages(image->image, &Hcalib);

	// hwjtest 测试直接显示pal点云
	// {

	// 	// printf("%lu %lu %lu %lu\n", fh->pointHessians.size(), fh->immaturePoints.size(), fh->pointHessiansMarginalized.size(), fh->pointHessiansOut.size());
	// 	if(id > 5)
	// 		return;

	// 	for(int u=0; u<wG[0]; u+=10){
	// 		for(int v=0; v<hG[0]; v+=10){
	// 			if(!pal_check_in_range_g(u, v, 3))
	// 				continue;

	// 			int val = fh->dI[v*wG[0]+u][0];
	// 			auto pt = pal_model_g->cam2world(u, v);
	// 			ImmaturePoint *ip = new ImmaturePoint(u, v, fh, 1, &Hcalib);
	// 			ip->idepth_max = ip->idepth_min = 1.0;
	// 			PointHessian *p = new PointHessian(ip, &Hcalib);
	// 			p->idepth_scaled = pt[2];
	// 			p->maxRelBaseline = 10.0;
	// 			p->idepth_hessian = 1000.0;
	// 			fh->pointHessians.push_back(p);
	// 		}
	// 	}

	// 	Vec6 pose;
	// 	pose << id*10, 0, 0, 0, 0, 0;
	// 	fh->PRE_camToWorld = SE3::exp(pose);
	// 	fh->PRE_worldToCam = fh->PRE_camToWorld.inverse();
	// 	fh->frameID = id;

	// 	frameHessians.push_back(fh);
	// 	for(IOWrap::Output3DWrapper* ow : outputWrapper)
	// 	{
	// 		ow->publishKeyframes(frameHessians, false, &Hcalib);
	// 	}
	// 	frameHessians.clear();
	// 	return ;
	// }

	// 初始化
	if(!initialized)
	{
		// use initializer!
		// 第一帧初始化
		if(coarseInitializer->frameID<0)	// first frame set. fh is kept by coarseInitializer.
		{
			// 初始化梯度较大的点，并计算KNN
			coarseInitializer->setFirst(&Hcalib, fh);
		}
		// 后续帧初始化，尝试track
		else if(coarseInitializer->trackFrame(fh, outputWrapper))	// if SNAPPED
		{

			initializeFromInitializer(fh); //初始第一帧加入fh
			lock.unlock();
			deliverTrackedFrame(fh, true); //当前帧也加入fh
		}
		// track失败，丢弃帧
		else
		{
			// if still initializing
			fh->shell->poseValid = false;
			delete fh;
		}
		return;
	}
	else	// do front-end operation.
	{
		// =========================== SWAP tracking reference?. =========================
		if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
		{
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			CoarseTracker* tmp = coarseTracker; 
			coarseTracker = coarseTracker_forNewKF; 
			coarseTracker_forNewKF = tmp;
		}

		// 基于最后关键帧的点云，用直接发进行跟踪
		Vec4 tres = trackNewCoarse(fh);
		if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            printf("Initial Tracking failed: LOST!\n");
			isLost=true;
            return;
        }


		bool needToMakeKF = false;
		// 如果强制每秒创建一个关键帧
		if(setting_keyframesPerSecond > 0)
		{
			needToMakeKF = allFrameHistory.size()== 1 ||
					(fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
		}
		else
		{
			Vec2 refToFh=AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
					coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

			// BRIGHTNESS CHECK
			// 如果位姿或者光度变化足够大 || 当前误差大于第一次误差的2倍 就创建新的关键帧
			if(allFrameHistory.size()== 1){
				needToMakeKF = true;
				printf("\n创建新关键帧(allFrameHistory = 1)");
			}
			else if(setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +
					setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1 ){
				float t = setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]);
				float r = setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]);
				float rt = setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]);
				float aff = setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) ;
				
				needToMakeKF = true;

				if(needToMakeKF)
					printf("\n创建新关键帧(位姿或光度变化过大) t=%.2f r=%.2f rt=%.2f aff=%.2f", t, r, rt, aff);

			}
			else if(2*coarseTracker->firstCoarseRMSE < tres[0]){
				needToMakeKF = true;
				printf("\n创建新关键帧(当前误差(%.2f) > 第一次跟踪误差的2倍(%.2f))", tres[0], coarseTracker->firstCoarseRMSE);
			}
			// hwjdebug ----------
			static int kfNum = 0;
			if(needToMakeKF){
				printf(" [No %d keyframe]\n", kfNum++);
			}
			// ----------------

		}

        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);


		lock.unlock();
		deliverTrackedFrame(fh, needToMakeKF);
		return;
	}
}


void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{

	if(linearizeOperation)
	{
		// 判断要不要单步运行
		if(goStepByStep && lastRefStopID != coarseTracker->refFrameID)
		{
			MinimalImageF3 img(wG[0], hG[0], fh->dI);
			IOWrap::displayImage("frameToTrack", &img);
			while(true)
			{
				char k=IOWrap::waitKey(0);
				if(k==' ') break;
				handleKey( k );
			}
			lastRefStopID = coarseTracker->refFrameID;
		}
		else 
			handleKey( IOWrap::waitKey(1) );

		if(needKF) 
			makeKeyFrame(fh);
		else 
			makeNonKeyFrame(fh);
	}
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		unmappedTrackedFrames.push_back(fh);
		if(needKF) needNewKFAfter=fh->shell->trackingRef->id;
		trackedFrameSignal.notify_all();

		while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);
		}

		lock.unlock();
	}
}

void FullSystem::mappingLoop()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while(runMapping)
	{
		while(unmappedTrackedFrames.size()==0)
		{
			trackedFrameSignal.wait(lock);
			if(!runMapping) return;
		}

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();


		// guaranteed to make a KF for the very first two tracked frames.
		if(allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();
			makeKeyFrame(fh);
			lock.lock();
			mappedFrameSignal.notify_all();
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;


		if(unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
		{
			lock.unlock();
			makeNonKeyFrame(fh);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)
			{
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
				}
				delete fh;
			}

		}
		else
		{
			if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
			{
				lock.unlock();
				makeKeyFrame(fh);
				needToKetchupMapping=false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				makeNonKeyFrame(fh);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	mappingThread.join();

}

void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh);
	
	// hwjdebug --------------

	// using namespace cv;
	// FrameHessian* key_fh = frameHessians.front();
	// Mat color = IOWrap::getOCVImg(key_fh->dI, wG[0], hG[0]);
	// Mat imp_img = Mat::zeros(hG[0], wG[0], CV_8UC1); 
	// for(auto &imp : key_fh->immaturePoints){
	// 	imp_img.at<uchar>(imp->v, imp->u) = (imp->idepth_max + imp->idepth_min) / 2 * 100;
	// }
	// imshow("imp_img", imp_img);
	// moveWindow("imp_img", 50, 50);

	// Mat ph_img = Mat::zeros(hG[0], wG[0], CV_8UC1);
	// for(auto &ph : key_fh->pointHessians){
	// 	ph_img.at<uchar>(ph->v, ph->u) = ph->idepth * 100;
	// }
	// imshow("ph_img", ph_img);
	// moveWindow("ph_img", 50 + wG[0] + 50, 50);
	// printf(" $ %ul im points, %ul ph points\n", key_fh->immaturePoints.size(), key_fh->pointHessians.size());

	// waitKey();

	// -----------
	delete fh;
}

// fh 当前帧
void FullSystem::makeKeyFrame( FrameHessian* fh)
{

	// needs to be set by mapping thread
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		//更新当前帧的全局位姿
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		// TODO: 看不懂
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh);

	boost::unique_lock<boost::mutex> lock(mapMutex);

	// =========================== Flag Frames to be Marginalized. =========================
	// 标记要Mag掉的帧
	flagFramesForMarginalization(fh);


	// =========================== add New Frame to Hessian Struct. =========================
	// 添加初始化的最后一帧到fh
	fh->idx = frameHessians.size();
	frameHessians.push_back(fh);
	fh->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(fh->shell);
	ef->insertFrame(fh, &Hcalib);

	setPrecalcValues();

	// =========================== add new residuals for old points =========================
	int numFwdResAdde=0;
	// 枚举所有老帧
	for(FrameHessian* fh1 : frameHessians)		// go through all active frames
	{
		if(fh1 == fh) 
			continue;
		// 枚举老帧的所有点, 新建老点和新帧的残差，并加入到ef
		for(PointHessian* ph : fh1->pointHessians)
		{
			PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
			r->setState(ResState::IN);
			ph->residuals.push_back(r);
			ef->insertResidual(r);
			ph->lastResiduals[1] = ph->lastResiduals[0];
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
			numFwdResAdde+=1;
		}
	}


	// =========================== Activate Points (& flag for marginalization). =========================
	// 尝试把未熟点激活
	activatePointsMT();
	ef->makeIDX() ;

	// =========================== OPTIMIZE ALL =========================

	// 滑动窗口优化
	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
	float rmse = optimize(setting_maxOptIterations);



	// =========================== Figure Out if INITIALIZATION FAILED =========================
	// 判断初始化是否成功
	if(allKeyFramesHistory.size() <= 4)
	{
		if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	}

    if(isLost) 
		return;

	// =========================== REMOVE OUTLIER =========================
	removeOutliers();

	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		// 把新的calib用于tracker
		coarseTracker_forNewKF->makeK(&Hcalib);
		// 新的关键帧队列用于tracker
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);

        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
	}

	debugPlot("post Optimize");

	// =========================== (Activate-)Marginalize Points =========================

	// 标记外点
	flagPointsForRemoval();
	// 再次移除外点
	ef->dropPointsF();
	// 获取零空间
	getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);

	// TODO: 边缘化点，待看。。
	// 边缘化
	ef->marginalizePointsF();

	// =========================== add new Immature points & new residuals =========================
	// 增加新的未熟点
	makeNewTraces(fh, 0);

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
        ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
    }

	// =========================== Marginalize Frames =========================

	for(unsigned int i=0;i<frameHessians.size();i++)
		if(frameHessians[i]->flaggedForMarginalization)
		{
			// TODO: 边缘化,待看
			marginalizeFrame(frameHessians[i]); 
			i=0;
		}

	printLogLine();
}


void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	// add firstframe.
	// 添加第一帧到ef中
	FrameHessian* firstFrame = coarseInitializer->firstFrame;
	firstFrame->idx = frameHessians.size();
	frameHessians.push_back(firstFrame);
	firstFrame->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(firstFrame->shell);
	ef->insertFrame(firstFrame, &Hcalib);
	setPrecalcValues(); //提前设置一些值(旋转乘以K等)

	// 预留总点数的20%的空间
	firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);

	// 累加金字塔顶层的反深度，计算归一化稀疏
	float sumID=1e-5, numID=1e-5;
	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		sumID += coarseInitializer->points[0][i].iR;
		numID++;
	}
	float rescaleFactor = 1 / (sumID / numID);

	// randomly sub-select the points I need.
	// 随机选一些点初始化
	float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];
    if(!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
                (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );

	// 枚举每个点
	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		// 运气不好 拜拜
		if(rand()/(float)RAND_MAX > keepPercentage){
			continue;
		}

		// 运气还行，初始化点海森

		// 尝试初始化未熟点
		Pnt* point = coarseInitializer->points[0]+i;
		ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f, point->v+0.5f, firstFrame, point->my_type, &Hcalib);
		if(!std::isfinite(pt->energyTH)) 
			{ delete pt; continue; }


		// 尝试初始化点海森
		pt->idepth_max=pt->idepth_min=1;
		PointHessian* ph = new PointHessian(pt, &Hcalib);
		// 蛤？ 直接诶删掉未熟点？
		delete pt;
		if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

		// 设置点海森，激活
		ph->setIdepthScaled(point->iR*rescaleFactor);
		ph->setIdepthZero(ph->idepth);
		ph->hasDepthPrior=true;
		ph->setPointStatus(PointHessian::ACTIVE);

		// 点海森添加到帧海森中
		firstFrame->pointHessians.push_back(ph);
		ef->insertPoint(ph);
	}

	// 位移缩放
	SE3 firstToNew = coarseInitializer->thisToNext;
	firstToNew.translation() /= rescaleFactor;


	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		firstFrame->shell->camToWorld = SE3();
		firstFrame->shell->aff_g2l = AffLight(0,0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef=0;
		firstFrame->shell->camToTrackingRef = SE3();

		newFrame->shell->camToWorld = firstToNew.inverse();
		newFrame->shell->aff_g2l = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();

	}

	initialized=true;
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

// 给最新关键帧初始化一些未熟点
void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
{
	pixelSelector->allowFast = true;
	//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

	// 预留一些空间
	newFrame->pointHessians.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);


	for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++){
		for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
		{

			// 排除外部的点
			if(USE_PAL == 1 || USE_PAL == 2){
				if(!pal_check_in_range_g(x, y, patternPadding+1)){
					continue;
				}
			}

			int i = x+y*wG[0];
			if(selectionMap[i]==0) 
				continue;

			ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);
			if(!std::isfinite(impt->energyTH)) 
				delete impt;
			else 
				newFrame->immaturePoints.push_back(impt);

		}
	}

	//printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

}


// 任意两个相邻帧之间设置预计算（和相机参数相关）
void FullSystem::setPrecalcValues()
{

	// fh 枚举帧
	for(FrameHessian* fh : frameHessians)
	{
		fh->targetPrecalc.resize(frameHessians.size());
		// 再次枚举帧
		for(unsigned int i=0;i<frameHessians.size();i++)
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
	}

	// TODO: 看不懂
	ef->setDeltaF(&Hcalib);
}


void FullSystem::printLogLine()
{
	if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n\n",
                allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                ef->resInA,
                ef->resInL,
                ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                allKeyFramesHistory.back()->aff_g2l.a,
                allKeyFramesHistory.back()->aff_g2l.b,
                frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                (int)frameHessians.size());


	if(!setting_logStuff) return;

	if(numsLog != 0)
	{
		(*numsLog) << allKeyFramesHistory.back()->id << " "  <<
				statistics_lastFineTrackRMSE << " "  <<
				(int)statistics_numCreatedPoints << " "  <<
				(int)statistics_numActivatedPoints << " "  <<
				(int)statistics_numDroppedPoints << " "  <<
				(int)statistics_lastNumOptIts << " "  <<
				ef->resInA << " "  <<
				ef->resInL << " "  <<
				ef->resInM << " "  <<
				statistics_numMargResFwd << " "  <<
				statistics_numMargResBwd << " "  <<
				statistics_numForceDroppedResFwd << " "  <<
				statistics_numForceDroppedResBwd << " "  <<
				frameHessians.back()->aff_g2l().a << " "  <<
				frameHessians.back()->aff_g2l().b << " "  <<
				frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
				(int)frameHessians.size() << " "  << "\n";
		numsLog->flush();
	}

}



void FullSystem::printEigenValLine()
{
	if(!setting_logStuff) return;
	if(ef->lastHS.rows() < 12) return;


	MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	int n = Hp.cols()/8;
	assert(Hp.cols()%8==0);

	// sub-select
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(i*8,0,6,n*8);
		Hp.block(i*6,0,6,n*8) = tmp6;

		MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
		Ha.block(i*2,0,2,n*8) = tmp2;
	}
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(0,i*8,n*8,6);
		Hp.block(0,i*6,n*8,6) = tmp6;

		MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
		Ha.block(0,i*2,n*8,2) = tmp2;
	}

	VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
	VecX diagonal = ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data()+eigenP.size());
	std::sort(eigenA.data(), eigenA.data()+eigenA.size());

	int nz = std::max(100,setting_maxFrames*10);

	if(eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if(eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if(eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if(DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if(variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
		(*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
	(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
	for(unsigned int i=0;i<nsp.size();i++)
		(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
	if(!setting_logStuff) return;


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for(FrameShell* s : allFrameHistory)
	{
		(*lg) << s->id
			<< " " << s->marginalizedAt
			<< " " << s->statistics_goodResOnThis
			<< " " << s->statistics_outlierResOnThis
			<< " " << s->movedByOpt;



		(*lg) << "\n";
	}





	lg->close();
	delete lg;

}


void FullSystem::printEvalLine()
{
	return;
}

bool FullSystem::loadTrajectory(std::string trajectoryFile)
{
	using namespace std;
	printf("start loading trajectory...\n");
	ifstream fin(trajectoryFile);

	if(!fin.good()){
		printf(" ! Bad trajcetory file path in ");
		cout << trajectoryFile << endl;
		return false;
	}
	while(!fin.eof()){
		static int cnt;
		Vec3f p;
		fin >> p(0) >> p(1) >> p(2);
		traj.push_back(p);	
	}
	printf(" ! load trajectory success, %lu points\n", traj.size());

	return false;
}




}
