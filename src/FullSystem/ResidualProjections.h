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


#pragma once

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "util/settings.h"
#include "util/pal_interface.h"

namespace dso
{

// 输入的uv是相机坐标系!!!!
EIGEN_STRONG_INLINE float derive_idepth(
		const Vec3f &t, const float &u, const float &v,
		const int &dx, const int &dy, const float &dxInterp,
		const float &dyInterp, const float &drescale)
{
	return (dxInterp*drescale * (t[0]-t[2]*u)
			+ dyInterp*drescale * (t[1]-t[2]*v))*SCALE_IDEPTH;
}

// 输入的uv是相机坐标系!!!!
EIGEN_STRONG_INLINE float derive_idepth_pal(
		const Vec3f &t, 
		float u, float v, float idepth,
		float gx,float gy, 
		float drescale)
{
	Vec3f pt = Vec3f(u, v, 1)/idepth;
	Eigen::Matrix<float, 2, 3> duvdxyz;
	Eigen::Matrix<float, 2, 6> duvdSE;
	pal_model_g->jacobian_xyz2uv(pt, duvdSE, duvdxyz);

	// printf("(%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f) dres = %.2f", u, v, idepth, pt[0], pt[1], pt[2], drescale);
	
	Vec3f dxyzdd = Vec3f((t[0]-t[2]*u), (t[1]-t[2]*v), 0);
	Eigen::Matrix<float, 1, 1> dd_scalar= Vec2f(gx, gy).transpose() * duvdxyz * dxyzdd; 	

	// using namespace std;
	// cout << endl;
	// cout << "d1 = " << Vec2f(gx, gy).transpose() << endl;
	// cout << "d2 = " << duvdxyz << endl;
	// cout << "d3 = " << dxyzdd.transpose() << endl;

	return dd_scalar[0] * drescale;
}


// 传入像素坐标系的点和反深度 和 SE3，判断再投影回来是否出界
EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt, const float &v_pt, const float &idepth,
		const Mat33f &KRKi, const Vec3f &Kt,
		// -----------------------------------
		float &Ku, float &Kv)
{
	if(USE_PAL){

	Vec3f ptp = KRKi * pal_model_g->cam2world(u_pt, v_pt) + Kt*idepth;
	Vec2f ptp_2d = pal_model_g->world2cam(ptp);
	Ku = ptp_2d[0];
	Kv = ptp_2d[1];
	return  pal_check_in_range_g(Ku, Kv, 2, 0);
	}

	else{

	Vec3f ptp = KRKi * Vec3f(u_pt,v_pt, 1) + Kt*idepth;
	Ku = ptp[0] / ptp[2];
	Kv = ptp[1] / ptp[2];
	return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G;
	}
}


// 把点投影到零一帧，并判断是否在图像内
EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt, const float &v_pt, const float &idepth,
		const int &dx, const int &dy,
		CalibHessian* const &HCalib,
		const Mat33f &R, const Vec3f &t,
		// ---------以下为返回值-----------
		float &drescale, float &u, float &v, // z变化比例系数；归一化坐标系的值；
		float &Ku, float &Kv, Vec3f &KliP, float &new_idepth)	// 像素坐标系的值；原始归一化坐标系的值；
{
	if(USE_PAL){

		KliP = pal_model_g->cam2world(u_pt+dx, v_pt+dy) / idepth;	
		Vec3f P2 = R * KliP + t;
		new_idepth = 1.0 / P2[2];
		drescale = new_idepth / idepth;
		
		// TODO: uv 可能有问题(PAL的归一化坐标系没意义)
		u = P2[0] * drescale;
		v = P2[1] * drescale;
		Vec2f KP2 = pal_model_g->world2cam(P2);
		Ku = KP2[0];
		Kv = KP2[1];

		return pal_check_in_range_g(Ku, Kv, 2, 0);
	}
	else{
		// K*P
		KliP = Vec3f(
				(u_pt+dx-HCalib->cxl())*HCalib->fxli(),
				(v_pt+dy-HCalib->cyl())*HCalib->fyli(),
				1);

		// RKP + t/d
		Vec3f ptp = R * KliP + t*idepth;
		
		// ptp[2] = d2/d1
		// drescale = d1/d2 = id2/id1;
		drescale = 1.0f/ptp[2];
		new_idepth = idepth*drescale;

		if(!(drescale>0)) 
			return false;

		// 按照深度缩放后再投影到图像上
		// u v 是归一化坐标系的坐标
		u = ptp[0] * drescale;
		v = ptp[1] * drescale;
		Ku = u*HCalib->fxl() + HCalib->cxl();
		Kv = v*HCalib->fyl() + HCalib->cyl();

		// 返回投影后的点是否在图像内
		return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G;
	}
}


}

