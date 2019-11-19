#include <line/line_init.h>

#include <FullSystem/CoarseInitializer.h>
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"

#include "util/pal_model.h"
#include "util/pal_interface.h"

int init_method_g = 0;

namespace dso{


Vec3f CoarseInitializer::calcResAndGS_v2(
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

	// hwjdebug -------------------------------------
	// cout << R << endl; 
	// cout << RKi << endl;
	// cout << t.transpose() << endl;
	cv::Mat resImg = cv::Mat::zeros(hl, wl, CV_8UC1);
	cv::Mat projImg = cv::Mat::zeros(hl, wl, CV_8UC1);
	// --------------------------------------------

	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];

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
			Vec3f pt;
			Vec2f p2_pal;
			float u ;
			float v ;
			float Ku; 
			float Kv; 
			float new_idepth;

			if(USE_PAL == 1){ // 0 1 2 
				pt = R * pal_model_g->cam2world(point->u+dx, point->v+dy, lvl) + t*point->idepth_new;
				p2_pal = pal_model_g->world2cam(pt, lvl);
				u = pt[0]/pt[2];	// u v 归一化平面坐标
				v = pt[1]/pt[2];	
				Ku = p2_pal[0];	// Ku Kv 像素平面坐标
				Kv = p2_pal[1];
				new_idepth = point->idepth_new/pt[2];
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
			}
			else{
				pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;
				u = pt[0] / pt[2];
				v = pt[1] / pt[2];
				Ku = fxl * u + cxl;
				Kv = fyl * v + cyl;
				new_idepth = point->idepth_new/pt[2];
				// 如果新的点出界或者深度异常，那么点设置为无效
				if(USE_PAL == 0){
					if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0)){
						isGood = false;
						break;
					}
				}
				else{
					if(!(pal_check_in_range_g(Ku, Kv, 2, lvl) && new_idepth > 0))
					{
						isGood = false;
						break;
					}

				}
			}

			// 亚像素梯度和亮度
			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);  // 新帧的 [亮度 dx dy]
			float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl); // 参考帧亮度
			
			// hwjdebug==========================
			if( dx == 0 && dy == 0){
				projImg.at<uchar>(Kv, Ku) = (uchar)hitColor[0];
			}
			// ================================

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

			float dxdd ; 
			float dydd ;
			float maxstep ;
			if(USE_PAL == 1){ // 0 1
				// if(ENH_PAL){ // 初始化部分,直接法GN优化部分增加pal视场的权重
				// 	hw *= pal_get_weight(Vec2f(Ku, Kv), lvl) * pal_get_weight(Vec2f(point->u+dx, point->v+dy), lvl);
				// }

				dxdd = (t[0]-t[2]*u); // \rho_2 / \rho1 * (tx - u'_2 * tz)
				dydd = (t[1]-t[2]*v); // \rho_2 / \rho1 * (ty - v'_2 * tz)
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
				Vec2f duvdd =  duv2dxyz * dxyzdd;
				auto dd_scalar = dr2duv2.transpose() * duvdd; 
				dd[idx] = dd_scalar[0];
				maxstep = 1.0f/ duvdd.norm();
			}
			else{
				dxdd = (t[0]-t[2]*u)/pt[2]; // \rho_2 / \rho1 * (tx - u'_2 * tz)
				dydd = (t[1]-t[2]*v)/pt[2]; // \rho_2 / \rho1 * (ty - v'_2 * tz)
				float dxInterp = hw*hitColor[1]*fxl;
				float dyInterp = hw*hitColor[2]*fyl;
				dp0[idx] = new_idepth*dxInterp;						// dE/d(SE1) = gx*fx/Z
				dp1[idx] = new_idepth*dyInterp;						// dE/d(SE2) = gy*fy/Z
				dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp);	// dE/d(SE3)
				dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;		// ...d(SE4)
				dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp;			// ...d(SE5)
				dp5[idx] = -v*dxInterp + u*dyInterp;				// ...d(SE6)
				dd[idx] = dxInterp * dxdd  + dyInterp * dydd;		// dE/d(depth)
				maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();
			}

			dp6[idx] = - hw*r2new_aff[0] * rlR;					// dE/d(e^(aj))
			dp7[idx] = - hw*1;									// dE/d(bj)	
			r[idx] = hw*residual;								// res

			if(maxstep < point->maxstep) 
				point->maxstep = maxstep;

			// immediately compute dp*dd1' and dd*dd' in JbBuffer1.
			// pattern的所有点都要累加进来计算梯度
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

		// 如果pattern中有一些点挂了，或者总的误差过大
		if(!isGood || energy > point->outlierTH*20)
		{
			// 单纯累计误差，不继续算了
			E.updateSingle((float)(point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}

		// hwjdebug ----------------------
		resImg.at<uchar>(point->v, point->u) = energy/10;

		// ------------------------------------

		// 计算完成pattern 8个点的误差，累计能量E，累计8个梯度到acc9中
		// add into energy.
		E.updateSingle(energy);
		point->isGood_new = true;
		point->energy_new[0] = energy;

		// update Hessian matrix.
		// pattern每个小点的J都用SSE的方式累计到H中
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

	// 计算alphaEnergy = alphaW * norm(t) * npts 
	// AlphaEnergy用来表征当前估计的质量（位移越大，点数越多，质量越高）
	// 	等于位移乘以点数
	float alphaEnergy = alphaW*(refToNew.translation().squaredNorm() * npts);

	// compute alpha opt.
	float alphaOpt;
	// 如果位移足够多
	if(alphaEnergy > alphaK*npts)
	{
		// hwjdebug---------------------------------------
		if(printDebug) 
			printf("\t  SNAPPING (lvl %d)  |t|=%.4f npts=%d alphaE=%.2f (th=%.2f)\n", lvl, 
				refToNew.translation().squaredNorm(), npts, alphaEnergy, alphaK*npts);
		// --------------------------------------------

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

		point->lastHessian_new = JbBuffer_new[i][9];	// dd*dd

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

		JbBuffer_new[i][9] = 1/(1 + JbBuffer_new[i][9]); // 防止dd=0? 
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2], 	// d(SE3) * dd
				(float)JbBuffer_new[i][3],(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],	
				(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],							// d(AB) * dd
				(float)JbBuffer_new[i][8],														// r * dd
				(float)JbBuffer_new[i][9]);														// (weight) dd * dd
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

	// // hwjdebug ===================
	// Mat resImg_toshow, projImg_show;
	// resize(resImg, resImg_toshow, cv::Size(w[0], h[0]));
	// resize(projImg, projImg_show, cv::Size(w[0], h[0]));
	// imshow("Res", resImg_toshow);
	// imshow("Proj", projImg_show);
	// moveWindow("Res", 50, 50);
	// moveWindow("Proj", 50+w[0]+20, 50);
	// waitKey();
	// // -------------------------

	return Vec3f(E.A, alphaEnergy ,E.num);
}

}