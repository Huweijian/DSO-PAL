#pragma once
#include <opencv2/opencv.hpp>

namespace cv{
Ptr<LineSegmentDetector> createLineSegmentDetectorMy(
	int _refine = LineSegmentDetectorModes::LSD_REFINE_STD, double _scale = 0.8,
	double _sigma_scale = 0.6, double _quant = 2.0, double _ang_th = 22.5,
	double _log_eps = 0, double _density_th = 0.7, int _n_bins = 1024);
}

