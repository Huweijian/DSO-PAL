#include <line/lsd.h>

#include <opencv2/opencv.hpp>
#include <vector>


/////////////////////////////////////////////////////////////////////////////////////////
// Default LSD parameters
// SIGMA_SCALE 0.6    - Sigma for Gaussian filter is computed as sigma = sigma_scale/scale.
// QUANT       2.0    - Bound to the quantization error on the gradient norm.
// ANG_TH      22.5   - Gradient angle tolerance in degrees.
// LOG_EPS     0.0    - Detection threshold: -log10(NFA) > log_eps
// DENSITY_TH  0.7    - Minimal density of region points in rectangle.
// N_BINS      1024   - Number of bins in pseudo-ordering of gradient modulus.

#define M_3_2_PI    (3 * CV_PI) / 2   // 3/2 pi
#define M_2__PI     (2 * CV_PI)         // 2 pi

#ifndef M_LN10
#define M_LN10      2.30258509299404568402
#endif

#define NOTDEF      double(-1024.0) // Label for pixels with undefined gradient.

#define NOTUSED     0   // Label for pixels not used in yet.
#define USED        1   // Label for pixels already used in detection.

#define RELATIVE_ERROR_FACTOR 100.0

const double DEG_TO_RADS = CV_PI / 180;

#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

struct edge
{
    cv::Point p;
    bool taken;
};

/////////////////////////////////////////////////////////////////////////////////////////

inline double distSq(const double x1, const double y1,
                     const double x2, const double y2)
{
    return (x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1);
}

inline double dist(const double x1, const double y1,
                   const double x2, const double y2)
{
    return sqrt(distSq(x1, y1, x2, y2));
}

// Signed angle difference
inline double angle_diff_signed(const double& a, const double& b)
{
    double diff = a - b;
    while(diff <= -CV_PI) diff += M_2__PI;
    while(diff >   CV_PI) diff -= M_2__PI;
    return diff;
}

// Absolute value angle difference
inline double angle_diff(const double& a, const double& b)
{
    return std::fabs(angle_diff_signed(a, b));
}

// Compare doubles by relative error.
inline bool double_equal(const double& a, const double& b)
{
    // trivial case
    if(a == b) return true;

    double abs_diff = fabs(a - b);
    double aa = fabs(a);
    double bb = fabs(b);
    double abs_max = (aa > bb)? aa : bb;

    if(abs_max < DBL_MIN) abs_max = DBL_MIN;

    return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}

inline bool AsmallerB_XoverY(const edge& a, const edge& b)
{
    if (a.p.x == b.p.x) return a.p.y < b.p.y;
    else return a.p.x < b.p.x;
}

/**
 *   Computes the natural logarithm of the absolute value of
 *   the gamma function of x using Windschitl method.
 *   See http://www.rskey.org/gamma.htm
 */
inline double log_gamma_windschitl(const double& x)
{
    return 0.918938533204673 + (x-0.5)*log(x) - x
         + 0.5*x*log(x*sinh(1/x) + 1/(810.0*pow(x, 6.0)));
}

/**
 *   Computes the natural logarithm of the absolute value of
 *   the gamma function of x using the Lanczos approximation.
 *   See http://www.rskey.org/gamma.htm
 */
inline double log_gamma_lanczos(const double& x)
{
    static double q[7] = { 75122.6331530, 80916.6278952, 36308.2951477,
                         8687.24529705, 1168.92649479, 83.8676043424,
                         2.50662827511 };
    double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
    double b = 0;
    for(int n = 0; n < 7; ++n)
    {
        a -= log(x + double(n));
        b += q[n] * pow(x, double(n));
    }
    return a + log(b);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cv{
LineSegmentDetectorMy::LineSegmentDetectorMy(int _refine, double _scale, double _sigma_scale, double _quant,
        double _ang_th, double _log_eps, double _density_th, int _n_bins)
        : img_width(0), img_height(0), LOG_NT(0), w_needed(false), p_needed(false), n_needed(false),
          SCALE(_scale), doRefine(_refine), SIGMA_SCALE(_sigma_scale), QUANT(_quant),
          ANG_TH(_ang_th), LOG_EPS(_log_eps), DENSITY_TH(_density_th), N_BINS(_n_bins)
{
    CV_Assert(_scale > 0 && _sigma_scale > 0 && _quant >= 0 &&
              _ang_th > 0 && _ang_th < 180 && _density_th >= 0 && _density_th < 1 &&
              _n_bins > 0);
}

void LineSegmentDetectorMy::detect(InputArray _image, OutputArray _lines,
                OutputArray _width, OutputArray _prec, OutputArray _nfa)
{
    image = _image.getMat();
    CV_Assert(!image.empty() && image.type() == CV_8UC1);
    cvtColor(_image, image_line_3b_, COLOR_GRAY2BGR);
    image_reg_3b_ = image_line_3b_.clone(); 

    region_mask_ = Mat::zeros(image.size(), CV_8UC1);

    std::vector<Vec4f> lines;
    std::vector<double> w, p, n;
    w_needed = _width.needed();
    p_needed = _prec.needed();
    if (doRefine < LSD_REFINE_ADV)
        n_needed = false;
    else
        n_needed = _nfa.needed();

    flsd(lines, w, p, n);

    Mat(lines).copyTo(_lines);
    if(w_needed) Mat(w).copyTo(_width);
    if(p_needed) Mat(p).copyTo(_prec);
    if(n_needed) Mat(n).copyTo(_nfa);

    // Clear used structures
    ordered_points.clear();
}

void LineSegmentDetectorMy::flsd(std::vector<Vec4f>& lines,
    std::vector<double>& widths, std::vector<double>& precisions,
    std::vector<double>& nfas)
{
    // Angle tolerance
    const double prec = CV_PI * ANG_TH / 180;
    const double p = ANG_TH / 180;
    const double rho = QUANT / sin(prec);    // gradient magnitude threshold

    // hwjdebug output parameters
    {
        printf("ANG_TH = %.2f\n", ANG_TH);
    }

    if(SCALE != 1)
    {
        Mat gaussian_img;
        const double sigma = (SCALE < 1)?(SIGMA_SCALE / SCALE):(SIGMA_SCALE);
        const double sprec = 3;
        const unsigned int h =  (unsigned int)(ceil(sigma * sqrt(2 * sprec * log(10.0))));
        Size ksize(1 + 2 * h, 1 + 2 * h); // kernel size
        GaussianBlur(image, gaussian_img, ksize, sigma);
        // Scale image to needed size
        resize(gaussian_img, scaled_image, Size(), SCALE, SCALE, INTER_LINEAR_EXACT);
        ll_angle(rho, N_BINS);
    }
    else
    {
        scaled_image = image;
        ll_angle(rho, N_BINS);
    }

    LOG_NT = 5 * (log10(double(img_width)) + log10(double(img_height))) / 2 + log10(11.0);
    const size_t min_reg_size = size_t(-LOG_NT/log10(p)); // minimal number of points in region that can give a meaningful event

    // // Initialize region only when needed
    // Mat region = Mat::zeros(scaled_image.size(), CV_8UC1);
    used = Mat_<uchar>::zeros(scaled_image.size()); // zeros = NOTUSED
    std::vector<RegionPoint> reg;

    // Search for line segments
    for(size_t i = 0, points_size = ordered_points.size(); i < points_size; ++i)
    {
        const Point2i& point = ordered_points[i].p;
        if((used.at<uchar>(point) == NOTUSED) && (angles.at<double>(point) != NOTDEF))
        {
            double reg_angle;
            cur_refine_times_ = 0;
            region_grow(ordered_points[i].p, reg, reg_angle, prec);

            // Ignore small regions
            if(reg.size() < min_reg_size) { continue; }

            // Construct rectangular approximation for the region
            rect rec;
            region2rect(reg, reg_angle, prec, p, rec);


            // hwjdebug  显示原始区域
            {
                int R = (rand() % (int)(205 + 1)) + 50;
                int G = (rand() % (int)(205 + 1)) + 50;
                int B = (rand() % (int)(205 + 1)) + 50;
                Vec3b reg_color = Vec3b(B, G, R);
                for(auto &p : reg){
                    image_reg_3b_.at<Vec3b>(p.y, p.x) = reg_color;
                }           
                Mat show;
                resize(image_reg_3b_, show, Size(), 2, 2, INTER_NEAREST);
                imshow("region", show);
            }
            

            double log_nfa = -1;
            if(doRefine > LSD_REFINE_NONE)
            {
                // At least REFINE_STANDARD lvl.
                if(!refine(reg, reg_angle, prec, p, rec, DENSITY_TH)) { continue; }

                if(doRefine >= LSD_REFINE_ADV)
                {
                    // Compute NFA
                    log_nfa = rect_improve(rec);
                    if(log_nfa <= LOG_EPS) { continue; }
                }
            }

            // Add the offset
            rec.x1 += 0.5; rec.y1 += 0.5;
            rec.x2 += 0.5; rec.y2 += 0.5;

            // // hwjdebug =====================
            {
                int R, G, B;
                R = cur_refine_times_ % 10 * 10;  //密度不足,除了降低角度阈值之外又进行了半径缩小的次数
                G = cur_refine_times_ / 100 % 10 * 50;  // NFA 优化次数
                B = cur_refine_times_ / 1000 % 10 * 128;  // 密度不足,需要调整
                Scalar lineColor = Scalar(B, G, R);
                // convert black to red (perfect line)
                if(lineColor == Scalar(0, 0, 0)){
                    lineColor = Scalar(0, 0, 255);
                }
                line(image_line_3b_, Point(rec.x1, rec.y1), Point(rec.x2, rec.y2), lineColor);

                Mat image_show;
                resize(image_line_3b_, image_show, Size(), 2, 2, INTER_NEAREST);
                imshow("detected", image_show);
                waitKey();
            }
            // // =================================

            // scale the result values if a sub-sampling was performed
            if(SCALE != 1)
            {
                rec.x1 /= SCALE; rec.y1 /= SCALE;
                rec.x2 /= SCALE; rec.y2 /= SCALE;
                rec.width /= SCALE;
            }

            //Store the relevant data
            lines.push_back(Vec4f(float(rec.x1), float(rec.y1), float(rec.x2), float(rec.y2)));
            if(w_needed) widths.push_back(rec.width);
            if(p_needed) precisions.push_back(rec.p);
            if(n_needed && doRefine >= LSD_REFINE_ADV) nfas.push_back(log_nfa);
            line_refine_times_.push_back(cur_refine_times_);
        }
    }
}

void LineSegmentDetectorMy::ll_angle(const double& threshold,
                                   const unsigned int& n_bins)
{
    //Initialize data
    angles = Mat_<double>(scaled_image.size());
    modgrad = Mat_<double>(scaled_image.size());

    img_width = scaled_image.cols;
    img_height = scaled_image.rows;

    // Undefined the down and right boundaries
    angles.row(img_height - 1).setTo(NOTDEF);
    angles.col(img_width - 1).setTo(NOTDEF);

    // Computing gradient for remaining pixels
    double max_grad = -1;
    for(int y = 0; y < img_height - 1; ++y)
    {
        const uchar* scaled_image_row = scaled_image.ptr<uchar>(y);
        const uchar* next_scaled_image_row = scaled_image.ptr<uchar>(y+1);
        double* angles_row = angles.ptr<double>(y);
        double* modgrad_row = modgrad.ptr<double>(y);
        for(int x = 0; x < img_width-1; ++x)
        {
            int DA = next_scaled_image_row[x + 1] - scaled_image_row[x];
            int BC = scaled_image_row[x + 1] - next_scaled_image_row[x];
            int gx = DA + BC;    // gradient x component
            int gy = DA - BC;    // gradient y component
            double norm = std::sqrt((gx * gx + gy * gy) / 4.0); // gradient norm

            modgrad_row[x] = norm;    // store gradient
            angles_row[x] = fastAtan2(float(gx), float(-gy)) * DEG_TO_RADS; // gradient angle computation
            if (norm > max_grad)
            {
                max_grad = norm;
            }

        }
    }
    
    Mat valid_grad, modgrad1b;
    modgrad.convertTo(modgrad1b, CV_8UC1);

    // 计算可以考虑的阈值
    adaptiveThreshold(modgrad1b, valid_grad, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 31, 0);
    // cv::threshold(modgrad1b, valid_grad, threshold, 255, THRESH_BINARY);
    angles.setTo(NOTDEF, ~valid_grad);
    valid_grad_ = valid_grad;


    // hwjdebug ---------------------------
    // {
    //     imshow("thres", valid_grad);
    //     imshow("grad", modgrad1b);
    //     waitKey();
    // }
    {
		using namespace std;
		Mat angles_deg = angles / DEG_TO_RADS;

		// double ang_min, ang_max;
		// cv::minMaxIdx(angles_deg, &ang_min, &ang_max);
		// printf("ang min=%.2f max=%.2f\n", ang_min, ang_max);
		
		Mat v = (angles >= 0);
		Mat s(angles.size(), CV_8UC1, Scalar(200));
		Mat h;
		angles_deg.convertTo(h, CV_8UC1, 0.5);
		vector<Mat> hsv_vec{ h, s, v };

		Mat hsv, angle_rgb;
		cv::merge(hsv_vec, hsv);
		cvtColor(hsv, angle_rgb, COLOR_HSV2BGR);

		Mat grad1b;
		modgrad.convertTo(grad1b, CV_8UC1);

        angle_rgb_ = angle_rgb;
        angle_degd2_ = h;
		// imshow("angle", angle_rgb);
		// imshow("grad", grad1b);
		// waitKey();
        }

    // ----------------

    // Compute histogram of gradient values
    double bin_coef = (max_grad > 0) ? double(n_bins - 1) / max_grad : 0; // If all image is smooth, max_grad <= 0
    for(int y = 0; y < img_height - 1; ++y)
    {
        const double* modgrad_row = modgrad.ptr<double>(y);
        for(int x = 0; x < img_width - 1; ++x)
        {
            normPoint _point;
            int i = int(modgrad_row[x] * bin_coef);
            _point.p = Point(x, y);
            _point.norm = i;
            ordered_points.push_back(_point);
        }
    }

    // Sort
    std::sort(ordered_points.begin(), ordered_points.end(), compare_norm);
}

void LineSegmentDetectorMy::region_grow(const Point2i& s, std::vector<RegionPoint>& reg,
                                      double& reg_angle, double prec)
{
    reg.clear();

    // Point to this region
    RegionPoint seed;
    seed.x = s.x;
    seed.y = s.y;
    seed.used = &used.at<uchar>(s);
    reg_angle = angles.at<double>(s);
    seed.angle = reg_angle;
    seed.modgrad = modgrad.at<double>(s);
    reg.push_back(seed);

    float sumdx = float(std::cos(reg_angle));
    float sumdy = float(std::sin(reg_angle));
    *seed.used = USED;

    int forgive_count = 0;

    //Try neighboring regions
    for (size_t i = 0;i<reg.size();i++)
    {
        const RegionPoint& rpoint = reg[i];
        int xx_min = std::max(rpoint.x - 1, 0), xx_max = std::min(rpoint.x + 1, img_width - 1);
        int yy_min = std::max(rpoint.y - 1, 0), yy_max = std::min(rpoint.y + 1, img_height - 1);
        int curSize = reg.size();
        for(int yy = yy_min; yy <= yy_max; ++yy)
        {
            uchar* used_row = used.ptr<uchar>(yy);
            const double* angles_row = angles.ptr<double>(yy);
            const double* modgrad_row = modgrad.ptr<double>(yy);
            for(int xx = xx_min; xx <= xx_max; ++xx)
            {
                uchar& is_used = used_row[xx];
                if(is_used != USED &&
                   (isAligned(xx, yy, reg_angle, prec)))
                {
                    const double& angle = angles_row[xx];
                    // Add point
                    is_used = USED;
                    RegionPoint region_point;
                    region_point.x = xx;
                    region_point.y = yy;
                    region_point.used = &is_used;
                    region_point.modgrad = modgrad_row[xx];
                    region_point.angle = angle;
                    reg.push_back(region_point);

                    // Update region's angle
                    sumdx += cos(float(angle));
                    sumdy += sin(float(angle));
                    // reg_angle is used in the isAligned, so it needs to be updates?
                    reg_angle = fastAtan2(sumdy, sumdx) * DEG_TO_RADS;
                }
            }
        }
    }
}

void LineSegmentDetectorMy::region2rect(const std::vector<RegionPoint>& reg,
                                      const double reg_angle, const double prec, const double p, rect& rec) const
{
    double x = 0, y = 0, sum = 0;
    for(size_t i = 0; i < reg.size(); ++i)
    {
        const RegionPoint& pnt = reg[i];
        const double& weight = pnt.modgrad;
        x += double(pnt.x) * weight;
        y += double(pnt.y) * weight;
        sum += weight;
    }

    // Weighted sum must differ from 0
    CV_Assert(sum > 0);

    x /= sum;
    y /= sum;

    double theta = get_theta(reg, x, y, reg_angle, prec);

    // Find length and width
    double dx = cos(theta);
    double dy = sin(theta);
    double l_min = 0, l_max = 0, w_min = 0, w_max = 0;

    for(size_t i = 0; i < reg.size(); ++i)
    {
        double regdx = double(reg[i].x) - x;
        double regdy = double(reg[i].y) - y;

        double l = regdx * dx + regdy * dy;
        double w = -regdx * dy + regdy * dx;

        if(l > l_max) l_max = l;
        else if(l < l_min) l_min = l;
        if(w > w_max) w_max = w;
        else if(w < w_min) w_min = w;
    }

    // Store values
    rec.x1 = x + l_min * dx;
    rec.y1 = y + l_min * dy;
    rec.x2 = x + l_max * dx;
    rec.y2 = y + l_max * dy;
    rec.width = w_max - w_min;
    rec.x = x;
    rec.y = y;
    rec.theta = theta;
    rec.dx = dx;
    rec.dy = dy;
    rec.prec = prec;
    rec.p = p;

    // Min width of 1 pixel
    if(rec.width < 1.0) rec.width = 1.0;
}

double LineSegmentDetectorMy::get_theta(const std::vector<RegionPoint>& reg, const double& x,
                                      const double& y, const double& reg_angle, const double& prec) const
{
    double Ixx = 0.0;
    double Iyy = 0.0;
    double Ixy = 0.0;

    // Compute inertia matrix
    for(size_t i = 0; i < reg.size(); ++i)
    {
        const double& regx = reg[i].x;
        const double& regy = reg[i].y;
        const double& weight = reg[i].modgrad;
        double dx = regx - x;
        double dy = regy - y;
        Ixx += dy * dy * weight;
        Iyy += dx * dx * weight;
        Ixy -= dx * dy * weight;
    }

    // Check if inertia matrix is null
    CV_Assert(!(double_equal(Ixx, 0) && double_equal(Iyy, 0) && double_equal(Ixy, 0)));

    // Compute smallest eigenvalue
    double lambda = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0 * Ixy * Ixy));

    // Compute angle
    double theta = (fabs(Ixx)>fabs(Iyy))?
                    double(fastAtan2(float(lambda - Ixx), float(Ixy))):
                    double(fastAtan2(float(Ixy), float(lambda - Iyy))); // in degs
    theta *= DEG_TO_RADS;

    // Correct angle by 180 deg if necessary
    if(angle_diff(theta, reg_angle) > prec) { theta += CV_PI; }

    return theta;
}

bool LineSegmentDetectorMy::refine(std::vector<RegionPoint>& reg, double reg_angle,
                                 const double prec, double p, rect& rec, const double& density_th)
{
    double density = double(reg.size()) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density >= density_th) { return true; }
    cur_refine_times_+=1000;

    // Try to reduce angle tolerance
    double xc = double(reg[0].x);
    double yc = double(reg[0].y);
    const double& ang_c = reg[0].angle;
    double sum = 0, s_sum = 0;
    int n = 0;

    for (size_t i = 0; i < reg.size(); ++i)
    {
        *(reg[i].used) = NOTUSED;
        if (dist(xc, yc, reg[i].x, reg[i].y) < rec.width)
        {
            const double& angle = reg[i].angle;
            double ang_d = angle_diff_signed(angle, ang_c);
            sum += ang_d;
            s_sum += ang_d * ang_d;
            ++n;
        }
    }
    CV_Assert(n > 0);
    double mean_angle = sum / double(n);
    // 2 * standard deviation
    double tau = 2.0 * sqrt((s_sum - 2.0 * mean_angle * sum) / double(n) + mean_angle * mean_angle);

    // Try new region
    region_grow(Point(reg[0].x, reg[0].y), reg, reg_angle, tau);

    // 一个点也扩展不出来,直接gg
    if (reg.size() < 2) { return false; }

    region2rect(reg, reg_angle, prec, p, rec);
    density = double(reg.size()) / (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);

    if (density < density_th)
    {
        // 密度依然不足
        return reduce_region_radius(reg, reg_angle, prec, p, rec, density, density_th);
    }
    else
    {
        // 二次区域生长后密度足够大了
        return true;
    }
}

bool LineSegmentDetectorMy::reduce_region_radius(std::vector<RegionPoint>& reg, double reg_angle,
                const double prec, double p, rect& rec, double density, const double& density_th)
{
    // Compute region's radius
    double xc = double(reg[0].x);
    double yc = double(reg[0].y);
    double radSq1 = distSq(xc, yc, rec.x1, rec.y1);
    double radSq2 = distSq(xc, yc, rec.x2, rec.y2);
    double radSq = radSq1 > radSq2 ? radSq1 : radSq2;

    while(density < density_th)
    {
        cur_refine_times_++;
        radSq *= 0.75*0.75; // Reduce region's radius to 75% of its value
        // Remove points from the region and update 'used' map
        for (size_t i = 0; i < reg.size(); ++i)
        {
            if(distSq(xc, yc, double(reg[i].x), double(reg[i].y)) > radSq)
            {
                // Remove point from the region
                *(reg[i].used) = NOTUSED;
                std::swap(reg[i], reg[reg.size() - 1]);
                reg.pop_back();
                --i; // To avoid skipping one point
            }
        }

        if(reg.size() < 2) { return false; }

        // Re-compute rectangle
        region2rect(reg ,reg_angle, prec, p, rec);

        // Re-compute region points density
        density = double(reg.size()) /
                  (dist(rec.x1, rec.y1, rec.x2, rec.y2) * rec.width);
    }

    return true;
}

double LineSegmentDetectorMy::rect_improve(rect& rec) 
{
    double delta = 0.5;
    double delta_2 = delta / 2.0;

    double log_nfa = rect_nfa(rec);

    if(log_nfa > LOG_EPS) return log_nfa; // Good rectangle

    cur_refine_times_+=100;
    // Try to improve
    // Finer precision
    rect r = rect(rec); // Copy
    for(int n = 0; n < 5; ++n)
    {
        r.p /= 2;
        r.prec = r.p * CV_PI;
        double log_nfa_new = rect_nfa(r);
        if(log_nfa_new > log_nfa)
        {
            log_nfa = log_nfa_new;
            rec = rect(r);
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;

    cur_refine_times_+=100;
    // Try to reduce width
    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.width -= delta;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;
    cur_refine_times_+=100;
    // Try to reduce one side of rectangle
    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.x1 += -r.dy * delta_2;
            r.y1 +=  r.dx * delta_2;
            r.x2 += -r.dy * delta_2;
            r.y2 +=  r.dx * delta_2;
            r.width -= delta;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;
    cur_refine_times_+=100;
    // Try to reduce other side of rectangle
    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.x1 -= -r.dy * delta_2;
            r.y1 -=  r.dx * delta_2;
            r.x2 -= -r.dy * delta_2;
            r.y2 -=  r.dx * delta_2;
            r.width -= delta;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }
    if(log_nfa > LOG_EPS) return log_nfa;
    cur_refine_times_+=100;
    // Try finer precision
    r = rect(rec);
    for(unsigned int n = 0; n < 5; ++n)
    {
        if((r.width - delta) >= 0.5)
        {
            r.p /= 2;
            r.prec = r.p * CV_PI;
            double log_nfa_new = rect_nfa(r);
            if(log_nfa_new > log_nfa)
            {
                rec = rect(r);
                log_nfa = log_nfa_new;
            }
        }
    }

    return log_nfa;
}

double LineSegmentDetectorMy::rect_nfa(const rect& rec) const
{
    int total_pts = 0, alg_pts = 0;
    double half_width = rec.width / 2.0;
    double dyhw = rec.dy * half_width;
    double dxhw = rec.dx * half_width;

    edge ordered_x[4];
    edge* min_y = &ordered_x[0];
    edge* max_y = &ordered_x[0]; // Will be used for loop range

    ordered_x[0].p.x = int(rec.x1 - dyhw); ordered_x[0].p.y = int(rec.y1 + dxhw); ordered_x[0].taken = false;
    ordered_x[1].p.x = int(rec.x2 - dyhw); ordered_x[1].p.y = int(rec.y2 + dxhw); ordered_x[1].taken = false;
    ordered_x[2].p.x = int(rec.x2 + dyhw); ordered_x[2].p.y = int(rec.y2 - dxhw); ordered_x[2].taken = false;
    ordered_x[3].p.x = int(rec.x1 + dyhw); ordered_x[3].p.y = int(rec.y1 - dxhw); ordered_x[3].taken = false;

    std::sort(ordered_x, ordered_x + 4, AsmallerB_XoverY);

    // Find min y. And mark as taken. find max y.
    for(unsigned int i = 1; i < 4; ++i)
    {
        if(min_y->p.y > ordered_x[i].p.y) {min_y = &ordered_x[i]; }
        if(max_y->p.y < ordered_x[i].p.y) {max_y = &ordered_x[i]; }
    }
    min_y->taken = true;

    // Find leftmost untaken point;
    edge* leftmost = 0;
    for(unsigned int i = 0; i < 4; ++i)
    {
        if(!ordered_x[i].taken)
        {
            if(!leftmost) // if uninitialized
            {
                leftmost = &ordered_x[i];
            }
            else if (leftmost->p.x > ordered_x[i].p.x)
            {
                leftmost = &ordered_x[i];
            }
        }
    }
    CV_Assert(leftmost != NULL);
    leftmost->taken = true;

    // Find rightmost untaken point;
    edge* rightmost = 0;
    for(unsigned int i = 0; i < 4; ++i)
    {
        if(!ordered_x[i].taken)
        {
            if(!rightmost) // if uninitialized
            {
                rightmost = &ordered_x[i];
            }
            else if (rightmost->p.x < ordered_x[i].p.x)
            {
                rightmost = &ordered_x[i];
            }
        }
    }
    CV_Assert(rightmost != NULL);
    rightmost->taken = true;

    // Find last untaken point;
    edge* tailp = 0;
    for(unsigned int i = 0; i < 4; ++i)
    {
        if(!ordered_x[i].taken)
        {
            if(!tailp) // if uninitialized
            {
                tailp = &ordered_x[i];
            }
            else if (tailp->p.x > ordered_x[i].p.x)
            {
                tailp = &ordered_x[i];
            }
        }
    }
    CV_Assert(tailp != NULL);
    tailp->taken = true;

    double flstep = (min_y->p.y != leftmost->p.y) ?
                    (min_y->p.x - leftmost->p.x) / (min_y->p.y - leftmost->p.y) : 0; //first left step
    double slstep = (leftmost->p.y != tailp->p.x) ?
                    (leftmost->p.x - tailp->p.x) / (leftmost->p.y - tailp->p.x) : 0; //second left step

    double frstep = (min_y->p.y != rightmost->p.y) ?
                    (min_y->p.x - rightmost->p.x) / (min_y->p.y - rightmost->p.y) : 0; //first right step
    double srstep = (rightmost->p.y != tailp->p.x) ?
                    (rightmost->p.x - tailp->p.x) / (rightmost->p.y - tailp->p.x) : 0; //second right step

    double lstep = flstep, rstep = frstep;

    double left_x = min_y->p.x, right_x = min_y->p.x;

    // Loop around all points in the region and count those that are aligned.
    int min_iter = min_y->p.y;
    int max_iter = max_y->p.y;
    for(int y = min_iter; y <= max_iter; ++y)
    {
        if (y < 0 || y >= img_height) continue;

        for(int x = int(left_x); x <= int(right_x); ++x)
        {
            if (x < 0 || x >= img_width) continue;

            ++total_pts;
            if(isAligned(x, y, rec.theta, rec.prec))
            {
                ++alg_pts;
            }
        }

        if(y >= leftmost->p.y) { lstep = slstep; }
        if(y >= rightmost->p.y) { rstep = srstep; }

        left_x += lstep;
        right_x += rstep;
    }

    return nfa(total_pts, alg_pts, rec.p);
}

double LineSegmentDetectorMy::nfa(const int& n, const int& k, const double& p) const
{
    // Trivial cases
    if(n == 0 || k == 0) { return -LOG_NT; }
    if(n == k) { return -LOG_NT - double(n) * log10(p); }

    double p_term = p / (1 - p);

    double log1term = (double(n) + 1) - log_gamma(double(k) + 1)
                - log_gamma(double(n-k) + 1)
                + double(k) * log(p) + double(n-k) * log(1.0 - p);
    double term = exp(log1term);

    if(double_equal(term, 0))
    {
        if(k > n * p) return -log1term / M_LN10 - LOG_NT;
        else return -LOG_NT;
    }

    // Compute more terms if needed
    double bin_tail = term;
    double tolerance = 0.1; // an error of 10% in the result is accepted
    for(int i = k + 1; i <= n; ++i)
    {
        double bin_term = double(n - i + 1) / double(i);
        double mult_term = bin_term * p_term;
        term *= mult_term;
        bin_tail += term;
        if(bin_term < 1)
        {
            double err = term * ((1 - pow(mult_term, double(n-i+1))) / (1 - mult_term) - 1);
            if(err < tolerance * fabs(-log10(bin_tail) - LOG_NT) * bin_tail) break;
        }

    }
    return -log10(bin_tail) - LOG_NT;
}

inline bool LineSegmentDetectorMy::isAligned(int x, int y, const double& theta, const double& prec) const
{
    // 图像边界
    if(x < 0 || y < 0 || x >= angles.cols || y >= angles.rows) { return false; }
   
    // 是否有角度 
    const double& a = angles.at<double>(y, x);
    if(a == NOTDEF) { return false; }

    // It is assumed that 'theta' and 'a' are in the range [-pi,pi]
    // 角度差,缩放到[-pi, pi]
    double n_theta = theta - a;
    if(n_theta < 0) { n_theta = -n_theta; }
    if(n_theta > M_3_2_PI)
    {
        n_theta -= M_2__PI;
        if(n_theta < 0) n_theta = -n_theta;
    }

    return n_theta <= prec;
}

void LineSegmentDetectorMy::drawSegments(InputOutputArray _image, InputArray lines)
{
    CV_Assert(!_image.empty() && (_image.channels() == 1 || _image.channels() == 3));

    if (_image.channels() == 1)
    {
        cvtColor(_image, _image, COLOR_GRAY2BGR);
    }

    Mat _lines = lines.getMat();
    const int N = _lines.checkVector(4);

    CV_Assert(_lines.depth() == CV_32F || _lines.depth() == CV_32S);


    for (int i = 0; i < N; ++i)
    {
        // Draw segments
        // int R = (rand() % (int)(255 + 1));
        // int G = (rand() % (int)(255 + 1));
        // int B = (rand() % (int)(255 + 1));

        int R, G, B;
        R = line_refine_times_[i]%10 * 10; //密度不足,除了降低角度阈值之外又进行了半径缩小的次数
        G = line_refine_times_[i]/100 % 10 * 50;//NFA 优化次数
        B = line_refine_times_[i]/1000 % 10 * 128;// 密度不足,需要调整
        Scalar lineColor = Scalar(B, G, R);

        if (_lines.depth() == CV_32F)
        {
            const Vec4f &v = _lines.at<Vec4f>(i);
            const Point2f b(v[0], v[1]);
            const Point2f e(v[2], v[3]);
            line(_image, b, e, lineColor, 1);
        }
        else
        {
            const Vec4i &v = _lines.at<Vec4i>(i);
            const Point2i b(v[0], v[1]);
            const Point2i e(v[2], v[3]);
            line(_image, b, e, lineColor, 1);
        }
    }
}

int LineSegmentDetectorMy::compareSegments(const Size& size, InputArray lines1, InputArray lines2, InputOutputArray _image)
{

    Size sz = size;
    if (_image.needed() && _image.size() != size) sz = _image.size();
    CV_Assert(!sz.empty());

    Mat_<uchar> I1 = Mat_<uchar>::zeros(sz);
    Mat_<uchar> I2 = Mat_<uchar>::zeros(sz);

    Mat _lines1 = lines1.getMat();
    Mat _lines2 = lines2.getMat();
    const int N1 = _lines1.checkVector(4);
    const int N2 = _lines2.checkVector(4);

    CV_Assert(_lines1.depth() == CV_32F || _lines1.depth() == CV_32S);
    CV_Assert(_lines2.depth() == CV_32F || _lines2.depth() == CV_32S);

    if (_lines1.depth() == CV_32S)
        _lines1.convertTo(_lines1, CV_32F);
    if (_lines2.depth() == CV_32S)
        _lines2.convertTo(_lines2, CV_32F);

    // Draw segments
    for(int i = 0; i < N1; ++i)
    {
        const Point2f b(_lines1.at<Vec4f>(i)[0], _lines1.at<Vec4f>(i)[1]);
        const Point2f e(_lines1.at<Vec4f>(i)[2], _lines1.at<Vec4f>(i)[3]);
        line(I1, b, e, Scalar::all(255), 1);
    }
    for(int i = 0; i < N2; ++i)
    {
        const Point2f b(_lines2.at<Vec4f>(i)[0], _lines2.at<Vec4f>(i)[1]);
        const Point2f e(_lines2.at<Vec4f>(i)[2], _lines2.at<Vec4f>(i)[3]);
        line(I2, b, e, Scalar::all(255), 1);
    }

    // Count the pixels that don't agree
    Mat Ixor;
    bitwise_xor(I1, I2, Ixor);
    int N = countNonZero(Ixor);

    if (_image.needed())
    {
        CV_Assert(_image.channels() == 3);
        Mat img = _image.getMatRef();
        CV_Assert(img.isContinuous() && I1.isContinuous() && I2.isContinuous());

        for (unsigned int i = 0; i < I1.total(); ++i)
        {
            uchar i1 = I1.ptr()[i];
            uchar i2 = I2.ptr()[i];
            if (i1 || i2)
            {
                unsigned int base_idx = i * 3;
                if (i1) img.ptr()[base_idx] = 255;
                else img.ptr()[base_idx] = 0;
                img.ptr()[base_idx + 1] = 0;
                if (i2) img.ptr()[base_idx + 2] = 255;
                else img.ptr()[base_idx + 2] = 0;
            }
        }
    }

    return N;
}

} // namespace cv