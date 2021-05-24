#ifndef __DOMAIN_TRANSFORM__
#define __DOMAIN_TRANSFORM__

#include <opencv2/opencv.hpp>
#include "TIME_UTILITIES.h"
#include "SETTINGS.h"

#define VARIABLES IplImage* src_image, const double& sigma_s,const double& sigma_r, const double& sigma_H,int N,IplImage* joint_image
#define DECLARE_VARIABLES VARIABLES=NULL

namespace CFL2
{
class DOMAIN_TRNASFORM
{
public:
	DOMAIN_TRNASFORM();
	~DOMAIN_TRNASFORM();

	IplImage* Rolling_Guidance(SETTINGS *settings, IplImage *src, IplImage *depth);
	IplImage* Normalized_Convolution(DECLARE_VARIABLES);	
	IplImage* Normalized_ConvolutionD(IplImage *src_image, IplImage *joint_image, IplImage *depth_image, SETTINGS *settings);
	IplImage* Interpolated_Convolution(DECLARE_VARIABLES);
	IplImage* Recursive_Filtering(DECLARE_VARIABLES);

private:
	void Execute_Transform(IplImage* joint_image, const double& sigma_s,const double& sigma_r);
	void Execute_TransformD(IplImage* joint_image, IplImage* depth_image, SETTINGS *settings);
	void Transpose_Image(IplImage& I,bool preserve_data=true);
	void Transpose_Sigma(double*** data, int width, int height);
	void Normalized_Kernel(IplImage& I,double** data,int w,int h,double kernel_radius);
	void Normalized_KernelD(IplImage& I, double** data, int w, int h,int iter, int N);
	void Interpolated_Kernel(IplImage& I,double** data,int w,int h,double kernel_radius);
	void Recursive_Kernel(IplImage& I,double** data,int w,int h,double kernel_radius, IplImage& J);

	int round(double x){return (int)(floor(x+0.5));}

	double		sqrt3;
	double**	row_data_array,**cum_row;
	double**	col_data_array,**cum_col;
	double**	sigma_sp;
};//End of class DOMAIN_TRANSFORM
}//End of namespace CFL2
#endif