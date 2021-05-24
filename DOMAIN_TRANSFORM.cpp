#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <limits.h>
#include <float.h>
#include "DOMAIN_TRANSFORM.h"

using namespace CFL2;

DOMAIN_TRNASFORM::DOMAIN_TRNASFORM():sqrt3(sqrt((double)3))
{

}

DOMAIN_TRNASFORM::~DOMAIN_TRNASFORM()
{
	delete[] row_data_array;
	delete[] col_data_array;
	delete[] cum_row;
	delete[] cum_col;
}

void DOMAIN_TRNASFORM::Execute_Transform(IplImage* joint_image, const double& sigma_s,const double& sigma_r)
{
//#pragma message ("DOMAIN_TRNASFORM::Execute_Transform(const double& sigma_s,const double& sigma_r) will be removed")
	int	width,height,channel;
	if(joint_image!=0)
	{
		width=joint_image->width;
		height=joint_image->height;
		channel=joint_image->nChannels;
		row_data_array=new double*[height];
		cum_row=new double*[height];
		for(int i=0;i<height;i++)
		{
			row_data_array[i]=new double[width];
			cum_row[i]=new double[width];
		}
		col_data_array=new double*[width];
		cum_col=new double*[width];
		for(int i=0;i<width;i++)
		{
			col_data_array[i]=new double[height];
			cum_col[i]=new double[height];
		}
	}
	else
	{
		std::cout<<"joint image is invalid"<<std::endl;
	}

	const double sigma_s_over_r=sigma_s/sigma_r;
#pragma omp parallel for
	for(int i=0;i<height;i++){double cumulator=(double)0;
		for(int j=0;j<width;j++){

			double pixel_diff=(double)0;int col=j,colh=j+1;
			
			if(colh>=width)colh=j;
			for(int c=0;c<channel;c++){
				double _xh=((double)(unsigned char)(joint_image->imageData + i*joint_image->widthStep)[colh*channel+c])/255.0f;
				double _x=((double)(unsigned char)(joint_image->imageData + i*joint_image->widthStep)[col*channel+c])/255.0f;
				pixel_diff+=fabs(_xh-_x);}

			if(j==width-1)pixel_diff=DBL_MAX;
			row_data_array[i][j]=(1+sigma_s_over_r*pixel_diff);
			cumulator+=row_data_array[i][j];
			cum_row[i][j]=cumulator;
		}
	}

#pragma omp parallel for
	for(int i=0;i<width;i++){double cumulator=(double)0;
		for(int j=0;j<height;j++){
			double pixel_diff=(double)0;
			int row=j,rowh=j+1;
			if(rowh>=height)rowh=j;

			for(int c=0;c<channel;c++){
				double _xh=((double)(unsigned char)(joint_image->imageData + row*joint_image->widthStep)[i*channel+c])/255.0f;
				double _x=((double)(unsigned char)(joint_image->imageData + rowh*joint_image->widthStep)[i*channel+c])/255.0f;
				pixel_diff+=fabs(_xh-_x);}
			
			if(j==height-1)pixel_diff=DBL_MAX;
			//cumulator+=(1+sigma_s_over_r*pixel_diff);
			col_data_array[i][j]=(1+sigma_s_over_r*pixel_diff);
			cumulator+=col_data_array[i][j];
			cum_col[i][j]=cumulator;

		}
	}
}


IplImage* DOMAIN_TRNASFORM::Normalized_Convolution(VARIABLES)
{
	if(joint_image==NULL)
	{
		joint_image=src_image;
	}
	else if(src_image->width != joint_image->width || src_image->height != joint_image->height)
	{
		std::cout<<"The size is different"<<std::endl;
		exit(0);
	}
	
	Execute_Transform(joint_image,sigma_s,sigma_r);
	
	IplImage* output=cvCreateImage(cvSize(src_image->width,src_image->height),src_image->depth,src_image->nChannels);
	cvCopy(src_image,output);
	int width=output->width,height=output->height;

	for(int iter=0;iter<N;iter++)
	{
		double sigma_H_i=sigma_H*sqrt(3.0f)*pow(2.0,(double)(N-(iter+1)))/sqrt(pow(4.0,(double)N)-1);
		double kernel_radius=sqrt3*sigma_H_i;

		Normalized_Kernel(*output,cum_row,width,height,kernel_radius);
		Transpose_Image(*output);
		Normalized_Kernel(*output,cum_col,height,width,kernel_radius);
		Transpose_Image(*output);
	}

	return output;
}

IplImage* DOMAIN_TRNASFORM::Interpolated_Convolution(VARIABLES)
{
	if(joint_image==NULL)
	{
		joint_image=src_image;
	}
	else if(src_image->width != joint_image->width || src_image->height != joint_image->height)
	{
		std::cout<<"The size is different"<<std::endl;
		exit(0);
	}

	Execute_Transform(joint_image,sigma_s,sigma_r);
	IplImage* output=new IplImage(*src_image);
	int width=output->width,height=output->height;
	
	for(int iter=0;iter<N;iter++)
	{
		double sigma_H_i=sigma_H*sqrt(3.0f)*pow(2.0,(double)(N-(iter+1)))/sqrt(pow(4.0,(double)N)-1);
		double kernel_radius=sqrt(3.0f)*sigma_H_i;

		Interpolated_Kernel(*output,cum_row,width,height,kernel_radius);
		Transpose_Image(*output);
		Interpolated_Kernel(*output,cum_col,height,width,kernel_radius);
		Transpose_Image(*output);
	}

	return output;
}

IplImage* DOMAIN_TRNASFORM::Recursive_Filtering(VARIABLES)
{
	if(joint_image==NULL)
	{
		joint_image=src_image;
	}
	else if(src_image->width != joint_image->width || src_image->height != joint_image->height)
	{
		std::cout<<"(Joint iamge size != source image size)"<<std::endl;
		exit(0);
	}

	Execute_Transform(joint_image,sigma_s,sigma_r);
	IplImage* output=cvCreateImage(cvSize(src_image->width,src_image->height),src_image->depth,src_image->nChannels);
	IplImage* prev=cvCreateImage(cvSize(src_image->width,src_image->height),src_image->depth,src_image->nChannels);
	cvCopy(src_image,output);

	int width=output->width,height=output->height;
	for(int iter=0;iter<N;iter++)
	{
		double sigma_H_i=sigma_H*sqrt(3.0f)*pow(2.0,(double)(N-(iter+1)))/sqrt(pow(4.0,(double)N)-1);

		//Prev variable does not need to be initialized.
		Recursive_Kernel(*output,row_data_array,width,height,sigma_H_i,*prev);
		Transpose_Image(*output);Transpose_Image(*prev,false);

		Recursive_Kernel(*output,col_data_array,height,width,sigma_H_i,*prev);
		Transpose_Image(*output);Transpose_Image(*prev,false);
	}
	cvReleaseImage(&prev);
	return output;
}

void DOMAIN_TRNASFORM::Transpose_Image(IplImage& I,bool preserve_data)
{
	IplImage* J;
	J=new IplImage(I);
 	I=*cvCreateImage(cvSize(J->height,J->width),J->depth,J->nChannels);

	if(preserve_data)
	{
#pragma omp parallel for
		for(int i=0;i<J->height;i++){
			for(int j=0;j<J->width;j++){
				for(int c=0;c<J->nChannels;c++){
					(I.imageData + j*I.widthStep)[i*I.nChannels+c]=(J->imageData + i*J->widthStep)[j*I.nChannels+c];}}}
	}

	delete J;
}

//#define __BRUTE_FORCE__
void DOMAIN_TRNASFORM::Normalized_Kernel(IplImage& I,double** data,int w,int h,double kernel_radius)
{
	IplImage* J;
	J=cvCreateImage(cvSize(I.width,I.height),I.depth,I.nChannels);
	cvCopy(&I,J);

	//I=*cvCreateImage(cvSize(w,h),I.depth,I.nChannels);
	int row_length=h;
	int col_length=w;

#pragma omp parallel for
	for(int j=0;j<row_length;j++){
		double cumulator[3];for(int m=0;m<3;m++)cumulator[m]=0;
		int prev_min_index=0,prev_max_index=0;
		int min_index=0,max_index=col_length;

		for(int i=0;i<col_length;i++){
			int stands=MAX(0,i-1);int nStands=MAX(0,stands-1);

			double data_st=data[j][stands];
			for(int ki=MAX(prev_min_index-1,0);ki<=stands;ki++){
				int ix=ki-1;double data_cur;
				if(ix<0)data_cur=0.0f;else data_cur=data[j][ki];
				if(data_st-data_cur<kernel_radius){min_index=ki+1;break;}}if(min_index<0)min_index=0;

				for(int ki=prev_max_index-1;ki<col_length;ki++){
					if(data[j][ki]-data[j][stands]>kernel_radius){max_index=ki+1;break;}}
				if(max_index>=col_length)max_index=col_length;

#ifndef __BRUTE_FORCE__
			for(int k=prev_min_index;k<min_index;k++){
				for(int c=0;c<J->nChannels;c++){
					cumulator[c]-=((double)(unsigned char)((J->imageData+j*J->widthStep)[k*J->nChannels+c]))/255.0f;}}

			for(int k=prev_max_index;k<max_index;k++){
				for(int c=0;c<J->nChannels;c++){
					cumulator[c]+=((double)(unsigned char)((J->imageData+j*J->widthStep)[k*J->nChannels+c]))/255.0f;}}

#else
			for(int m=0;m<3;m++)cumulator[m]=0;
			for(int k=min_index;k<max_index;k++){
				for(int c=0;c<J->nChannels;c++){
					cumulator[c]+=((double)(unsigned char)((J->imageData+j*J->widthStep)[k*J->nChannels+c]))/255.0f;}}

#endif
			double normalize_coeff=1.0/(double)MAX(1,max_index-min_index);

			for(int c=0;c<J->nChannels;c++)
			{
				(I.imageData+j*I.widthStep)[i*I.nChannels+c]=(unsigned char)(int)round(normalize_coeff*cumulator[c]*255);
			}
			prev_min_index=min_index;
			prev_max_index=max_index;
		}
	}

	//delete J;
	cvReleaseImage(&J);
}

void DOMAIN_TRNASFORM::Interpolated_Kernel( IplImage& I,double** data,int w,int h,double kernel_radius )
{
	IplImage* J;
	J=new IplImage(I);
	memcpy(J,&I,sizeof(I));
	I=*cvCreateImage(cvSize(w,h),I.depth,I.nChannels);

	int row_length=h;
	int col_length=w;

	double rr=kernel_radius*2.0f;
	double one_over_2r=(double)1.0f/kernel_radius*0.5f;

#pragma omp parallel for
	for(int j=0;j<row_length;j++){
		double cumulator[3],accumulator[3];
		int min_index=0,max_index=0;int prev_min_index=0,prev_max_index=0;
		double* row_data=data[j];
		
		for(int m=0;m<3;m++)accumulator[m]=0;
		double sum_of_distance=0.0f;

		for(int i=0;i<col_length;i++){
#ifdef __BRUTE_FORCE__
			for(int m=0;m<3;m++)accumulator[m]=0;
			sum_of_distance=0;
#endif
			int min_index=0,max_index=col_length;
			int stands=MAX(0,i-1);

			double data_st=data[j][stands];
			for(int ki=MAX(prev_min_index-1,0);ki<=stands;ki++){
				int ix=ki-1;double data_cur=data[j][ki];
				if(ix<0)data_cur=0.0f;
				if(data_st-data_cur<kernel_radius){min_index=ki+1;break;}}
				if(min_index<0)min_index=0;

			for(int ki=prev_max_index-1;ki<col_length;ki++){
				if(data[j][ki]-data[j][stands]>kernel_radius){max_index=ki+1;break;}}
			if(max_index>=col_length)max_index=col_length;
//			double data_st=row_data[stands];
//			for(int ki=prev_min_index-1;ki<=stands;ki++){
//				int ix=ki-1;double data_cur=row_data[ki];if(ix<0)data_cur=0.0f;
//				if(data_st-data_cur<kernel_radius){min_index=ki+1;break;}}
//			min_index=MAX(0,min_index);

			//for(int ki=i;ki<col_length;ki++){
			//	if(row_data[ki]-row_data[stands]>kernel_radius){max_index=ki+1;break;}}
			//max_index=MIN(max_index,col_length);
			
			char* image_row=(J->imageData+j*J->widthStep);

#ifndef __BRUTE_FORCE__
			//Center remove
			for(int k=prev_min_index;k<min_index;k++){
				int xi=k,xhi=k+1;int di=MAX(0,xi-1);
				double step_size=data[j][xi]-data[j][di];
				sum_of_distance-=step_size;

				for(int c=0;c<I.nChannels;c++){
					double step_height=((double)(unsigned char)(image_row[xi*I.nChannels+c]))/255.0f+((double)(unsigned char)(image_row[xhi*I.nChannels+c]))/255.0f; 
					accumulator[c]-=(0.5*step_size*step_height);}}

			//Center add
			for(int k=prev_max_index;k<max_index-1;k++){
				int xi=k,xhi=k+1;int di=MAX(0,xi-1);
				double step_size=data[j][xi]-data[j][di];
				sum_of_distance+=step_size;

				for(int c=0;c<I.nChannels;c++){
					double step_height=((double)(unsigned char)(image_row[xi*I.nChannels+c]))/255.0f+((double)(unsigned char)(image_row[xhi*I.nChannels+c]))/255.0f; 
					accumulator[c]+=(0.5*step_size*step_height);}}
			
#else
			for(int k=min_index;k<max_index-1;k++){
				int xi=k,xhi=k+1;if(xi<min_index)xi=min_index;if(xhi>max_index-1)xhi=max_index-1;
				int di=MAX(0,xi-1);
				double step_size=data[j][xi]-data[j][di];
				sum_of_distance+=step_size;

				for(int c=0;c<I.nChannels;c++){
					double step_height=((double)(unsigned char)(image_row[xi*I.nChannels+c]))/255.0f+((double)(unsigned char)(image_row[xhi*I.nChannels+c]))/255.0f; 
					accumulator[c]+=(0.5*step_size*step_height);}}
#endif

			for(int ch=0;ch<I.nChannels;ch++)cumulator[ch]=accumulator[ch];
			double current_distance=sum_of_distance;
			
			//Left Area
			int ldi=MAX(0,min_index-1);double bPart=kernel_radius-(data_st-data[j][ldi]);
			if(bPart>0.0f){
				current_distance+=bPart;
				int lldi=MAX(0,ldi-1);
				for(int c=0;c<I.nChannels;c++){
					double value=0.0f;double tX=data[j][ldi]-data[j][lldi];
					double stand=((double)(unsigned char)(image_row[min_index*I.nChannels+c]))/255.0f;
					if(tX>0){
						double step_grad=(((double)(unsigned char)(image_row[ldi*I.nChannels+c]))/255.0f-stand);
						value=stand+step_grad*(bPart/tX);}
					else{
						value=stand;}
					cumulator[c]+=(0.5*(value+stand)*bPart);}}
			
			
			//Right Area
			int udi=MAX(0,max_index-1);int udi_d=udi-1;
			double uPart=kernel_radius-(data[j][udi_d]-data_st);
			if(uPart>0.0f){
				current_distance+=uPart;
				for(int c=0;c<I.nChannels;c++){
					int uudi=udi+1;
					double value=0.0f;
					double stand=((double)(unsigned char)(image_row[udi*I.nChannels+c]))/255.0f;
					double tX=(data[j][udi]-data[j][udi_d]);
					if(uudi<col_length){
						double step_grad=((double)(unsigned char)(image_row[uudi*I.nChannels+c]))/255.0f-stand;
						value=stand+step_grad*(uPart/tX);}
					else{
						value=stand;}
					cumulator[c]+=(0.5*(value+stand)*uPart);}}
			
			//for(int c=0;c<I.nChannels;c++)(I.imageData+j*I.widthStep)[i*I.nChannels+c]=(unsigned char)(int)(255.0f*/*one_over_2r*/(1.0f/current_distance)*(cumulator[c]));
			for(int c=0;c<I.nChannels;c++)(I.imageData+j*I.widthStep)[i*I.nChannels+c]=(unsigned char)(int)(255.0f*one_over_2r*(cumulator[c]));

			prev_min_index=min_index;
			prev_max_index=max_index-1;
		}
	}
	delete J;
}


void DOMAIN_TRNASFORM::Recursive_Kernel(IplImage& I, double** data,int w,int h,double kernel_radius, IplImage& J)
{
	const double feedback_coefficient=exp(-sqrt((double)2)/kernel_radius);

	int row_cnt=h;
	int col_cnt=w;
	double D2U=255.0f;
	double U2D=1.0f/D2U;

	cvCopy(&I,&J);
#pragma omp parallel for
	for(int j=0;j<row_cnt;j++){
		for(int i=1;i<col_cnt;i++){
			int i_minus_one=MAX(0,i-1);
			double coeff=pow(feedback_coefficient,data[j][i_minus_one]);
			for(int c=0;c<I.nChannels;c++)
			{
				double cur_px=((double)(unsigned char)(I.imageData+j*I.widthStep)[i*I.nChannels+c])*U2D;
				double left_px=((double)(unsigned char)(I.imageData+j*I.widthStep)[i_minus_one*I.nChannels+c])*U2D;

				(I.imageData+j*I.widthStep)[i*I.nChannels+c]=(unsigned char)(int)round((cur_px+coeff*(left_px-cur_px))*D2U);
			}
		}
	}

#pragma omp parallel for
	for(int j=0;j<row_cnt;j++){
		for(int i=col_cnt-1;i>=0;i--){
			int i_plus_one=MIN(col_cnt-1,i+1);
			double coeff=pow(feedback_coefficient,data[j][i]);
			for(int c=0;c<I.nChannels;c++)
			{
				double cur_px=((double)(unsigned char)(I.imageData+j*I.widthStep)[i*I.nChannels+c])*U2D;
				double right_px=((double)(unsigned char)(I.imageData+j*I.widthStep)[i_plus_one*I.nChannels+c])*U2D;
				(I.imageData+j*I.widthStep)[i*I.nChannels+c]=(unsigned char)(int)round((cur_px+coeff*(right_px-cur_px))*D2U);
			}
		}
	}
}

void DOMAIN_TRNASFORM::Execute_TransformD(IplImage* joint_image, IplImage* depth_image, SETTINGS *settings)
{
	double sigma_s = settings->sigma_s;
	double sigma_r = settings->sigma_r;
	double sigma_d = settings->sigma_d;

	int	width, height, channel;
	if (joint_image != 0)
	{
		width = joint_image->width;
		height = joint_image->height;
		channel = joint_image->nChannels;
		row_data_array = new double*[height];		
		cum_row = new double*[height];
		sigma_sp = new double*[height];
		for (int i = 0; i < height; i++)
		{
			row_data_array[i] = new double[width];
			cum_row[i] = new double[width];
			sigma_sp[i] = new double[width];
		}
		col_data_array = new double*[width];
		cum_col = new double*[width];
		
		for (int i = 0; i < width; i++)
		{
			col_data_array[i] = new double[height];
			cum_col[i] = new double[height];			
		}

		if (depth_image == NULL) std::cout << "No depth image available" << std::endl;
		else if ((depth_image->width != width) || (depth_image->height != height))
		{
			std::cout << "Depth image is invalid" << std::endl;
			exit(0);
		}
	}
	else
	{
		std::cout << "joint image is invalid" << std::endl;
	}
		
#pragma omp parallel for
	for (int i = 0; i < height; i++){
		double cumulator = (double)0;
		for (int j = 0; j < width; j++){
			
			double pixel_diff = (double)0; 
			double depth_diff = (double)0;
			int col = j, colh = j + 1;
			if (colh >= width)colh = j;
			
			double _zh = 1.0f - ((double)(uchar)(depth_image->imageData + i*depth_image->widthStep)[colh]) / 255.0f;
			double _z = 1.0f - ((double)(uchar)(depth_image->imageData + i*depth_image->widthStep)[col]) / 255.0f;
			//SIGMA SP
			sigma_sp[i][j] = sigma_s * settings->focalLength / (_z*255 -30 + settings->focalLength);

			depth_diff = fabs(_zh - _z);

			for (int c = 0; c < channel; c++){
				double _xh = ((double)(uchar)(joint_image->imageData + i*joint_image->widthStep)[colh*channel + c]) / 255.0f;
				double _x = ((double)(uchar)(joint_image->imageData + i*joint_image->widthStep)[col*channel + c]) / 255.0f;
				pixel_diff += fabs(_xh - _x);
			}
			

			if (j == width - 1){
				pixel_diff = DBL_MAX;
				depth_diff = DBL_MAX;
			}			
			double sigma_s_over_r = sigma_sp[i][j] / sigma_r;
			double sigma_s_over_d = sigma_sp[i][j] / sigma_d;
			//cout << sigma_s_over_r*pixel_diff << "  " << sigma_s_over_d * depth_diff << endl;
			row_data_array[i][j] = (1 + sigma_s_over_r*pixel_diff + sigma_s_over_d*depth_diff);
			cumulator += row_data_array[i][j];
			cum_row[i][j] = cumulator;
		}
	}

#pragma omp parallel for
	for (int i = 0; i < width; i++){
		double cumulator = (double)0;
		for (int j = 0; j < height; j++){

			double pixel_diff = (double)0;
			double depth_diff = (double)0;
			int row = j, rowh = j + 1;
			if (rowh >= height)rowh = j;

			double _zh = 1.0f - ((double)(uchar)(depth_image->imageData + row*depth_image->widthStep)[i]) / 255.0f;
			double _z = 1.0f - ((double)(uchar)(depth_image->imageData + rowh*depth_image->widthStep)[i]) / 255.0f;			
						
			depth_diff = fabs(_zh - _z);
			
			for (int c = 0; c < channel; c++){
				double _xh = ((double)(uchar)(joint_image->imageData + row*joint_image->widthStep)[i*channel + c]) / 255.0f;
				double _x = ((double)(uchar)(joint_image->imageData + rowh*joint_image->widthStep)[i*channel + c]) / 255.0f;
				pixel_diff += fabs(_xh - _x);				
			}
			
			if (j == height - 1){
				pixel_diff = DBL_MAX;
				depth_diff = DBL_MAX;
			}
			double sigma_s_over_r = sigma_sp[j][i] / sigma_r;
			double sigma_s_over_d = sigma_sp[j][i] / sigma_d;

			//cout << sigma_s_over_r*pixel_diff << "  " << sigma_s_over_d*depth_diff << endl;
			//cumulator+=(1+sigma_s_over_r*pixel_diff);
			
			col_data_array[i][j] = (1 + sigma_s_over_r*pixel_diff + sigma_s_over_d*depth_diff);
			cumulator += col_data_array[i][j];
			cum_col[i][j] = cumulator;

		}
	}
}

IplImage* DOMAIN_TRNASFORM::Rolling_Guidance(SETTINGS *settings, IplImage *src, IplImage *dep)
{
	CvSize sz;
	sz.width = src->width; sz.height = src->height;
	IplImage *guide = cvCreateImage(sz, IPL_DEPTH_8U, src->nChannels);
	//cvZero(guide);
	//guide = Normalized_ConvolutionD(src, NULL, dep, settings);
	//guide = Normalized_Convolution(src, settings->sigma_s, settings->sigma_r, settings->sigma_h, settings->num_iterDT, NULL);
	for (int i = 0; i < settings->num_iterRoll; i++){
		if (dep != NULL ) 
			guide = Normalized_ConvolutionD(src, guide, dep, settings);
		else
			guide = Normalized_Convolution(src, settings->sigma_s, settings->sigma_r, settings->sigma_h, settings->num_iterDT, NULL);
	}

	return guide;
}

IplImage* DOMAIN_TRNASFORM::Normalized_ConvolutionD(IplImage *src_image, IplImage *joint_image, IplImage *depth_image, SETTINGS *settings)
{	
	double sigma_H = settings->sigma_h;
	int N = settings->num_iterDT;

	if (joint_image == NULL)
	{
		joint_image = src_image;
	}
	else if (src_image->width != joint_image->width || src_image->height != joint_image->height)
	{
		std::cout << "The size is different" << std::endl;
		exit(0);
	}

	Execute_TransformD(joint_image, depth_image, settings);
	//Execute_Transform(joint_image, settings->sigma_s, settings->sigma_r);
	IplImage* output = cvCreateImage(cvSize(src_image->width, src_image->height), src_image->depth, src_image->nChannels);
	cvCopy(src_image, output);
	int width = output->width, height = output->height;
	
	for (int iter = 0; iter < N; iter++)
	{
 		double sigma_H_i = sigma_H*sqrt(3.0f)*pow(2.0, (double)(N - (iter + 1))) / sqrt(pow(4.0, (double)N) - 1);
 		double kernel_radius = sqrt3*sigma_H_i;		
		//cout << kernel_radius << endl;
		Normalized_KernelD(*output, cum_row, width, height, iter, N);
		//Normalized_Kernel(*output, cum_row, width, height, kernel_radius);
		Transpose_Image(*output);
		
		Transpose_Sigma(&sigma_sp, width, height);
		
		Normalized_KernelD(*output, cum_col, height, width, iter, N);
		//Normalized_Kernel(*output, cum_col, height, width, kernel_radius);
		Transpose_Image(*output);
		Transpose_Sigma(&sigma_sp, height, width);
	}

	


	return output;
}

void CFL2::DOMAIN_TRNASFORM::Normalized_KernelD(IplImage& I, double** data, int w, int h, int iter, int N)
{
	IplImage* J;
	J = cvCreateImage(cvSize(I.width, I.height), I.depth, I.nChannels);
	cvCopy(&I, J);

	//I=*cvCreateImage(cvSize(w,h),I.depth,I.nChannels);
	int row_length = h;
	int col_length = w;

#pragma omp parallel for
	for (int j = 0; j < row_length; j++){
		double cumulator[3]; for (int m = 0; m < 3; m++)cumulator[m] = 0;
		int prev_min_index = 0, prev_max_index = 0;
		int min_index = 0, max_index = col_length;

		for (int i = 0; i < col_length; i++){

			double sigma_H_i = sigma_sp[j][i] * sqrt(3.0f)*pow(2.0, (double)(N - (iter + 1))) / sqrt(pow(4.0, (double)N) - 1);
			double kernel_radius = sqrt3*sigma_H_i;


			//cout << kernel_radius << endl;
			int stands = MAX(0, i - 1); int nStands = MAX(0, stands - 1);

			double data_st = data[j][stands];
			for (int ki = MAX(prev_min_index - 1, 0); ki <= stands; ki++){
				int ix = ki - 1; double data_cur;
				if (ix < 0)data_cur = 0.0f; else data_cur = data[j][ki];
				if (data_st - data_cur < kernel_radius){ min_index = ki + 1; break; }
			}if (min_index < 0)min_index = 0;

			for (int ki = prev_max_index - 1; ki < col_length; ki++){
				if (data[j][ki] - data[j][stands] > kernel_radius){ max_index = ki + 1; break; }
			}
			if (max_index >= col_length)max_index = col_length;

#ifndef __BRUTE_FORCE__
			for (int k = prev_min_index; k < min_index; k++){
				for (int c = 0; c < J->nChannels; c++){
					cumulator[c] -= ((double)(unsigned char)((J->imageData + j*J->widthStep)[k*J->nChannels + c])) / 255.0f;
				}
			}

			for (int k = prev_max_index; k < max_index; k++){
				for (int c = 0; c < J->nChannels; c++){
					cumulator[c] += ((double)(unsigned char)((J->imageData + j*J->widthStep)[k*J->nChannels + c])) / 255.0f;
				}
			}

#else
			for (int m = 0; m < 3; m++)cumulator[m] = 0;
			for (int k = min_index; k < max_index; k++){
				for (int c = 0; c < J->nChannels; c++){
					cumulator[c] += ((double)(unsigned char)((J->imageData + j*J->widthStep)[k*J->nChannels + c])) / 255.0f;
				}
			}

#endif
			double normalize_coeff = 1.0 / (double)MAX(1, max_index - min_index);

			for (int c = 0; c < J->nChannels; c++)
			{
				(I.imageData + j*I.widthStep)[i*I.nChannels + c] = (unsigned char)(int)round(normalize_coeff*cumulator[c] * 255);
			}
			prev_min_index = min_index;
			prev_max_index = max_index;
		}
	}

	//delete J;
	cvReleaseImage(&J);
}

void CFL2::DOMAIN_TRNASFORM::Transpose_Sigma(double*** data,  int width, int height)
{	
	double **temp = new double*[height];
	for (int i = 0; i < height; i++){
		temp[i] = new double[width];
		memcpy(temp[i], (*data)[i], sizeof(double)*width);
		delete[](*data)[i];
	}
	delete[] (*data);
		
	(*data) = new double*[width];
	for (int i = 0; i < width; i++){		
		(*data)[i] = new double[height];
	}
	
	for (int i = 0; i < width; i++){
		for (int j = 0; j < height; j++){
			(*data)[i][j] = temp[j][i];
		}
	}
		
	for (int i = 0; i < height; i++){
		delete[] temp[i];
	}
	delete[] temp;	
}

