#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <string>
using namespace std;

class SETTINGS
{
public:
	SETTINGS(){
		memset(this, 0, sizeof(SETTINGS));
		sigma_r = 0.4;
		sigma_s = 60;
		sigma_h = sigma_s;
		sigma_d = 0.4;
		focalLength = 35;
		num_iterDT = 3;
		num_iterRoll = 5;
	}
	double sigma_r;
	double sigma_s;
	double sigma_h;
	double sigma_d;
	double focalLength;
	int num_iterDT;
	int num_iterRoll;
};

#endif //__SETTINGS_H__
