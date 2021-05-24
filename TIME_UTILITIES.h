//#####################################################################
// Copyright 2013 Seunghoon Cha, CFL2.
// This file is part of CFL2 project.
//#####################################################################

#ifndef __TIME_UTILITIES__
#define __TIME_UTILITIES__

#ifdef WIN32
#include <iostream>
#include <windows.h>

namespace CFL2
{

class TIME_UTILITIES
{
public:
	TIME_UTILITIES(){
		QueryPerformanceFrequency(&m_timer);
		is_global_started=false;is_local=false;
	}
	~TIME_UTILITIES(){}

	void Start_Program_Timer()
	{
		if(!is_global_started)
		{
			std::cout<<"Program timer started"<<std::endl;
			QueryPerformanceCounter(&m_global_start);
			is_global_started=true;
		}
	}

	void Start_Local_Timer()
	{
		if(!is_local)
		{
#ifdef __REPORT__
			std::cout<<"Local timer started"<<std::endl;
#endif
			QueryPerformanceCounter(&m_local_start);
			is_local=true;
		}
	}

	double End_Program_Timer()
	{
		if(!is_global_started)return 0.0;is_global_started=false;
		LARGE_INTEGER global_end;
		QueryPerformanceCounter(&global_end);
		return (double)(global_end.QuadPart-m_global_start.QuadPart)/(double)m_timer.QuadPart;
	}

	double End_Local_Timer()
	{
		if(!is_local)return 0.0f;
		LARGE_INTEGER local_end;
		QueryPerformanceCounter(&local_end);is_local=false;
		return (double)(local_end.QuadPart-m_local_start.QuadPart)/(double)m_timer.QuadPart;
	}
	
private:
	static LARGE_INTEGER  m_timer,m_global_start;
	LARGE_INTEGER m_local_start;
	bool is_local;
	static bool is_global_started;


};

#else
class TIME_UTILITIES
{
public:
	TIME_UTILITIES(){}
	~TIME_UTILITIES(){}

	void Start_Program_Timer(){}
	void Start_Local_Timer(){}
	double End_Program_Timer(){}
	double End_Local_Timer(){}
};

#endif
}//End of namespace CFL2
#endif

