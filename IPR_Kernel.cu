// IPR_Kernel.cu
// Author: John Summerfield
//
// Description:
// GPU kernel for computing the impulse response (IPR) of a bistatic SAR system 
// using a stepped-frequency waveform. This kernel is designed to run on NVIDIA 
// GPUs and is part of a larger CUDA-accelerated SAR simulation framework.
//
// Model Update:
// This code represents a significant update to the original IPR simulation kernel 
// developed during my Ph.D. dissertation research. The original version was based 
// on the conventional move–stop–move (MSM) approximation, which assumes the radar 
// platform is stationary during each pulse and models the phasor using only the 
// transmit frequency and the bistatic range gradient vector.
//
// The updated kernel replaces the MSM assumption with a **constant fast-time velocity 
// (CFV)** model, which captures continuous platform motion during the pulse and 
// is valid for high-speed radar platforms, including orbital or hypersonic systems. 
// This change is essential for accurately modeling Doppler effects and signal 
// time-scaling phenomena when platform velocities are on the order of hundreds of 
// kilometers per second.
//
// Key Physics Changes:
// - **MSM model:** Phasor used only the stepped frequency term and the bistatic range 
//   gradient vector (∇R) to represent phase evolution.
// - **CFV model (this version):** Phasor now incorporates a more complete physical 
//   description that includes:
//      • Stepped frequency term (f)  
//      • Bistatic range gradient vector (∇R)  
//      • Bistatic range rate gradient vector (∇Ṙ)  
//      • Bistatic range from transmitter to scene center to receiver (R₀)
//
// These additional terms account for Doppler-time scaling and higher-order motion 
// effects, enabling accurate IPR computation under continuous motion and high-speed 
// conditions.
//
// Notes:
// - The kernel computes complex impulse response values (real and imaginary parts) 
//   for each spatial sample point in the scene.
// - Computation is fully parallelized across GPU threads to support large-scale 
//   IPR simulations efficiently.
// - This implementation is compatible with stepped-frequency waveform synthesis 
//   and frequency-agile waveform strategies used elsewhere in the simulation framework.

#include "math.h"

struct cuComplex {
    double   r;
    double   i;
    __device__ cuComplex( double a, double b ) : r(a), i(b)  {}
    __device__ double magnitude2( void ){ 
				return r * r + i * i; 
				}
    __device__ cuComplex operator*(const cuComplex& a) {
        		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    			}
    __device__ cuComplex operator+(const cuComplex& a) {
        		return cuComplex(r+a.r, i+a.i);
    			}
};



__device__ cuComplex exp(cuComplex x)
{
    return  cuComplex( exp(x.r)*cos(x.i), exp(x.r)*sin(x.i));            
}

__global__ void IPR(double *IPR_real,double *IPR_imag,
const double *X,const double *Y,const double *Z,
const double *RGrad_x,const double *RGrad_y,const double *RGrad_z,
const double *RRGrad_x,const double *RRGrad_y,const double *RRGrad_z,
const double *Grad_O, const double *f_c,const double *BW,const double *w2d, 
        int length_t, int Nf,int length_X)
{
    int tid = threadIdx.x +blockIdx.x*blockDim.x;
    while(tid<length_X)
    {
        double pi = 3.141592653589793;
        double c_sol = 299792458.0;
        double ratio = 2.0*pi/c_sol;
        double ratio2 = 2.0/c_sol;

        cuComplex temp = cuComplex(0.0,0.0);
        for(int ii=0;ii< length_t;ii++)
        {
            
            double df = BW[ii]/(Nf-1);
            double f_start = f_c[ii]-.5*BW[ii];

            double F_x = RGrad_x[ii] + ratio2*Grad_O[ii]*RRGrad_x[ii];
            double F_y = RGrad_y[ii] + ratio2*Grad_O[ii]*RRGrad_y[ii];
            double F_z = RGrad_z[ii] + ratio2*Grad_O[ii]*RRGrad_z[ii];
            double r_dot_F = X[tid]*F_x + Y[tid]*F_y + Z[tid]*F_z;

            // double r_dot_RGrad = X[tid]*RGrad_x[ii]+Y[tid]*RGrad_y[ii]+Z[tid]*RGrad_z[ii];
            cuComplex temp2 = cuComplex(0.0,0.0);
        
            for(int jj=0;jj< Nf;jj++)
            {
                double f_jj = f_start + jj*df;
                double wij = w2d[jj + ii*Nf];
                
                // temp2 = temp2 + cuComplex(wij,0.0)*exp( cuComplex(0.0,ratio*f_jj*r_dot_RGrad));
                temp2 = temp2 + cuComplex(wij,0.0)*exp( cuComplex(0.0,ratio*f_jj*r_dot_F));
                
            }
            // temp = temp + cuComplex(df,0.0)*temp2;
            temp = temp + temp2;
            
        }
        IPR_real[tid] = temp.r;
        IPR_imag[tid] = temp.i;
        tid += blockDim.x*gridDim.x;
    }
}
