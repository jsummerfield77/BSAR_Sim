//IPR_Kernel.cu
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
const double *f_c,const double *BW,const double *w2d, 
        int length_t, int Nf,int length_X)
{
    int tid = threadIdx.x +blockIdx.x*blockDim.x;
    while(tid<length_X)
    {
        double pi = 3.141592653589793;
        double c_sol = 299792458.0;
        double ratio = 2.0*pi/c_sol;

        cuComplex temp = cuComplex(0.0,0.0);
        for(int ii=0;ii< length_t;ii++)
        {
            
            double df = BW[ii]/(Nf-1);
            double f_start = f_c[ii]-.5*BW[ii];
            double r_dot_RGrad = X[tid]*RGrad_x[ii]+Y[tid]*RGrad_y[ii]+Z[tid]*RGrad_z[ii];
            cuComplex temp2 = cuComplex(0.0,0.0);
        
            for(int jj=0;jj< Nf;jj++)
            {
                double f_jj = f_start + jj*df;
                double wij = w2d[jj + ii*Nf];
                
                temp2 = temp2 + cuComplex(wij,0.0)*exp( cuComplex(0.0,ratio*f_jj*r_dot_RGrad));
            }
            // temp = temp + cuComplex(df,0.0)*temp2;
            temp = temp + temp2;
            
        }
        IPR_real[tid] = temp.r;
        IPR_imag[tid] = temp.i;
        tid += blockDim.x*gridDim.x;
    }
}