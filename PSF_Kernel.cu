//PSF_Kernel.cu
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

__global__ void PSF(double *PSF_real,double *PSF_imag,
const double *X,const double *Y,const double *Z,
const double *Pos_TX_x,const double *Pos_TX_y,const double *Pos_TX_z,
const double *Pos_RX_x,const double *Pos_RX_y,const double *Pos_RX_z,
double Pos_mid_x,double Pos_mid_y,double Pos_mid_z,
const double *f_c,const double *BW,const double *w2d, int length_t, int Nf,int length_X)
{
    int tid = threadIdx.x +blockIdx.x*blockDim.x;
    while(tid<length_X)
    {
        double pi = 3.141592653589793;
        double c_sol = 299792458;
        double ratio = 2*pi/c_sol;
        
        cuComplex temp = cuComplex(0.0,0.0);
        for(int ii=0;ii< length_t;ii++)
        {
            double df = BW[ii]/(Nf-1);
            double f_start = f_c[ii]-.5*BW[ii];

            double del_x_tx = Pos_TX_x[ii]-.5*X[tid]+Pos_mid_x;
            double del_y_tx = Pos_TX_y[ii]-.5*Y[tid]+Pos_mid_y;
            double del_z_tx = Pos_TX_z[ii]-.5*Z[tid]+Pos_mid_z;

            double del_x_rx = Pos_RX_x[ii]-.5*X[tid]+Pos_mid_x;
            double del_y_rx = Pos_RX_y[ii]-.5*Y[tid]+Pos_mid_y;
            double del_z_rx = Pos_RX_z[ii]-.5*Z[tid]+Pos_mid_z;

            double del_x_tx2 = Pos_TX_x[ii]+.5*X[tid]+Pos_mid_x;
            double del_y_tx2 = Pos_TX_y[ii]+.5*Y[tid]+Pos_mid_y;
            double del_z_tx2 = Pos_TX_z[ii]+.5*Z[tid]+Pos_mid_z;

            double del_x_rx2 = Pos_RX_x[ii]+.5*X[tid]+Pos_mid_x;
            double del_y_rx2 = Pos_RX_y[ii]+.5*Y[tid]+Pos_mid_y;
            double del_z_rx2 = Pos_RX_z[ii]+.5*Z[tid]+Pos_mid_z;


            double bi_range = sqrt( del_x_tx*del_x_tx + del_y_tx*del_y_tx + del_z_tx*del_z_tx) + sqrt( del_x_rx*del_x_rx + del_y_rx*del_y_rx + del_z_rx*del_z_rx);
            double bi_range2 = sqrt( del_x_tx2*del_x_tx2 + del_y_tx2*del_y_tx2 + del_z_tx2*del_z_tx2) + sqrt( del_x_rx2*del_x_rx2 + del_y_rx2*del_y_rx2 + del_z_rx2*del_z_rx2);
            //double range_0 = sqrt(Pos_TX_x[ii]*Pos_TX_x[ii] + Pos_TX_y[ii]*Pos_TX_y[ii]+ Pos_TX_z[ii]*Pos_TX_z[ii]) + sqrt(Pos_RX_x[ii]*Pos_RX_x[ii] + Pos_RX_y[ii]*Pos_RX_y[ii] + Pos_RX_z[ii]*Pos_RX_z[ii]);
            cuComplex temp2 = cuComplex(0.0,0.0);
        
            for(int jj=0;jj< Nf;jj++)
            {
                double f_jj = f_start + jj*df;
                double wij = w2d[jj + ii*Nf];
                temp2 = temp2 + cuComplex(wij,0.0)*exp( cuComplex(0.0,ratio*f_jj*(bi_range-bi_range2)));            
            }
            // temp = temp + cuComplex(df,0.0)*temp2;
            temp = temp + temp2;
            
        }
        PSF_real[tid] = temp.r;
        PSF_imag[tid] = temp.i;
        tid += blockDim.x*gridDim.x;

    }
}