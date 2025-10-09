//MF_Kernel.cu
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

__global__ void MF_Kernel(double *SAR_real,double *SAR_imag,
const double *PhaseHist_real,const double *PhaseHist_imag,
const double *X,const double *Y,const double *Z,
const double *Pos_TX_x,const double *Pos_TX_y,const double *Pos_TX_z,
const double *Pos_RX_x,const double *Pos_RX_y,const double *Pos_RX_z,
const double *f_c,const double *BW,
const double *w2d, int length_t, int Nf,int length_X)
{
    int tid = threadIdx.x +blockIdx.x*blockDim.x;
    while(tid<length_X)
    {
        double pi = 3.141592653589793;
        double c_sol = 299792458.0;
        
        cuComplex temp = cuComplex(0.0,0.0);
        for(int ii=0;  ii< length_t;  ii++)
        {
            double df = BW[ii]/(Nf-1);
            double f_start = f_c[ii]-.5*BW[ii];
            
            double del_x_tx = Pos_TX_x[ii]-X[tid];
            double del_y_tx = Pos_TX_y[ii]-Y[tid];
            double del_z_tx = Pos_TX_z[ii]-Z[tid];
            
            double del_x_rx = Pos_RX_x[ii]-X[tid];
            double del_y_rx = Pos_RX_y[ii]-Y[tid];
            double del_z_rx = Pos_RX_z[ii]-Z[tid];
            
            double bi_range = 
                    sqrt( del_x_tx*del_x_tx + del_y_tx*del_y_tx + del_z_tx*del_z_tx) +
                    sqrt( del_x_rx*del_x_rx + del_y_rx*del_y_rx + del_z_rx*del_z_rx);
           
        
            for(int jj=0;jj< Nf;jj++)
            {
                double f_jj = f_start + jj*df;
                double wij = w2d[jj + ii*Nf];
                //int kk = ii*length_t +jj;
                ////kk=f_jj*N_t + t_ii
                int kk = jj*length_t +ii;
                        
                temp = temp + 
                      cuComplex(PhaseHist_real[kk],PhaseHist_imag[kk])*
                      cuComplex(wij,0.0)*exp( cuComplex(0.0,2.0*pi*f_jj*bi_range/c_sol));
           
            }
            
        }
        // SAR_real[tid] = temp.r/(length_t*Nf);
        // SAR_imag[tid] = temp.i/(length_t*Nf);

        SAR_real[tid] = temp.r;
        SAR_imag[tid] = temp.i;
        tid += blockDim.x*gridDim.x;

    }

}