//PhaseHist_Kernel.cu
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

__global__ void PhaseHist_Kernel(
double *PhaseHist_real,double *PhaseHist_imag,
const double *tgt_x,const double *tgt_y,const double *tgt_z,const double *tgt_rho,
const double *Pos_TX_x,const double *Pos_TX_y,const double *Pos_TX_z,
const double *Pos_RX_x,const double *Pos_RX_y,const double *Pos_RX_z,
const double *f_c,const double *BW,
const double *w2d, int N_t, int N_f, int N_tgt)
{
    double pi = 3.141592653589793;
    double c_sol = 299792458.0;

    int tid = threadIdx.x +blockIdx.x*blockDim.x;
    while(tid < N_t*N_f )
    {
        //tid=f_jj*N_t + t_ii        
        int ii,jj;
        ii = tid%N_t;
        jj = tid/N_t;
        double wij = w2d[jj + ii*N_f];
         double freq_jj = f_c[ii]+BW[ii]*((1.0*jj)/(N_f-1.0)-.5);
        cuComplex temp = cuComplex(0.0,0.0);
        for(int kk=0;kk< N_tgt;kk++)
        {
                double diff_tx_x = Pos_TX_x[ii]-tgt_x[kk];
                double diff_tx_y = Pos_TX_y[ii]-tgt_y[kk];
                double diff_tx_z = Pos_TX_z[ii]-tgt_z[kk];
                double R_TX = sqrt(diff_tx_x*diff_tx_x + diff_tx_y*diff_tx_y  + diff_tx_z*diff_tx_z);

                double diff_rx_x = Pos_RX_x[ii]-tgt_x[kk];
                double diff_rx_y = Pos_RX_y[ii]-tgt_y[kk];
                double diff_rx_z = Pos_RX_z[ii]-tgt_z[kk];
                double R_RX = sqrt(diff_rx_x*diff_rx_x + diff_rx_y*diff_rx_y + diff_rx_z*diff_rx_z);
            
                double bi_range_ii = R_TX + R_RX;
                cuComplex amplitude = cuComplex(tgt_rho[kk]*wij,0.0);
                double phase = -2.0*pi*freq_jj*bi_range_ii/c_sol;

                temp = temp + amplitude*exp( cuComplex(0.0,phase));
        }
 
        PhaseHist_real[tid] = temp.r;
        PhaseHist_imag[tid] = temp.i;
// 	PhaseHist_real[tid] = 1.0*ii;
// 	PhaseHist_imag[tid] = 1.0*jj;
        tid += blockDim.x*gridDim.x;
    }

}