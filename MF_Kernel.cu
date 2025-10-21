// MF_Kernel.cu
// Author: John Summerfield
//
// Description:
// CUDA kernel implementing matched-filter SAR image formation for a bistatic SAR system. 
// This kernel performs the backprojection (or correlation) of the measured phase history 
// data against reference signals computed from hypothesized scene locations to form a 
// focused SAR image. The computation is fully GPU-parallelized over scene pixels.
//
// Model Update (Dissertation Lineage → CFV):
// This kernel is an updated version of the matched-filter image formation code 
// originally developed during my Ph.D. dissertation research. The previous version 
// used the conventional move–stop–move (MSM) model, which assumes the radar platforms 
// are stationary during each pulse. In that formulation, the matched-filter phasor 
// depends only on the stepped frequency and bistatic range:
//
//     φ_MSM ∝  +2π f · [ R_TX(t, r) + R_RX(t, r) ] / c
//
// This version replaces MSM with a **Constant Fast-time Velocity (CFV)** model, which 
// accounts for continuous platform motion during pulse transmission and reception. The 
// CFV formulation introduces Doppler-time scaling via the factor η(t, r), enabling 
// physically accurate matched filtering even for high-velocity platforms such as 
// orbital or hypersonic systems.
//
// Matched Filter Formulation (CFV):
// Under the CFV model, the matched-filter reference signal for a hypothesized 
// pixel location r is:
//
//     h(f, t; r) = exp[ +j (2π f / c) · η(t, r) · R(t, r) ]
//
// where:
//   • R(t, r)   = bistatic range from transmitter → r → receiver
//   • Ṙ(t, r)   = bistatic range rate from platform motion and geometry
//   • η(t, r)   = Doppler time-scaling factor = (c + Ṙ(t, r)) / (c - Ṙ(t, r))
//   • f         = stepped transmit frequency
//   • c         = speed of light
//
// The matched-filter image pixel value is then computed by correlating the measured 
// phase history signal s(f, t) with this reference function:
//
//     I(r) = ∑_{f,t} s(f, t) · h*(f, t; r)
//
// The inclusion of η(t, r) · R(t, r) in the phasor accounts for Doppler-induced 
// time dilation and ensures coherent integration across all frequencies and slow-time 
// samples in the presence of continuous motion.
//
// Implementation Notes:
// - Each GPU thread computes the complex image pixel value for one spatial location 
//   (X[tid], Y[tid], Z[tid]).
// - For each slow-time index ii and frequency index jj:
//     – Compute bistatic range R_TX + R_RX
//     – Compute bistatic range rate Ṙ(t, r) from platform velocities and geometry
//     – Compute Doppler time-scaling factor η(t, r)
//     – Form matched-filter phasor with η-weighted bistatic range
// - Accumulate complex contributions from all frequency bins and slow-time samples.
// - The resulting SAR_real[tid] and SAR_imag[tid] arrays store the focused SAR image.
//
// High-Speed Accuracy:
// The inclusion of η(t, r) ensures that the matched filter compensates for Doppler-time 
// scaling effects caused by continuous platform motion, maintaining coherence and 
// image focus even for platforms with velocities on the order of hundreds of km/s.
//
// Inputs/Outputs (abridged):
//   Inputs: Phase history data (real/imag), scene coordinates (X,Y,Z), platform 
//           positions/velocities vs. slow time, center frequencies f_c(ii), 
//           bandwidths BW(ii), weighting function w2d
//   Outputs: SAR_real[tid], SAR_imag[tid]  // complex matched-filter SAR image

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
const double *Vel_TX_x,const double *Vel_TX_y,const double *Vel_TX_z,
const double *Vel_RX_x,const double *Vel_RX_y,const double *Vel_RX_z,
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
            double R_TX = sqrt(del_x_tx*del_x_tx + del_y_tx*del_y_tx  + del_z_tx*del_z_tx);


            double del_x_rx = Pos_RX_x[ii]-X[tid];
            double del_y_rx = Pos_RX_y[ii]-Y[tid];
            double del_z_rx = Pos_RX_z[ii]-Z[tid];
            double R_RX = sqrt(del_x_rx*del_x_rx + del_y_rx*del_y_rx + del_z_rx*del_z_rx);
 
            double bi_range = R_TX + R_RX;

            double bi_range_rate =
                (Vel_TX_x[ii]*del_x_tx + Vel_TX_y[ii]*del_y_tx + Vel_TX_z[ii]*del_z_tx)/R_TX +
                (Vel_RX_x[ii]*del_x_rx + Vel_RX_y[ii]*del_y_rx + Vel_RX_z[ii]*del_z_rx)/R_RX;
           
        
            double eta_ii = (c_sol+bi_range_rate)/(c_sol-bi_range_rate);

            for(int jj=0;jj< Nf;jj++)
            {
                double f_jj = f_start + jj*df;
                double wij = w2d[jj + ii*Nf];
                //int kk = ii*length_t +jj;
                ////kk=f_jj*N_t + t_ii
                int kk = jj*length_t +ii;
                        
                // temp = temp + 
                //       cuComplex(PhaseHist_real[kk],PhaseHist_imag[kk])*
                //       cuComplex(wij,0.0)*exp( cuComplex(0.0,2.0*pi*f_jj*bi_range/c_sol));
           
                temp = temp + 
                      cuComplex(PhaseHist_real[kk],PhaseHist_imag[kk])*
                      cuComplex(wij,0.0)*exp( cuComplex(0.0,2.0*pi*f_jj*eta_ii*bi_range/c_sol));
            }
            
        }
        // SAR_real[tid] = temp.r/(length_t*Nf);
        // SAR_imag[tid] = temp.i/(length_t*Nf);

        SAR_real[tid] = temp.r;
        SAR_imag[tid] = temp.i;
        tid += blockDim.x*gridDim.x;
    }
}
