// PhaseHist_Kernel.cu
// Author: John Summerfield
//
// Description:
// CUDA kernel that computes the raw bistatic SAR phase history (PH) data for a 
// scene of discrete point targets. This kernel is part of the end-to-end 
// GPU-accelerated SAR simulation framework and is designed to generate the 
// frequency–slow-time domain signal used for matched filtering and image formation.
//
// Model Update (Dissertation Lineage → CFV):
// This kernel is an updated version of the phase history computation originally 
// developed for my Ph.D. dissertation work. The previous version used the 
// conventional move–stop–move (MSM) approximation, in which the radar platform 
// was assumed stationary during each pulse and the phasor was modeled as a 
// function of frequency and bistatic range:
//
//     φ_MSM ∝ -2π f · [ R_TX(t, r) + R_RX(t, r) ] / c
//
// The current version replaces the MSM assumption with a **Constant Fast-time 
// Velocity (CFV)** model, which preserves continuous platform motion during the 
// pulse and accounts for Doppler-induced time scaling. This update is essential 
// for accurately modeling high-speed radar platforms (e.g., orbital or hypersonic 
// systems) where MSM assumptions fail.
//
// Received Signal Model:
// Under the CFV formulation, the received signal is modeled as a superposition 
// of scattered returns from all point targets in the scene:
//
//     s(f, t) = ∫_r ρ(r) · exp[ -j (2π f / c) · η(t, r) · R(t, r) ] d r
//
// where:
//   • ρ(r)             = target reflectivity at position r
//   • R(t, r)          = bistatic range from transmitter → r → receiver
//   • Ṙ(t, r)          = bistatic range rate
//   • η(t, r)          = Doppler time-scaling factor = (c + Ṙ(t, r)) / (c - Ṙ(t, r))
//   • f               = transmit frequency
//   • c               = speed of light
//
// Phasor Form (MSM → CFV):
// - MSM phasor: depends only on frequency and bistatic range R(t, r)
// - CFV phasor (this code): depends on frequency and the **η-weighted** bistatic range:
//
//       φ_CFV = -2π f · [ η(t, r) · R(t, r) ] / c
//
// This η-weighted range captures time dilation and Doppler evolution due to 
// continuous platform motion and remains accurate even at high platform speeds.
//
// Implementation Notes:
// - The kernel iterates over all frequency samples (jj) and slow-time samples (ii).
// - For each point target kk, it computes:
//     – R_TX, R_RX: bistatic range components
//     – Ṙ(t, r): bistatic range rate from platform velocity and LOS projection
//     – η(t, r): Doppler time-scaling factor
// - Each target’s contribution is weighted by its reflectivity ρ and 2D weighting 
//   function w2d(ii, jj).
// - The total phase history signal is accumulated over all scatterers to form the 
//   received signal.
//
// High-Speed Accuracy:
// The inclusion of η(t, r) ensures that the model captures motion-induced time 
// scaling and higher-order Doppler effects, enabling physically accurate phase 
// history simulation for high-velocity platforms.
//
// Inputs/Outputs (abridged):
//   Inputs: target positions and reflectivities, TX/RX positions and velocities 
//           vs. slow time, center frequencies f_c(ii), bandwidths BW(ii), weights w2d
//   Outputs: PhaseHist_real[tid], PhaseHist_imag[tid]  // complex phase history samples

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
const double *Vel_TX_x,const double *Vel_TX_y,const double *Vel_TX_z,
const double *Vel_RX_x,const double *Vel_RX_y,const double *Vel_RX_z,
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
                double bi_range_rate_ii = 
                    (Vel_TX_x[ii]*diff_tx_x + Vel_TX_y[ii]*diff_tx_y + Vel_TX_z[ii]*diff_tx_z)/R_TX +
                    (Vel_RX_x[ii]*diff_rx_x + Vel_RX_y[ii]*diff_rx_y + Vel_RX_z[ii]*diff_rx_z)/R_RX;

                double eta_ii = (c_sol+bi_range_rate_ii)/(c_sol-bi_range_rate_ii);
            
                cuComplex amplitude = cuComplex(tgt_rho[kk]*wij,0.0);
                // double phase = -2.0*pi*freq_jj*bi_range_ii/c_sol;
                double phase = -2.0*pi*freq_jj*eta_ii*bi_range_ii/c_sol;

                temp = temp + amplitude*exp( cuComplex(0.0,phase));
        }
 
        PhaseHist_real[tid] = temp.r;
        PhaseHist_imag[tid] = temp.i;
// 	PhaseHist_real[tid] = 1.0*ii;
// 	PhaseHist_imag[tid] = 1.0*jj;
        tid += blockDim.x*gridDim.x;
    }

}
