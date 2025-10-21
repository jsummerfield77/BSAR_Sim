// PSF_Kernel.cu
// Author: John Summerfield
//
// Description:
// CUDA kernel that computes the point spread function (PSF) for a bistatic SAR
// system using a stepped-frequency waveform. The computation is fully
// GPU-parallelized across spatial sample points and slow-time / frequency indices.
//
// Model Update (Dissertation Lineage → CFV):
// This kernel is an updated version of the PSF computation originally developed
// for my Ph.D. dissertation work. The former implementation used the
// move–stop–move (MSM) approximation (platform assumed stationary within a pulse).
// The current implementation replaces MSM with a Constant Fast-time Velocity (CFV)
// model that preserves continuous platform motion during the pulse and correctly
// captures Doppler-time scaling. This upgrade is important for high-speed
// platforms (e.g., orbital/hypersonic) where MSM breaks down.
//
// Phasor Form (MSM → CFV):
//   • MSM phasor (old): depends on frequency f and the *difference* in bistatic
//     range signatures between two scene points r and r':
//         φ_MSM ∝ f · [ R(t, r) − R(t, r′) ]
//
//   • CFV phasor (this code): depends on frequency f and the *η-weighted*
//     difference in bistatic range signatures:
//         φ_CFV ∝ f · [ η(t, r) · R(t, r) − η(t, r′) · R(t, r′) ]
//
//     where the time-scale factor η(t, r) is the Doppler-time scaling arising
//     from constant fast-time velocity:
//         η(t, r) = (c + Ṙ(t, r)) / (c − Ṙ(t, r))
//     with c the speed of light and Ṙ(t, r) the bistatic range rate at point r.
//     This weighting accurately models time-warping of the return induced by
//     platform motion and remains valid in high-speed regimes.
//
// Geometry & Implementation Notes:
//   • The kernel evaluates two nearby scene points r = r0 − ½Δr and r′ = r0 + ½Δr
//     about a midpoint r0 = (Pos_mid_x, Pos_mid_y, Pos_mid_z). For each slow-time
//     index ii, it computes:
//       – Bistatic ranges:           R(t, r), R(t, r′)
//       – Bistatic range rates:      Ṙ(t, r), Ṙ(t, r′)   (via TX/RX velocities)
//       – Time-scale factors:        η(t, r), η(t, r′)
//       – Phase:                     f · [ ηR − η′R′ ]
//   • Frequency agility / weighting enters via the stepped-frequency grid
//     f_jj = f_start + jj·Δf and a 2-D weight w2d(jj, ii).
//   • The accumulated complex sum over (ii, jj) yields the PSF at each Δr sample.
//
// High-Speed Accuracy:
// The CFV phasor retains the first-order coupling between range and range-rate
// through η(t, r), ensuring the PSF properly reflects motion-induced time
// scaling and Doppler evolution at high platform speeds.
//
// Inputs/Outputs (abridged):
//   Inputs: positions/velocities of TX/RX vs. slow time, scene offsets (X,Y,Z),
//           midpoint r0, center frequencies f_c(ii), bandwidths BW(ii), weights w2d
//   Outputs: PSF_real[k], PSF_imag[k]  // complex PSF samples per spatial index
//
// Practical Notes:
//   • Guard small denominators when computing range rates / ranges if needed.
//   • η → ∞ as |Ṙ| → c; in realistic SAR |Ṙ| ≪ c, but checks can be added for safety.
//   • The code assumes double precision; keep it for stability at large f·R/c.
//
// -----------------------------------------------------------------------------
// Optional inline comments to add near key lines:
//
// // bi_range, bi_range2: R(t, r) and R(t, r′) about midpoint r0 = (Pos_mid_*)
// // bi_range_rate, bi_range_rate2: Ṙ(t, r) and Ṙ(t, r′) from TX/RX velocities
// // eta, eta2: Doppler-time scaling factors η(t, r) and η(t, r′)
// // phase term implements f * [ ηR − η′R′ ] under the CFV model
//
// // temp2 += w * exp( j * (2π/c) * f * [ ηR − η′R′ ] )

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
const double *Vel_TX_x,const double *Vel_TX_y,const double *Vel_TX_z,
const double *Vel_RX_x,const double *Vel_RX_y,const double *Vel_RX_z,
double Pos_mid_x,double Pos_mid_y,double Pos_mid_z,
const double *f_c,const double *BW,const double *w2d, int length_t, int Nf,int length_X)
{
    int tid = threadIdx.x +blockIdx.x*blockDim.x;
    while(tid<length_X)
    {
        double pi = 3.141592653589793;
        double c_sol = 299792458.0;
        double ratio = 2*pi/c_sol;
        
        cuComplex temp = cuComplex(0.0,0.0);
        for(int ii=0;ii< length_t;ii++)
        {
            double df = BW[ii]/(Nf-1);
            double f_start = f_c[ii]-.5*BW[ii];

            double del_x_tx = Pos_TX_x[ii]-.5*X[tid]+Pos_mid_x;
            double del_y_tx = Pos_TX_y[ii]-.5*Y[tid]+Pos_mid_y;
            double del_z_tx = Pos_TX_z[ii]-.5*Z[tid]+Pos_mid_z;

            double Range_tx = sqrt(del_x_tx*del_x_tx + del_y_tx*del_y_tx + del_z_tx*del_z_tx);

            double del_x_rx = Pos_RX_x[ii]-.5*X[tid]+Pos_mid_x;
            double del_y_rx = Pos_RX_y[ii]-.5*Y[tid]+Pos_mid_y;
            double del_z_rx = Pos_RX_z[ii]-.5*Z[tid]+Pos_mid_z;

            double Range_rx = sqrt(del_x_rx*del_x_rx+del_y_rx*del_y_rx+ del_z_rx*del_z_rx);

            double del_x_tx2 = Pos_TX_x[ii]+.5*X[tid]+Pos_mid_x;
            double del_y_tx2 = Pos_TX_y[ii]+.5*Y[tid]+Pos_mid_y;
            double del_z_tx2 = Pos_TX_z[ii]+.5*Z[tid]+Pos_mid_z;

            double Range_tx2 = sqrt(del_x_tx2*del_x_tx2+del_y_tx2*del_y_tx2+ del_z_tx2*del_z_tx2);

            double del_x_rx2 = Pos_RX_x[ii]+.5*X[tid]+Pos_mid_x;
            double del_y_rx2 = Pos_RX_y[ii]+.5*Y[tid]+Pos_mid_y;
            double del_z_rx2 = Pos_RX_z[ii]+.5*Z[tid]+Pos_mid_z;

            double Range_rx2 = sqrt(del_x_rx2*del_x_rx2+del_y_rx2*del_y_rx2+ del_z_rx2*del_z_rx2);

            double bi_range = sqrt( del_x_tx*del_x_tx + del_y_tx*del_y_tx + del_z_tx*del_z_tx) + sqrt( del_x_rx*del_x_rx + del_y_rx*del_y_rx + del_z_rx*del_z_rx);
            double bi_range2 = sqrt( del_x_tx2*del_x_tx2 + del_y_tx2*del_y_tx2 + del_z_tx2*del_z_tx2) + sqrt( del_x_rx2*del_x_rx2 + del_y_rx2*del_y_rx2 + del_z_rx2*del_z_rx2);
            //double range_0 = sqrt(Pos_TX_x[ii]*Pos_TX_x[ii] + Pos_TX_y[ii]*Pos_TX_y[ii]+ Pos_TX_z[ii]*Pos_TX_z[ii]) + sqrt(Pos_RX_x[ii]*Pos_RX_x[ii] + Pos_RX_y[ii]*Pos_RX_y[ii] + Pos_RX_z[ii]*Pos_RX_z[ii]);

            double bi_range_rate = 
                (Vel_TX_x[ii]*del_x_tx + Vel_TX_y[ii]*del_y_tx + Vel_TX_z[ii]*del_z_tx)/Range_tx +
                (Vel_RX_x[ii]*del_x_rx + Vel_RX_y[ii]*del_y_rx + Vel_RX_z[ii]*del_z_rx)/Range_rx;

            double bi_range_rate2 = 
                (Vel_TX_x[ii]*del_x_tx2 + Vel_TX_y[ii]*del_y_tx2 + Vel_TX_z[ii]*del_z_tx2)/Range_tx2 +
                (Vel_RX_x[ii]*del_x_rx2 + Vel_RX_y[ii]*del_y_rx2 + Vel_RX_z[ii]*del_z_rx2)/Range_rx2;
            
            double eta = (c_sol+bi_range_rate)/(c_sol-bi_range_rate);
            double eta2 = (c_sol+bi_range_rate2)/(c_sol-bi_range_rate2);

            cuComplex temp2 = cuComplex(0.0,0.0);
        
            for(int jj=0;jj< Nf;jj++)
            {
                double f_jj = f_start + jj*df;
                double wij = w2d[jj + ii*Nf];
                temp2 = temp2 + cuComplex(wij,0.0)*exp( cuComplex(0.0,ratio*f_jj*(eta*bi_range-eta2*bi_range2)));            
            }
            // temp = temp + cuComplex(df,0.0)*temp2;
            temp = temp + temp2;
            
        }
        PSF_real[tid] = temp.r;
        PSF_imag[tid] = temp.i;
        tid += blockDim.x*gridDim.x;

    }
}
