% PSF_1d.m
% Author: John Summerfield
%
% Description:
% This MATLAB function computes a 1-D point spread function (PSF) for a bistatic 
% SAR system using a stepped-frequency waveform. It is designed for analysis and 
% visualization of PSF properties and for quantitative measurement of image quality 
% metrics (e.g., resolution, peak sidelobe ratio, integrated sidelobe ratio) 
% directly in MATLAB.
%
% Geometry Model:
% The PSF is computed between two points symmetrically displaced about a midpoint:
%
%   r₁ = r_mid + (r / 2) · û
%   r₂ = r_mid − (r / 2) · û
%
% where:
%   • û      – unit vector defining the PSF measurement direction  
%   • r_mid   – midpoint between the two evaluation points  
%   • r       – scalar offset distance between the points  
%
% This formulation allows direct computation of the PSF along any chosen axis and 
% facilitates 1-D analysis of SAR system impulse response behavior.
%
% Model Update (MSM → CFV):
% This script is an updated version of the PSF computation originally developed 
% during my Ph.D. dissertation research. The previous implementation used the 
% conventional move–stop–move (MSM) approximation, in which the platform was 
% assumed stationary during each pulse and the PSF phasor depended only on 
% the bistatic range difference:
%
%     φ_MSM ∝ (2π / c) · f · [ R₁ − R₂ ]
%
% In this updated version, the model has been extended to a **constant fast-time 
% velocity (CFV)** formulation, which accounts for continuous platform motion 
% during pulse transmission and reception. The phasor now includes Doppler-induced 
% time-scaling effects through the η-weighted bistatic ranges:
%
%     φ_CFV ∝ (2π / c) · f · [ η₁ · R₁ − η₂ · R₂ ]
%
% where:
%   • R₁, R₂         – bistatic ranges from transmitter → r₁ / r₂ → receiver  
%   • Ṙ₁, Ṙ₂         – bistatic range rates for r₁ and r₂  
%   • η₁, η₂         – Doppler time-scaling factors:
%
%         η = (c + Ṙ) / (c − Ṙ)
%
% The inclusion of η ensures that Doppler-time scaling and continuous motion 
% effects are properly modeled, improving accuracy for high-velocity platforms 
% such as orbital or hypersonic sensors.
%
% Implementation Notes:
% - The PSF is computed over a vector of spatial separations r, using a parallel 
%   loop for efficiency.
% - Inputs include platform positions and velocities, frequency mesh, direction 
%   vector û, and midpoint location.
% - Outputs are complex PSF values for each r, from which key performance metrics 
%   can be derived.
%
% Usage:
% This 1-D PSF function is useful for analyzing the system’s impulse response, 
% measuring resolution and sidelobe levels, and validating CFV-based SAR models 
% against analytical predictions or simulation results.

function psf =  PSF_1d(weight,Pos_TX,Pos_RX,Vel_TX,Vel_RX,Freq_mesh,v_vec,r_mid,r),
u_vec = ones(3,1);
u_vec(:) = v_vec(:);
% nt = size(Pos_TX,2);
nf = size(Freq_mesh,1);
nt = size(Freq_mesh,2);
n_r = length(r);
c_sol = 299792458;
psf = zeros(size(r));
parfor ii=1:n_r,
    r_1 = r_mid + .5*r(ii)*u_vec;
    r_2 = r_mid - .5*r(ii)*u_vec;

    del_Pos_TX_1 = Pos_TX-r_1;
    del_Pos_RX_1 = Pos_RX-r_1;
    del_Pos_TX_2 = Pos_TX-r_2;
    del_Pos_RX_2 = Pos_RX-r_2;

    R_TX_1 = sqrt(dot(del_Pos_TX_1,del_Pos_TX_1,1));
    R_RX_1 = sqrt(dot(del_Pos_RX_1,del_Pos_RX_1,1));
    R_TX_2 = sqrt(dot(del_Pos_TX_2,del_Pos_TX_2,1));
    R_RX_2 = sqrt(dot(del_Pos_RX_2,del_Pos_RX_2,1));
    
    R_1 = R_TX_1 + R_RX_1;
    R_2 = R_TX_2 + R_RX_2;

    RR_1 = dot(del_Pos_TX_1,Vel_TX,1)./R_TX_1 + dot(del_Pos_RX_1,Vel_RX,1)./R_RX_1;
    RR_2 = dot(del_Pos_TX_2,Vel_TX,1)./R_TX_2 + dot(del_Pos_RX_2,Vel_RX,1)./R_RX_2;
    
    eta_1 = (c_sol + RR_1)./(c_sol - RR_1);
    eta_2 = (c_sol + RR_2)./(c_sol - RR_2);

    % phase = (2*pi/c_sol)*f_n.'*(R_1-R_2);   
    % phase = (2*pi/c_sol)*Freq_mesh.*(ones(nf,1)*(R_1-R_2))
    phase = (2*pi/c_sol)*Freq_mesh.*(ones(nf,1)*(eta_1.*R_1-eta_2.*R_2));
    psf(ii) = sum(weight(:).*exp(j*phase(:)));
end
