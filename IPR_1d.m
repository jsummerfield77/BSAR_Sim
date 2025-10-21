% IPR_1d.m
% Author: John Summerfield
%
% Description:
% This MATLAB function computes a 1-D impulse response (IPR) for a bistatic SAR 
% system using a stepped-frequency waveform. The IPR is the Fourier-domain 
% representation of the system’s point target response and is a fundamental tool 
% for analyzing system resolution, ambiguity structure, and passband shaping. 
% This 1-D version computes the IPR along a user-defined unit vector direction 
% (û), enabling high-resolution measurement of impulse response metrics such as 
% mainlobe width, peak sidelobe ratio, and integrated sidelobe ratio.
%
% Geometry and Implementation:
% The IPR is evaluated as a function of scalar displacement r along a unit vector û. 
% For each displacement value, the phase response is computed by projecting the 
% bistatic range gradient and its time derivatives onto û and accumulating the 
% corresponding complex response across frequency.
%
% Model Update (MSM → CFV):
% This script is an updated version of the IPR computation originally developed 
% during my Ph.D. dissertation research. The original formulation used the 
% conventional move–stop–move (MSM) model, which assumed the platform was 
% stationary during each pulse and modeled the phasor using only the bistatic 
% range gradient ∇R:
%
%     φ_MSM ∝ (2π / c) · f · ( ∇R · û ) · r
%
% The current implementation replaces MSM with a **constant fast-time velocity (CFV)** 
% formulation, which captures continuous platform motion during the pulse and 
% incorporates higher-order Doppler effects. In the CFV model, the phasor depends 
% not only on the bistatic range gradient ∇R but also on the bistatic range rate 
% gradient ∇Ṙ and the bistatic range R₀ to the scene center:
%
%     φ_CFV ∝ (2π / c) · f · [ ( ∇R + (2 R₀ / c) ∇Ṙ ) · û ] · r
%
% where:
%   • ∇R   – bistatic range gradient vector
%   • ∇Ṙ   – bistatic range rate gradient vector
%   • R₀   – bistatic range to the scene center
%   • û    – unit vector along which the IPR is evaluated
%   • r     – scalar displacement along û
%
% This inclusion of the ∇Ṙ term introduces Doppler-coupled spatial dependence 
% into the IPR, accurately modeling time-scaling effects caused by continuous 
% platform motion. As a result, the computed impulse response remains valid for 
% high-speed radar platforms (e.g., orbital or hypersonic) where MSM assumptions fail.
%
% Implementation Notes:
% - Inputs include frequency mesh, bistatic range gradient (RGrad), bistatic range 
%   rate gradient (RRGrad), bistatic range to scene center (RO), unit vector û, 
%   and a set of displacements r.
% - The function uses a parallel loop over r for efficient evaluation.
% - The output is the complex 1-D impulse response sampled along û.
%
% Usage:
% This 1-D IPR function is particularly useful for system performance assessment, 
% resolution analysis, and transfer function validation. By isolating the impulse 
% response along a single direction, it provides a clear view of how frequency 
% weighting, passband shaping, and Doppler coupling influence SAR image formation.

function ipr =  IPR_1d(weight,RGrad,RRGrad,RO,Freq_mesh,u_vec,r),
% nt = size(RGrad,2);
nf = size(Freq_mesh,1);
% nt = size(Freq_mesh,2);
n_r = length(r);
c_sol = 299792458;
ipr = zeros(size(r));
parfor ii=1:n_r,
    r_ii = r(ii);    
    % RGrad_dot_r = RGrad(1,:)*u_vec(1) + RGrad(2,:)*u_vec(2) + RGrad(3,:)*u_vec(3);
    F_x = (2*pi/c_sol)*Freq_mesh.*( ones(nf,1)*( RGrad(1,:)+ 2*RO.*RRGrad(1,:)/c_sol));
    F_y = (2*pi/c_sol)*Freq_mesh.*( ones(nf,1)*( RGrad(2,:)+ 2*RO.*RRGrad(2,:)/c_sol));
    F_z = (2*pi/c_sol)*Freq_mesh.*( ones(nf,1)*( RGrad(3,:)+ 2*RO.*RRGrad(3,:)/c_sol));

    phase = r_ii*(F_x*u_vec(1) + F_y*u_vec(2) + F_z*u_vec(3));    
    ipr(ii) = sum(weight(:).*exp(j*phase(:)));
end
