% Sim_Main.m
% Author: John Summerfield
%
% Description:
% This script is the main driver for the bistatic SAR simulation framework. 
% It represents a major update to the simulation originally developed as part 
% of my Ph.D. dissertation research at UMass Dartmouth. The current version 
% modernizes the underlying physics assumptions from a traditional 
% move–stop–move model to a constant fast-time velocity model, enabling 
% accurate modeling of continuous platform motion and Doppler effects, 
% especially in high-speed scenarios.
%
% Physics Model Update:
% - Previous versions assumed the radar platform stopped during each pulse 
%   (move–stop–move), an approximation valid only for relatively low-speed 
%   systems.
% - The updated model assumes constant platform velocity during fast time. 
%   At any given moment in slow time, the scattering signal from every point 
%   target in the scene is modeled as:
%       • Doppler-time scaled by: η = (c + RR) / (c - RR)
%       • Fast-time delayed by:   delay = Bistatic_Range / c
%   where RR is the bistatic range rate and Bistatic_Range is the instantaneous 
%   bistatic distance from transmitter to target to receiver. This continuous 
%   motion model captures Doppler evolution accurately and is essential for 
%   high-velocity platforms.
%
% Platform and Geometry Modeling:
% - Generates closed-form expressions for platform position and velocity vectors.
% - Computes line-of-sight (LOS) unit vectors from both transmitter and receiver 
%   platforms to the scene center as functions of slow time.
% - Derives time derivatives of LOS vectors to obtain closed-form expressions of:
%     • Bistatic range gradient vector
%     • Range rate gradient vector
%     • Range rate rate gradient vector
%
% Sampling and Resolution Design:
% - Establishes Nyquist sampling requirements based on scene size, slow-time 
%   aperture duration, and waveform bandwidth.
% - Determines slow-time sampling intervals and frequency-domain resolution to 
%   meet specified spatial resolution goals.
%
% K-Space Transfer Function Synthesis:
% - Constructs Fourier-domain (K-space) representations of the SAR transfer 
%   function (TF).
% - Designs agile waveform frequency parameters to reshape the TF passband 
%   support region in K-space for optimized imaging performance.
% - Implements frequency-dependent weighting functions to tailor system response 
%   characteristics.
%
% Waveform and Frequency Agility:
% - Uses a stepped-frequency waveform and varies the frequency sweep strategy to 
%   achieve frequency agility and passband control.
%
% Simulation and Image Formation:
% - Simulates raw SAR data (phase history) from a synthetic scene containing 
%   point targets, incorporating the full bistatic geometry and Doppler-time 
%   scaling.
% - Generates point spread functions (PSFs) and impulse responses (IPRs) from 
%   the simulated data.
% - Forms matched-filtered SAR images directly from GPU-accelerated phase 
%   history data using CUDA.
%
% Performance Metrics:
% - Computes impulse response (IPR) metrics, including:
%     • Spatial resolution
%     • Peak sidelobe ratio (PSLR)
%     • Integrated sidelobe ratio (ISLR)
%
% Notes:
% This script is optimized for execution on NVIDIA GPUs to enable large-scale 
% simulations of high-speed bistatic SAR scenarios with agile waveform control. 
% The combination of closed-form modeling, K-space shaping, and GPU acceleration 
% enables rapid exploration of waveform strategies and their impact on SAR image 
% quality and ambiguity performance.
function Sim_Main(dataFileName, parms)

device = gpuDevice(1);    
reset(device);
c_sol = physconst('lightspeed');

%Center freq
fc = parms.center_freq_Hz;
%Center wavelength
lambda_c = c_sol/fc;

%Initial LOS AZ  
initial_LOS_az_ang_rad_TX =  .5*parms.bearing_diff_deg*pi/180;
initial_LOS_az_ang_rad_RX = -.5*parms.bearing_diff_deg*pi/180;


%Initial LOS vectors
LOS_TX_0 = [cos(initial_LOS_az_ang_rad_TX)*cos(-parms.TX_el_ang_deg*pi/180);...
            sin(initial_LOS_az_ang_rad_TX)*cos(-parms.TX_el_ang_deg*pi/180);...
            sin(-parms.TX_el_ang_deg*pi/180)];

LOS_RX_0 = [cos(initial_LOS_az_ang_rad_RX)*cos(-parms.RX_el_ang_deg*pi/180);...
            sin(initial_LOS_az_ang_rad_RX)*cos(-parms.RX_el_ang_deg*pi/180);...
            sin(-parms.RX_el_ang_deg*pi/180)];

%Bistatic Range Grad at t=0
R_Grad_0 = LOS_TX_0 + LOS_RX_0;
%Bistatic angle at t=0
bi_ang_0 = 2*acos(.5*sqrt(dot(R_Grad_0,R_Grad_0)));
%Range gradient projected into ground plane at t=0
R_Grad_0_GP = cross([0;0;1],cross(R_Grad_0,[0;0;1]));
R_Grad_0_GP_mag = sqrt(dot(R_Grad_0_GP,R_Grad_0_GP));

%I'm going to use the range gradient at t=0 and the desired resolution to
%set the waveform bandwidth.  I'm going to add 10% margin

% BW_Kr_0 = 2*pi*BW*R_Grad_0_GP_mag/c_sol
% res_r = 2*pi/BW_Kr_0 -> this is peak to null res of sinc function
%        = c_sol/(BW*R_Grad_0_GP_mag)

BW = 1.1*c_sol/(parms.Cmd_Range_Res_m_GP*R_Grad_0_GP_mag);

%I'm going to use the range gradient at t=0 and the desired X-range resolution to
%set the rotation angle.  I'm going to add 10% margin agina

% BW_Kaz_0 = int_Ang_rad*2*pi*fc*R_Grad_0_GP_mag/c_sol
% res_az = 2*pi/BW_Kaz_0 -> this assumes const length range grad
%        = c_sol/(int_Ang_rad*fc*R_Grad_0_GP_mag) 

int_Ang_rad = 1.1*lambda_c/(parms.Cmd_XRange_Res_m_GP*R_Grad_0_GP_mag);

%Initial Positions
POS_TX_0 = -parms.TX_range_m*LOS_TX_0;
POS_RX_0 = -parms.RX_range_m*LOS_RX_0;

%I'm going to generate flight paths that go beyond the segment when the
%SAR image is being formed.  This will help the viewer to see the geometry 
%better when makeing plots.  The problem, the log-spiral and direct path 
%geometries have some time limitations (Range can not be allowed to go to 0). 
TX_time_limit_flag = 0;
RX_time_limit_flag = 0;

%TX Geometry
switch parms.TX_path_type

    case 0, %const vel        

        %Find Inital velocity Heading
        %This needs to get fixed so that we insure that the squint angle rotates in the correct direction, we are lossing some sign information with this version        
        
        if(sign(parms.TX_squint_ang_deg)==1),
            theta_head_TX = fminbnd(@(theta) abs(dot([cos(theta);sin(theta);0],LOS_TX_0)-cos(parms.TX_squint_ang_deg*pi/180)),initial_LOS_az_ang_rad_TX-pi,initial_LOS_az_ang_rad_TX);
        else
            theta_head_TX = fminbnd(@(theta) abs(dot([cos(theta);sin(theta);0],LOS_TX_0)-cos(parms.TX_squint_ang_deg*pi/180)),initial_LOS_az_ang_rad_TX,initial_LOS_az_ang_rad_TX+pi);            
        end
        

        %Inital velocity Vectors
        Vel_TX_0 = parms.TX_speed_mps*[cos(theta_head_TX);sin(theta_head_TX);0];
   
        
        POS_fun_TX_x = @(tau) POS_TX_0(1)+Vel_TX_0(1)*tau;
        POS_fun_TX_y = @(tau) POS_TX_0(2)+Vel_TX_0(2)*tau;
        POS_fun_TX_z = @(tau) POS_TX_0(3)+Vel_TX_0(3)*tau;
        POS_fun_TX = @(tau) [POS_fun_TX_x(tau);POS_fun_TX_y(tau);POS_fun_TX_z(tau)];

        %Method 1
        Range_fun_TX = @(tau) sqrt(dot( POS_fun_TX(tau),POS_fun_TX(tau),1));
        %Method 2
        % Range_fun_TX = @(tau) vecnorm(POS_fun_TX(tau), 2, 1);
        %Method 3        
        % Range_fun_TX = @(tau) hypot(hypot(POS_fun_TX_x(tau),POS_fun_TX_y(tau)), POS_fun_TX_z(tau));


        Vel_fun_TX_x = @(tau) Vel_TX_0(1)*ones(size(tau));
        Vel_fun_TX_y = @(tau) Vel_TX_0(2)*ones(size(tau));
        Vel_fun_TX_z = @(tau) Vel_TX_0(3)*ones(size(tau));
        Vel_fun_TX = @(tau) [Vel_fun_TX_x(tau);Vel_fun_TX_y(tau);Vel_fun_TX_z(tau)];

        
        LOS_fun_TX_x = @(tau) -POS_fun_TX_x(tau)./Range_fun_TX(tau);  
        LOS_fun_TX_y = @(tau) -POS_fun_TX_y(tau)./Range_fun_TX(tau);  
        LOS_fun_TX_z = @(tau) -POS_fun_TX_z(tau)./Range_fun_TX(tau);  
        LOS_fun_TX = @(tau) [LOS_fun_TX_x(tau);LOS_fun_TX_y(tau);LOS_fun_TX_z(tau)];

        
        theta_fun_TX = @(tau) unwrap(atan2( LOS_fun_TX_y(tau),LOS_fun_TX_x(tau) ));
        % d_theta_fun_TX = @(tau) (theta_fun_TX(tau + eps*1e4)- theta_fun_TX(tau))/(eps*1e4);
        % d_theta_fun_TX = @(tau) (theta_fun_TX(tau + eps*1e3)- theta_fun_TX(tau))/(eps*1e3);
        d_theta_fun_TX = @(tau) (theta_fun_TX(tau + eps*1e6)- theta_fun_TX(tau))/(eps*1e6);

        Vel_cross_LOS_fun_TX_x = @(tau) Vel_TX_0(2)*LOS_fun_TX_z(tau)-Vel_TX_0(3)*LOS_fun_TX_y(tau);
        Vel_cross_LOS_fun_TX_y = @(tau) Vel_TX_0(3)*LOS_fun_TX_x(tau)-Vel_TX_0(1)*LOS_fun_TX_z(tau);
        Vel_cross_LOS_fun_TX_z = @(tau) Vel_TX_0(1)*LOS_fun_TX_y(tau)-Vel_TX_0(2)*LOS_fun_TX_x(tau);

        d_LOS_fun_TX_x = @(tau) -(LOS_fun_TX_y(tau).*Vel_cross_LOS_fun_TX_z(tau)-LOS_fun_TX_z(tau).*Vel_cross_LOS_fun_TX_y(tau))./Range_fun_TX(tau);
        d_LOS_fun_TX_y = @(tau) -(LOS_fun_TX_z(tau).*Vel_cross_LOS_fun_TX_x(tau)-LOS_fun_TX_x(tau).*Vel_cross_LOS_fun_TX_z(tau))./Range_fun_TX(tau);
        d_LOS_fun_TX_z = @(tau) -(LOS_fun_TX_x(tau).*Vel_cross_LOS_fun_TX_y(tau)-LOS_fun_TX_y(tau).*Vel_cross_LOS_fun_TX_x(tau))./Range_fun_TX(tau);

        d_LOS_fun_TX = @(tau) [d_LOS_fun_TX_x(tau);d_LOS_fun_TX_y(tau);d_LOS_fun_TX_z(tau)];

        
    case 1, %circle path - const range
        %ground plane projection of range
            rho_TX = parms.TX_range_m*cos(parms.TX_el_ang_deg*pi/180);

        %So this is either +/- parms.TX_speed_mps/rho_TX
        d_theta_TX = sign(parms.TX_squint_ang_deg)*parms.TX_speed_mps/rho_TX;

        
        theta_fun_TX = @(tau) d_theta_TX*tau + initial_LOS_az_ang_rad_TX - pi; 
        d_theta_fun_TX = @(tau) d_theta_TX*ones(size(tau));


        LOS_fun_TX_x = @(tau) -cos(theta_fun_TX(tau))*cos(parms.TX_el_ang_deg*pi/180);
        LOS_fun_TX_y = @(tau) -sin(theta_fun_TX(tau))*cos(parms.TX_el_ang_deg*pi/180);
        LOS_fun_TX_z = @(tau) -ones(size(tau))*sin(parms.TX_el_ang_deg*pi/180);
        LOS_fun_TX = @(tau) [LOS_fun_TX_x(tau);LOS_fun_TX_y(tau);LOS_fun_TX_z(tau)];
        
        d_LOS_fun_TX_x = @(tau)  d_theta_TX*sin(theta_fun_TX(tau))*cos(parms.TX_el_ang_deg*pi/180);
        d_LOS_fun_TX_y = @(tau) -d_theta_TX*cos(theta_fun_TX(tau))*cos(parms.TX_el_ang_deg*pi/180);
        d_LOS_fun_TX_z = @(tau) zeros(size(tau));
        d_LOS_fun_TX = @(tau) [d_LOS_fun_TX_x(tau);d_LOS_fun_TX_y(tau);d_LOS_fun_TX_z(tau)];


        POS_fun_TX_x = @(tau) -parms.TX_range_m*LOS_fun_TX_x(tau);
        POS_fun_TX_y = @(tau) -parms.TX_range_m*LOS_fun_TX_y(tau);
        POS_fun_TX_z = @(tau) -parms.TX_range_m*LOS_fun_TX_z(tau);
        POS_fun_TX = @(tau) [POS_fun_TX_x(tau);POS_fun_TX_y(tau);POS_fun_TX_z(tau)];

        Range_fun_TX = @(tau) sqrt(dot( POS_fun_TX(tau),POS_fun_TX(tau),1));

        Vel_fun_TX_x = @(tau) -parms.TX_range_m*d_LOS_fun_TX_x(tau);
        Vel_fun_TX_y = @(tau) -parms.TX_range_m*d_LOS_fun_TX_y(tau);
        Vel_fun_TX_z = @(tau) -parms.TX_range_m*d_LOS_fun_TX_z(tau);
        Vel_fun_TX = @(tau) [Vel_fun_TX_x(tau);Vel_fun_TX_y(tau);Vel_fun_TX_z(tau)];


    case 2, %log spiral        

        %Set the flag for generating longer flight paths.
        TX_time_limit_flag = 1;

        %check squint angles
        if(abs(parms.TX_squint_ang_deg)==90),
            display('Constant Squint +/- 90 Deg, use type 1 for circle');
            error('Wrong type number');
        elseif (parms.TX_squint_ang_deg==0)||(abs(parms.TX_squint_ang_deg)==180),
            display('Constant Squint 0 Deg, use type 3 for zero squint direct');
            error('Wrong type number');
        end

        Range_Rate_TX = -parms.TX_speed_mps*cos(parms.TX_squint_ang_deg*pi/180);    
        Rho_Rate_TX = Range_Rate_TX*cos(parms.TX_el_ang_deg*pi/180);
        d_H_TX = -sign(Rho_Rate_TX)*sqrt(Range_Rate_TX^2-Rho_Rate_TX^2);  

        rho_TX_0 = parms.TX_range_m*cos(parms.TX_el_ang_deg*pi/180);
        H_TX_0 = parms.TX_range_m*sin(parms.TX_el_ang_deg*pi/180);
       
        Range_fun_TX = @(tau) parms.TX_range_m + Range_Rate_TX*tau;
        rho_fun_TX = @(tau) rho_TX_0 + Rho_Rate_TX*tau;
        H_fun_TX = @(tau) H_TX_0 + d_H_TX*tau;
        
        squint_TX_rad_GP = acos(-Rho_Rate_TX/parms.TX_speed_mps);

        theta_fun_TX = @(tau) -sign(parms.TX_squint_ang_deg)*tan(squint_TX_rad_GP)*log(rho_fun_TX(tau))+...
            sign(parms.TX_squint_ang_deg)*tan(squint_TX_rad_GP)*log(rho_fun_TX(0))+...
            initial_LOS_az_ang_rad_TX;

        d_theta_fun_TX = @(tau) -sign(parms.TX_squint_ang_deg)*tan(squint_TX_rad_GP)*Rho_Rate_TX./rho_fun_TX(tau);
    
        LOS_fun_TX_x = @(tau) cos(theta_fun_TX(tau))*cos(parms.TX_el_ang_deg*pi/180);  
        LOS_fun_TX_y = @(tau) sin(theta_fun_TX(tau))*cos(parms.TX_el_ang_deg*pi/180);  
        LOS_fun_TX_z = @(tau) -ones(size(tau))*sin(parms.TX_el_ang_deg*pi/180);  
        LOS_fun_TX = @(tau) [LOS_fun_TX_x(tau);LOS_fun_TX_y(tau);LOS_fun_TX_z(tau)];

        d_LOS_fun_TX_x = @(tau) -d_theta_fun_TX(tau).*sin(theta_fun_TX(tau))*cos(parms.TX_el_ang_deg*pi/180);  
        d_LOS_fun_TX_y = @(tau)  d_theta_fun_TX(tau).*cos(theta_fun_TX(tau))*cos(parms.TX_el_ang_deg*pi/180);  
        d_LOS_fun_TX_z = @(tau) zeros(size(tau));  
        d_LOS_fun_TX = @(tau) [d_LOS_fun_TX_x(tau);d_LOS_fun_TX_y(tau);d_LOS_fun_TX_z(tau)];

        POS_fun_TX_x = @(tau) -Range_fun_TX(tau).*LOS_fun_TX_x(tau);
        POS_fun_TX_y = @(tau) -Range_fun_TX(tau).*LOS_fun_TX_y(tau);
        POS_fun_TX_z = @(tau) -Range_fun_TX(tau).*LOS_fun_TX_z(tau);
        POS_fun_TX = @(tau) [POS_fun_TX_x(tau);POS_fun_TX_y(tau);POS_fun_TX_z(tau)];

        Vel_fun_TX_x = @(tau) -Range_Rate_TX*LOS_fun_TX_x(tau)-Range_fun_TX(tau).*d_LOS_fun_TX_x(tau);
        Vel_fun_TX_y = @(tau) -Range_Rate_TX*LOS_fun_TX_y(tau)-Range_fun_TX(tau).*d_LOS_fun_TX_y(tau);
        Vel_fun_TX_z = @(tau) -Range_Rate_TX*LOS_fun_TX_z(tau)-Range_fun_TX(tau).*d_LOS_fun_TX_z(tau);
        Vel_fun_TX = @(tau) [Vel_fun_TX_x(tau);Vel_fun_TX_y(tau);Vel_fun_TX_z(tau)];

    case 3, %zero squint Direct

        %Set the flag for generating longer flight paths.
        TX_time_limit_flag = 1;

        %Inital velocity Vectors
        if(parms.TX_squint_ang_deg==0) %Zero squint
            Vel_TX_0 = parms.TX_speed_mps*LOS_TX_0;
            Range_Rate_TX = -parms.TX_speed_mps;
        else %+/- 180 squint
            Vel_TX_0 = -parms.TX_speed_mps*LOS_TX_0;
            Range_Rate_TX = parms.TX_speed_mps;
        end

        POS_fun_TX_x = @(tau) POS_TX_0(1)+Vel_TX_0(1)*tau;
        POS_fun_TX_y = @(tau) POS_TX_0(2)+Vel_TX_0(2)*tau;
        POS_fun_TX_z = @(tau) POS_TX_0(3)+Vel_TX_0(3)*tau;
        POS_fun_TX = @(tau) [POS_fun_TX_x(tau);POS_fun_TX_y(tau);POS_fun_TX_z(tau)];

        Vel_fun_TX_x = @(tau) Vel_TX_0(1)*ones(size(tau));
        Vel_fun_TX_y = @(tau) Vel_TX_0(2)*ones(size(tau));
        Vel_fun_TX_z = @(tau) Vel_TX_0(3)*ones(size(tau));
        Vel_fun_TX = @(tau) [Vel_fun_TX_x(tau);Vel_fun_TX_y(tau);Vel_fun_TX_z(tau)];

        Range_fun_TX = @(tau) sqrt(dot( POS_fun_TX(tau),POS_fun_TX(tau),1));

        LOS_fun_TX_x = @(tau) -POS_fun_TX_x(tau)./Range_fun_TX(tau);  
        LOS_fun_TX_y = @(tau) -POS_fun_TX_y(tau)./Range_fun_TX(tau);  
        LOS_fun_TX_z = @(tau) -POS_fun_TX_z(tau)./Range_fun_TX(tau);  
        LOS_fun_TX = @(tau) [LOS_fun_TX_x(tau);LOS_fun_TX_y(tau);LOS_fun_TX_z(tau)];

        theta_fun_TX = @(tau) unwrap(atan2( LOS_fun_TX_y(tau),LOS_fun_TX_x(tau) ));
        % d_theta_fun_TX = @(tau) (theta_fun_TX(tau + eps*1e4)- theta_fun_TX(tau))/(eps*1e4);
        % d_theta_fun_TX = @(tau) (theta_fun_TX(tau + eps*1e3)- theta_fun_TX(tau))/(eps*1e3);
        d_theta_fun_TX = @(tau) (theta_fun_TX(tau + eps*1e6)- theta_fun_TX(tau))/(eps*1e6);


        Vel_cross_LOS_fun_TX_x = @(tau) Vel_TX_0(2)*LOS_fun_TX_z(tau)-Vel_TX_0(3)*LOS_fun_TX_y(tau);
        Vel_cross_LOS_fun_TX_y = @(tau) Vel_TX_0(3)*LOS_fun_TX_x(tau)-Vel_TX_0(1)*LOS_fun_TX_z(tau);
        Vel_cross_LOS_fun_TX_z = @(tau) Vel_TX_0(1)*LOS_fun_TX_y(tau)-Vel_TX_0(2)*LOS_fun_TX_x(tau);

        d_LOS_fun_TX_x = @(tau) (LOS_fun_TX_y(tau).*Vel_cross_LOS_fun_TX_z(tau)-LOS_fun_TX_z(tau).*Vel_cross_LOS_fun_TX_y(tau))./Range_fun_TX(tau);
        d_LOS_fun_TX_y = @(tau) (LOS_fun_TX_z(tau).*Vel_cross_LOS_fun_TX_x(tau)-LOS_fun_TX_x(tau).*Vel_cross_LOS_fun_TX_z(tau))./Range_fun_TX(tau);
        d_LOS_fun_TX_z = @(tau) (LOS_fun_TX_x(tau).*Vel_cross_LOS_fun_TX_y(tau)-LOS_fun_TX_y(tau).*Vel_cross_LOS_fun_TX_x(tau))./Range_fun_TX(tau);

        d_LOS_fun_TX = @(tau) [d_LOS_fun_TX_x(tau);d_LOS_fun_TX_y(tau);d_LOS_fun_TX_z(tau)];
 

    

    otherwise,
    display('Undefined Flight Path type');
    error('Undefined Flight Path type');

end


%RX Geometry
switch parms.RX_path_type

    case 0, %const vel        

        %Find Inital velocity Heading
        if(sign(parms.RX_squint_ang_deg)==1),
            theta_head_RX = fminbnd(@(theta) abs(dot([cos(theta);sin(theta);0],LOS_RX_0)-cos(parms.RX_squint_ang_deg*pi/180)),initial_LOS_az_ang_rad_RX-pi,initial_LOS_az_ang_rad_RX);
            
        else
            theta_head_RX = fminbnd(@(theta) abs(dot([cos(theta);sin(theta);0],LOS_RX_0)-cos(parms.RX_squint_ang_deg*pi/180)),initial_LOS_az_ang_rad_RX,initial_LOS_az_ang_rad_RX+pi);
        end

   
        %Inital velocity Vectors
        Vel_RX_0 = parms.RX_speed_mps*[cos(theta_head_RX);sin(theta_head_RX);0];
        
        POS_fun_RX_x = @(tau) POS_RX_0(1)+Vel_RX_0(1)*tau;
        POS_fun_RX_y = @(tau) POS_RX_0(2)+Vel_RX_0(2)*tau;
        POS_fun_RX_z = @(tau) POS_RX_0(3)+Vel_RX_0(3)*tau;
        POS_fun_RX = @(tau) [POS_fun_RX_x(tau);POS_fun_RX_y(tau);POS_fun_RX_z(tau)];

        Vel_fun_RX_x = @(tau) Vel_RX_0(1)*ones(size(tau));
        Vel_fun_RX_y = @(tau) Vel_RX_0(2)*ones(size(tau));
        Vel_fun_RX_z = @(tau) Vel_RX_0(3)*ones(size(tau));
        Vel_fun_RX = @(tau) [Vel_fun_RX_x(tau);Vel_fun_RX_y(tau);Vel_fun_RX_z(tau)];

        Range_fun_RX = @(tau) sqrt(dot( POS_fun_RX(tau),POS_fun_RX(tau),1));

        LOS_fun_RX_x = @(tau) -POS_fun_RX_x(tau)./Range_fun_RX(tau);  
        LOS_fun_RX_y = @(tau) -POS_fun_RX_y(tau)./Range_fun_RX(tau);  
        LOS_fun_RX_z = @(tau) -POS_fun_RX_z(tau)./Range_fun_RX(tau);  
        LOS_fun_RX = @(tau) [LOS_fun_RX_x(tau);LOS_fun_RX_y(tau);LOS_fun_RX_z(tau)];

        theta_fun_RX = @(tau) unwrap(atan2( LOS_fun_RX_y(tau),LOS_fun_RX_x(tau) ));
        % d_theta_fun_RX = @(tau) (theta_fun_RX(tau + eps*1e4)- theta_fun_RX(tau))/(eps*1e4);
        % d_theta_fun_RX = @(tau) (theta_fun_RX(tau + eps*1e3)- theta_fun_RX(tau))/(eps*1e3);
        d_theta_fun_RX = @(tau) (theta_fun_RX(tau + eps*1e6)- theta_fun_RX(tau))/(eps*1e6);


        Vel_cross_LOS_fun_RX_x = @(tau) Vel_RX_0(2)*LOS_fun_RX_z(tau)-Vel_RX_0(3)*LOS_fun_RX_y(tau);
        Vel_cross_LOS_fun_RX_y = @(tau) Vel_RX_0(3)*LOS_fun_RX_x(tau)-Vel_RX_0(1)*LOS_fun_RX_z(tau);
        Vel_cross_LOS_fun_RX_z = @(tau) Vel_RX_0(1)*LOS_fun_RX_y(tau)-Vel_RX_0(2)*LOS_fun_RX_x(tau);

        d_LOS_fun_RX_x = @(tau) -(LOS_fun_RX_y(tau).*Vel_cross_LOS_fun_RX_z(tau)-LOS_fun_RX_z(tau).*Vel_cross_LOS_fun_RX_y(tau))./Range_fun_RX(tau);
        d_LOS_fun_RX_y = @(tau) -(LOS_fun_RX_z(tau).*Vel_cross_LOS_fun_RX_x(tau)-LOS_fun_RX_x(tau).*Vel_cross_LOS_fun_RX_z(tau))./Range_fun_RX(tau);
        d_LOS_fun_RX_z = @(tau) -(LOS_fun_RX_x(tau).*Vel_cross_LOS_fun_RX_y(tau)-LOS_fun_RX_y(tau).*Vel_cross_LOS_fun_RX_x(tau))./Range_fun_RX(tau);

        d_LOS_fun_RX = @(tau) [d_LOS_fun_RX_x(tau);d_LOS_fun_RX_y(tau);d_LOS_fun_RX_z(tau)];


    case 1, %circle path - const range
        %ground plane projection of range
        rho_RX = parms.RX_range_m*cos(parms.RX_el_ang_deg*pi/180);

        %So this is either +/- parms.RX_speed_mps/rho_RX
        d_theta_RX = sign(parms.RX_squint_ang_deg)*parms.RX_speed_mps/rho_RX;

        theta_fun_RX = @(tau) d_theta_RX*tau + initial_LOS_az_ang_rad_RX - pi; 
        d_theta_fun_RX = @(tau) d_theta_RX*ones(size(tau));


        LOS_fun_RX_x = @(tau) -cos(theta_fun_RX(tau))*cos(parms.RX_el_ang_deg*pi/180);
        LOS_fun_RX_y = @(tau) -sin(theta_fun_RX(tau))*cos(parms.RX_el_ang_deg*pi/180);
        LOS_fun_RX_z = @(tau) -ones(size(tau))*sin(parms.RX_el_ang_deg*pi/180);
        LOS_fun_RX = @(tau) [LOS_fun_RX_x(tau);LOS_fun_RX_y(tau);LOS_fun_RX_z(tau)];
        
        d_LOS_fun_RX_x = @(tau)  d_theta_RX*sin(theta_fun_RX(tau))*cos(parms.RX_el_ang_deg*pi/180);
        d_LOS_fun_RX_y = @(tau)  -d_theta_RX*cos(theta_fun_RX(tau))*cos(parms.RX_el_ang_deg*pi/180);
        d_LOS_fun_RX_z = @(tau) zeros(size(tau));
        d_LOS_fun_RX = @(tau) [d_LOS_fun_RX_x(tau);d_LOS_fun_RX_y(tau);d_LOS_fun_RX_z(tau)];


        POS_fun_RX_x = @(tau) -parms.RX_range_m*LOS_fun_RX_x(tau);
        POS_fun_RX_y = @(tau) -parms.RX_range_m*LOS_fun_RX_y(tau);
        POS_fun_RX_z = @(tau) -parms.RX_range_m*LOS_fun_RX_z(tau);
        POS_fun_RX = @(tau) [POS_fun_RX_x(tau);POS_fun_RX_y(tau);POS_fun_RX_z(tau)];

        Range_fun_RX = @(tau) sqrt(dot( POS_fun_RX(tau),POS_fun_RX(tau),1));

        Vel_fun_RX_x = @(tau) -parms.RX_range_m*d_LOS_fun_RX_x(tau);
        Vel_fun_RX_y = @(tau) -parms.RX_range_m*d_LOS_fun_RX_y(tau);
        Vel_fun_RX_z = @(tau) -parms.RX_range_m*d_LOS_fun_RX_z(tau);
        Vel_fun_RX = @(tau) [Vel_fun_RX_x(tau);Vel_fun_RX_y(tau);Vel_fun_RX_z(tau)];


    case 2, %log-spiral - const range
        
        %Set the flag for generating longer flight paths.
        RX_time_limit_flag = 1;

        if(abs(parms.RX_squint_ang_deg)==90),
            display('Constant Squint +/- 90 Deg, use type 1 for circle');
            error('Wrong type number');
        elseif (parms.RX_squint_ang_deg==0)||(abs(parms.RX_squint_ang_deg)==180),
            display('Constant Squint 0 Deg, use type 3 for zero squint direct');
            error('Wrong type number');
        end

        Range_Rate_RX = -parms.RX_speed_mps*cos(parms.RX_squint_ang_deg*pi/180);    
        Rho_Rate_RX = Range_Rate_RX*cos(parms.RX_el_ang_deg*pi/180);
        d_H_RX = -sign(Rho_Rate_RX)*sqrt(Range_Rate_RX^2-Rho_Rate_RX^2);  

        rho_RX_0 = parms.RX_range_m*cos(parms.RX_el_ang_deg*pi/180);
        H_RX_0 = parms.RX_range_m*sin(parms.RX_el_ang_deg*pi/180);
        
        Range_fun_RX = @(tau) parms.RX_range_m + Range_Rate_RX*tau;
        rho_fun_RX = @(tau) rho_RX_0 + Rho_Rate_RX*tau;
        H_fun_RX = @(tau) H_RX_0 + d_H_RX*tau;
        
        squint_RX_rad_GP = acos(-Rho_Rate_RX/parms.RX_speed_mps);

        theta_fun_RX = @(tau) -sign(parms.RX_squint_ang_deg)*tan(squint_RX_rad_GP)*log(rho_fun_RX(tau))+...
            sign(parms.RX_squint_ang_deg)*tan(squint_RX_rad_GP)*log(rho_fun_RX(0))+...
            initial_LOS_az_ang_rad_RX;

        d_theta_fun_RX = @(tau) -sign(parms.RX_squint_ang_deg)*tan(squint_RX_rad_GP)*Rho_Rate_RX./rho_fun_RX(tau);
    
        LOS_fun_RX_x = @(tau) cos(theta_fun_RX(tau))*cos(parms.RX_el_ang_deg*pi/180);  
        LOS_fun_RX_y = @(tau) sin(theta_fun_RX(tau))*cos(parms.RX_el_ang_deg*pi/180);  
        LOS_fun_RX_z = @(tau) -ones(size(tau))*sin(parms.RX_el_ang_deg*pi/180);  
        LOS_fun_RX = @(tau) [LOS_fun_RX_x(tau);LOS_fun_RX_y(tau);LOS_fun_RX_z(tau)];

        d_LOS_fun_RX_x = @(tau) -d_theta_fun_RX(tau).*sin(theta_fun_RX(tau))*cos(parms.RX_el_ang_deg*pi/180);  
        d_LOS_fun_RX_y = @(tau)  d_theta_fun_RX(tau).*cos(theta_fun_RX(tau))*cos(parms.RX_el_ang_deg*pi/180);  
        d_LOS_fun_RX_z = @(tau) zeros(size(tau));  
        d_LOS_fun_RX = @(tau) [d_LOS_fun_RX_x(tau);d_LOS_fun_RX_y(tau);d_LOS_fun_RX_z(tau)];

        POS_fun_RX_x = @(tau) -Range_fun_RX(tau).*LOS_fun_RX_x(tau);
        POS_fun_RX_y = @(tau) -Range_fun_RX(tau).*LOS_fun_RX_y(tau);
        POS_fun_RX_z = @(tau) -Range_fun_RX(tau).*LOS_fun_RX_z(tau);
        POS_fun_RX = @(tau) [POS_fun_RX_x(tau);POS_fun_RX_y(tau);POS_fun_RX_z(tau)];

        Vel_fun_RX_x = @(tau) -Range_Rate_RX*LOS_fun_RX_x(tau)-Range_fun_RX(tau).*d_LOS_fun_RX_x(tau);
        Vel_fun_RX_y = @(tau) -Range_Rate_RX*LOS_fun_RX_y(tau)-Range_fun_RX(tau).*d_LOS_fun_RX_y(tau);
        Vel_fun_RX_z = @(tau) -Range_Rate_RX*LOS_fun_RX_z(tau)-Range_fun_RX(tau).*d_LOS_fun_RX_z(tau);
        Vel_fun_RX = @(tau) [Vel_fun_RX_x(tau);Vel_fun_RX_y(tau);Vel_fun_RX_z(tau)];

    case 3, %zero squint Direct

        %Set the flag for generating longer flight paths.
        RX_time_limit_flag = 1;

        %Inital velocity Vectors
        if(parms.RX_squint_ang_deg==0) %Zero squint
            Vel_RX_0 = parms.RX_speed_mps*LOS_RX_0;
            Range_Rate_RX = -parms.RX_speed_mps;
        else %+/- 180 squint
            Vel_RX_0 = -parms.RX_speed_mps*LOS_RX_0;
            Range_Rate_RX = parms.RX_speed_mps;
        end

       POS_fun_RX_x = @(tau) POS_RX_0(1)+Vel_RX_0(1)*tau;
        POS_fun_RX_y = @(tau) POS_RX_0(2)+Vel_RX_0(2)*tau;
        POS_fun_RX_z = @(tau) POS_RX_0(3)+Vel_RX_0(3)*tau;
        POS_fun_RX = @(tau) [POS_fun_RX_x(tau);POS_fun_RX_y(tau);POS_fun_RX_z(tau)];

        Vel_fun_RX_x = @(tau) Vel_RX_0(1)*ones(size(tau));
        Vel_fun_RX_y = @(tau) Vel_RX_0(2)*ones(size(tau));
        Vel_fun_RX_z = @(tau) Vel_RX_0(3)*ones(size(tau));
        Vel_fun_RX = @(tau) [Vel_fun_RX_x(tau);Vel_fun_RX_y(tau);Vel_fun_RX_z(tau)];

        Range_fun_RX = @(tau) sqrt(dot( POS_fun_RX(tau),POS_fun_RX(tau),1));

        LOS_fun_RX_x = @(tau) -POS_fun_RX_x(tau)./Range_fun_RX(tau);  
        LOS_fun_RX_y = @(tau) -POS_fun_RX_y(tau)./Range_fun_RX(tau);  
        LOS_fun_RX_z = @(tau) -POS_fun_RX_z(tau)./Range_fun_RX(tau);  
        LOS_fun_RX = @(tau) [LOS_fun_RX_x(tau);LOS_fun_RX_y(tau);LOS_fun_RX_z(tau)];


        theta_fun_RX = @(tau) unwrap(atan2( LOS_fun_RX_y(tau),LOS_fun_RX_x(tau) ));
        % d_theta_fun_RX = @(tau) (theta_fun_RX(tau + eps*1e4)- theta_fun_RX(tau))/(eps*1e4);
        % d_theta_fun_RX = @(tau) (theta_fun_RX(tau + eps*1e3)- theta_fun_RX(tau))/(eps*1e3);
        d_theta_fun_RX = @(tau) (theta_fun_RX(tau + eps*1e6)- theta_fun_RX(tau))/(eps*1e6);


        Vel_cross_LOS_fun_RX_x = @(tau) Vel_RX_0(2)*LOS_fun_RX_z(tau)-Vel_RX_0(3)*LOS_fun_RX_y(tau);
        Vel_cross_LOS_fun_RX_y = @(tau) Vel_RX_0(3)*LOS_fun_RX_x(tau)-Vel_RX_0(1)*LOS_fun_RX_z(tau);
        Vel_cross_LOS_fun_RX_z = @(tau) Vel_RX_0(1)*LOS_fun_RX_y(tau)-Vel_RX_0(2)*LOS_fun_RX_x(tau);

        d_LOS_fun_RX_x = @(tau) (LOS_fun_RX_y(tau).*Vel_cross_LOS_fun_RX_z(tau)-LOS_fun_RX_z(tau).*Vel_cross_LOS_fun_RX_y(tau))./Range_fun_RX(tau);
        d_LOS_fun_RX_y = @(tau) (LOS_fun_RX_z(tau).*Vel_cross_LOS_fun_RX_x(tau)-LOS_fun_RX_x(tau).*Vel_cross_LOS_fun_RX_z(tau))./Range_fun_RX(tau);
        d_LOS_fun_RX_z = @(tau) (LOS_fun_RX_x(tau).*Vel_cross_LOS_fun_RX_y(tau)-LOS_fun_RX_y(tau).*Vel_cross_LOS_fun_RX_x(tau))./Range_fun_RX(tau);

        d_LOS_fun_RX = @(tau) [d_LOS_fun_RX_x(tau);d_LOS_fun_RX_y(tau);d_LOS_fun_RX_z(tau)];
 

    otherwise,
    display('Undefined Flight Path type');
    error('Undefined Flight Path type');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

RGrad_fun_x = @(tau) LOS_fun_TX_x(tau) + LOS_fun_RX_x(tau);
RGrad_fun_y = @(tau) LOS_fun_TX_y(tau) + LOS_fun_RX_y(tau);
RGrad_fun_z = @(tau) LOS_fun_TX_z(tau) + LOS_fun_RX_z(tau);
RGrad_fun = @(tau) [RGrad_fun_x(tau);RGrad_fun_y(tau);RGrad_fun_z(tau)];

RGrad_GP_fun = @(tau) [RGrad_fun_x(tau);RGrad_fun_y(tau);zeros(size(tau))];
RGrad_GP_mag_fun = @(tau) sqrt(dot(RGrad_GP_fun(tau),RGrad_GP_fun(tau),1));
d_RGrad_GP_mag_fun = @(tau) (RGrad_GP_mag_fun(tau+eps*1e6)-RGrad_GP_mag_fun(tau))/(eps*1e6);


theta_ang_fun = @(tau) unwrap(atan2(RGrad_fun_y(tau),RGrad_fun_x(tau)));
% d_theta_ang_fun = @(tau) (theta_ang_fun(tau+eps*1e4) -theta_ang_fun(tau))/(eps*1e4);
d_theta_ang_fun = @(tau) (theta_ang_fun(tau+eps*1e6) -theta_ang_fun(tau))/(eps*1e6);

RRGrad_fun_x = @(tau) d_LOS_fun_TX_x(tau) + d_LOS_fun_RX_x(tau);
RRGrad_fun_y = @(tau) d_LOS_fun_TX_y(tau) + d_LOS_fun_RX_y(tau);
RRGrad_fun_z = @(tau) d_LOS_fun_TX_z(tau) + d_LOS_fun_RX_z(tau);
RRGrad_fun = @(tau) [RRGrad_fun_x(tau);RRGrad_fun_y(tau);RRGrad_fun_z(tau)];

RRGrad_GP_fun = @(tau) [RRGrad_fun_x(tau);RRGrad_fun_y(tau);zeros(size(tau))];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RRRGrad_fun_x = @(tau) (RRGrad_fun_x(tau + eps*1e6)-RRGrad_fun_x(tau))/(eps*1e6);
RRRGrad_fun_y = @(tau) (RRGrad_fun_y(tau + eps*1e6)-RRGrad_fun_y(tau))/(eps*1e6);
RRRGrad_fun_z = @(tau) (RRGrad_fun_z(tau + eps*1e6)-RRGrad_fun_z(tau))/(eps*1e6);
RRRGrad_fun = @(tau) [RRRGrad_fun_x(tau);RRRGrad_fun_y(tau);RRRGrad_fun_z(tau)];

RRRGrad_GP_fun = @(tau) [RRRGrad_fun_x(tau);RRRGrad_fun_y(tau);zeros(size(tau))];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mod_vec_fun_x = @(tau) RGrad_fun_x(tau) +2*(Range_fun_TX(tau)+Range_fun_RX(tau)).*RRGrad_fun_x(tau)/c_sol;
Mod_vec_fun_y = @(tau) RGrad_fun_y(tau) +2*(Range_fun_TX(tau)+Range_fun_RX(tau)).*RRGrad_fun_y(tau)/c_sol;
% Mod_vec_fun_z = @(tau) RGrad_fun_z(tau) +2*(Range_fun_TX(tau)+Range_fun_RX(tau))*RRGrad_fun_z(tau)/c_sol;
Mod_vec_fun_z = @(tau) zeros(size(tau));
Mod_vec_fun = @(tau) [Mod_vec_fun_x(tau);Mod_vec_fun_y(tau);Mod_vec_fun_z(tau)];
Mod_vec_mag_fun = @(tau) sqrt(dot(Mod_vec_fun(tau),Mod_vec_fun(tau),1)); 


d_Mod_vec_fun_x = @(tau) (Mod_vec_fun_x(tau+eps*1e6)-Mod_vec_fun_x(tau))/(eps*1e6);
d_Mod_vec_fun_y = @(tau) (Mod_vec_fun_y(tau+eps*1e6)-Mod_vec_fun_y(tau))/(eps*1e6);
d_Mod_vec_fun_z = @(tau) (Mod_vec_fun_z(tau+eps*1e6)-Mod_vec_fun_z(tau))/(eps*1e6);
d_Mod_vec_fun = @(tau) [d_Mod_vec_fun_x(tau);d_Mod_vec_fun_y(tau);d_Mod_vec_fun_z(tau)];

display('Update this');
% fc_fm_fun = @(tau) fc*RGrad_GP_mag_fun(0)./RGrad_GP_mag_fun(tau);
% BW_fm_fun = @(tau) BW*RGrad_GP_mag_fun(0)./RGrad_GP_mag_fun(tau);

fc_fm_fun = @(tau) fc*Mod_vec_mag_fun(0)./Mod_vec_mag_fun(tau);
BW_fm_fun = @(tau) BW*Mod_vec_mag_fun(0)./Mod_vec_mag_fun(tau);

%These might be wrong
% d_fc_fm_fun = @(tau) -fc*RGrad_GP_mag_fun(0)*d_RGrad_GP_mag_fun(tau).*RGrad_GP_mag_fun(tau).^(-2);
% d_BW_fm_fun = @(tau) -BW*RGrad_GP_mag_fun(0)*d_RGrad_GP_mag_fun(tau).*RGrad_GP_mag_fun(tau).^(-2);

d_fc_fm_fun = @(tau) (fc_fm_fun(tau+eps*1e6)-fc_fm_fun(tau))/(eps*1e6);
d_BW_fm_fun = @(tau) (BW_fm_fun(tau+eps*1e6)-BW_fm_fun(tau))/(eps*1e6);

% d_fc_fm_fun = @(tau) fc*RGrad_GP_mag_fun(0)*d_RGrad_GP_mag_fun(tau)./(RGrad_GP_mag_fun(tau).^2);
% d_BW_fm_fun = @(tau) BW*RGrad_GP_mag_fun(0)*d_RGrad_GP_mag_fun(tau)./(RGrad_GP_mag_fun(tau).^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

theta_ang_fun2 = @(t) .5*(unwrap(theta_fun_TX(t))+unwrap(theta_fun_RX(t)))-pi;
d_theta_ang_fun2 = @(t) .5*(d_theta_fun_TX(t)+d_theta_fun_RX(t));

% if(theta_ang_fun(1e10*eps)-theta_ang_fun(0)>0), %if increasing rotation rate
% 
%     t_max = fminbnd(@(tau) abs(theta_ang_fun(tau)-theta_ang_fun(0)-.5*int_Ang_rad),0,1e4);
%     t_min = fminbnd(@(tau) abs(theta_ang_fun_temp(tau)+.5*int_Ang_rad),-1e4,0);
% 
% else


% if(theta_ang_fun2(1e10*eps)-theta_ang_fun2(0)>0),
%     t_max = fminbnd(@(tau) abs(theta_ang_fun2(tau)+.5*int_Ang_rad),0,test_time/2);
%     t_min = fminbnd(@(tau) abs(theta_ang_fun2(tau)-.5*int_Ang_rad),-test_time/2,0);
% else
%     t_max = fminbnd(@(tau) abs(theta_ang_fun2(tau)-.5*int_Ang_rad),0,1e4);
%     t_min = fminbnd(@(tau) abs(theta_ang_fun2(tau)+.5*int_Ang_rad),-1e4,0);
% end


test_time =  2*int_Ang_rad/abs(d_theta_ang_fun2(0));
% t_max = fminbnd(@(tau) abs(theta_ang_fun2(tau)-theta_ang_fun2(-tau))-int_Ang_rad,0,2*int_Ang_rad/abs(d_theta_ang_fun2(0)))
% t_min = -t_max;

if(d_theta_ang_fun2(0)>0),
    t_max = fminbnd(@(tau) abs(theta_ang_fun2(tau)+pi-int_Ang_rad/2) ,0 ,test_time/2);
    t_min = fminbnd(@(tau) abs(theta_ang_fun2(tau)+pi+int_Ang_rad/2) ,-test_time/2,0);    
else
    t_max = fminbnd(@(tau) abs(theta_ang_fun2(tau)+pi+int_Ang_rad/2) ,0 ,test_time/2);
    t_min = fminbnd(@(tau) abs(theta_ang_fun2(tau)+pi-int_Ang_rad/2) ,-test_time/2,0);    
end

%Coherent processing interval;
T_c = t_max-t_min; 

%nyquist
scene_radius = 0;

%Put the target info into arrays, I'm going to use this later when I
%generate phase history data.  Needs to be in an array to upload to GPU
%memory
tgt_x = zeros(1,length(parms.tgt));
tgt_y = zeros(1,length(parms.tgt));
tgt_z = zeros(1,length(parms.tgt));
tgt_rho = zeros(1,length(parms.tgt));

for ii=1:length(parms.tgt),
    tgt_x(ii) = parms.tgt(ii).r(1);
    tgt_y(ii) = parms.tgt(ii).r(2);
    tgt_z(ii) = parms.tgt(ii).r(3);
    tgt_rho(ii) = parms.tgt(ii).rho;    
    scene_radius = max([scene_radius sqrt(dot(parms.tgt(ii).r,parms.tgt(ii).r))]);
end

%max rad will be when x=max and y=max so multiply sqrt(2)
%rect grid not a circle
% scene_radius = sqrt(2)*scene_radius;

scene_radius = 2*sqrt(2)*scene_radius;

% dK = (2*pi/c_sol)*(RGrad * df + RRGrad * dt)

%nyquist
% |dK| < 2*pi/scene_radius (rad/m)
% (RGrad * df + RRGrad * dt)< c_sol/(2*scene_radius)
% df_max = c_sol/(2*scene_radius*max(|RGrad|));
% dt_max = c_sol/(2*scene_radius*max(|RRGrad |));

%looking for when Range Gradient has max length, use fminbnd to find the reciprocal
[dont_care, reciprocal_max_RGrad_abs] = fminbnd(@(tau) 1./sqrt(dot(RGrad_GP_fun(tau),RGrad_GP_fun(tau),1)),t_min,t_max);
max_RGrad_abs = 1/reciprocal_max_RGrad_abs;

[dont_care,reciprocal_max_RRGrad_abs] = fminbnd(@(tau) 1./sqrt(dot(RRGrad_GP_fun(tau),RRGrad_GP_fun(tau),1)),t_min,t_max);
max_RRGrad_abs = 1/reciprocal_max_RRGrad_abs;

df_max = c_sol/(2*scene_radius*max_RGrad_abs);
dt_max = c_sol/(2*scene_radius*max_RRGrad_abs);

%discrete time, we are going to use the next power of 2 past Nyquist. This
%will give us some margin for agile modulations. Use 256 minimum sample number 

N_f = max([2^8 2^(ceil( log(BW/df_max)/log(2))+2)]);
N_t = max([2^8 2^(ceil( log(T_c/dt_max)/log(2))+2)]);

freq = linspace(-BW/2,BW/2,N_f)+fc;
t = linspace(t_min,t_max,N_t);

fc_fm = fc_fm_fun(t);
BW_fm = BW_fm_fun(t);
d_fc_fm = d_fc_fm_fun(t);
d_BW_fm = d_BW_fm_fun(t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
freq_mesh = freq.'*ones(1,N_t);
freq_fm_mesh = zeros(N_f,N_t);
dt_freq_fm_mesh = zeros(N_f,N_t);
for ii=1:N_t,
    freq_fm_mesh(:,ii) = linspace(-BW_fm(ii)/2,BW_fm(ii)/2,N_f)+fc_fm(ii);
    dt_freq_fm_mesh(:,ii) = linspace(-d_BW_fm(ii)/2,d_BW_fm(ii)/2,N_f)+d_fc_fm(ii);    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Discrete time Geometry parameters
%Can't stay cont time in a computer sim :)

%Pos, Vel, LOS. d/dt LOS
POS_TX = POS_fun_TX(t);
POS_RX = POS_fun_RX(t);
Vel_TX = Vel_fun_TX(t);
Vel_RX = Vel_fun_RX(t);
LOS_TX = LOS_fun_TX(t);
LOS_RX = LOS_fun_RX(t);
d_LOS_TX = d_LOS_fun_TX(t);
d_LOS_RX = d_LOS_fun_RX(t);

%%%%%
RGrad_x = RGrad_fun_x(t);
RGrad_y = RGrad_fun_y(t);
RGrad_z = RGrad_fun_z(t);
RGrad   = RGrad_fun(t);
RGrad_GP= RGrad_GP_fun(t);

%%%%
RRGrad_x = RRGrad_fun_x(t);
RRGrad_y = RRGrad_fun_y(t);
RRGrad_z = RRGrad_fun_z(t);
RRGrad   = RRGrad_fun(t);
RRGrad_GP= RRGrad_GP_fun(t);

%%%%
RRRGrad_x = RRRGrad_fun_x(t);
RRRGrad_y = RRRGrad_fun_y(t);
RRRGrad_z = RRRGrad_fun_z(t);
RRRGrad   = RRRGrad_fun(t);
RRRGrad_GP= RRRGrad_GP_fun(t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Range_TX = Range_fun_TX(t);
Range_RX = Range_fun_RX(t);
Bi_Range = Range_TX + Range_RX;

RRange_TX = -dot(Vel_TX,LOS_TX,1);
RRange_RX = -dot(Vel_RX,LOS_RX,1);
Bi_RRange = RRange_TX + RRange_RX;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This is just for showing the flight paths for a much longer flight time
%than when the SAR image is being displaced.

if(~TX_time_limit_flag && ~RX_time_limit_flag),
    %No time limiatations.
    t_min2 = 4*t_min;
    t_max2 = 4*t_max;


elseif(TX_time_limit_flag && ~RX_time_limit_flag),
 %TX has a time limit
    
     if(Range_Rate_TX<0), %max time limit
         %t_max2 < -Range_fun_TX(0)/Range_Rate_TX
         t_max2 = -Range_fun_TX(0)/Range_Rate_TX - eps*1e-10;
         t_min2 = t_max2-4*T_c;

     else %min time limit
         %tmin2 > -Range_fun_TX(0)/Range_Rate_TX
         t_min2 = -Range_fun_TX(0)/Range_Rate_TX + eps*1e-10;
         t_max2 = t_min2 + 4*T_c; 
     end


elseif(~TX_time_limit_flag && RX_time_limit_flag),
    %RX has a time limit
    
     if(Range_Rate_RX<0), %max time limit
         %t_max2 < -Range_fun_RX(0)/Range_Rate_RX
         t_max2 = -Range_fun_RX(0)/Range_Rate_RX - eps*1e-10;
         t_min2 = min([t_min t_max2-4*T_c]);
        
     else %min time limit
         %tmin2 > -Range_fun_RX(0)/Range_Rate_RX
         t_min2 = -Range_fun_RX(0)/Range_Rate_RX + eps*1e-10;
         % t_max2 = t_min2+4*T_c; 
         t_max2 = max([t_max t_min2+4*T_c]);
     end


elseif(TX_time_limit_flag && RX_time_limit_flag),

    if(Range_Rate_TX<0), %TX has max time limit
         TX_tmax2 = -Range_fun_TX(0)/Range_Rate_TX - eps*1e-10;
         TX_tmin2 = min([t_min TX_tmax2-4*T_c]);

     else %min time limit
         %tmin2 > -Range_fun_TX(0)/Range_Rate_TX
         TX_tmin2 = -Range_fun_TX(0)/Range_Rate_TX + eps*1e-10;
         TX_tmax2 = max([t_max TX_tmin2 + 4*T_c]); 
     end


     if(Range_Rate_RX<0), %max time limit
         %tmax2 < -Range_fun_RX(0)/Range_Rate_RX
         RX_tmax2 = -Range_fun_RX(0)/Range_Rate_RX - eps*1e-10;
         % RX_tmin2 = RX_tmax2-4*T_c;
         RX_tmin2 = min([t_min RX_tmax2-4*T_c]);

     else %min time limit
         %tmin2 > -Range_fun_RX(0)/Range_Rate_RX
         RX_tmin2 = -Range_fun_RX(0)/Range_Rate_RX + eps*1e-10;
         % RX_tmax2 = RX_tmin2 + 4*T_c; 
         RX_tmax2 = max([t_max RX_tmin2+4*T_c]);
     end

     t_max2 = min([TX_tmax2 RX_tmax2]);
     t_min2 = max([TX_tmin2 RX_tmin2]);
   

end


%The three things that will be saved for plotting
t2 =  linspace(t_min2,t_max2,8*N_t);
POS_TX2 = POS_fun_TX(t2);
POS_RX2 = POS_fun_RX(t2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%K-space
k_const = 2*pi/c_sol;

%Move-stop-Move
% F_x = k_const*freq_mesh.*(ones(N_f,1)*RGrad_x);
% F_y = k_const*freq_mesh.*(ones(N_f,1)*RGrad_y);
% F_z = zeros(N_f,N_t);

%Const Fast-Time Vel
F_x = k_const*freq_mesh.*(ones(N_f,1)*(RGrad_x + 2*Bi_Range.*RRGrad_x/c_sol));
F_y = k_const*freq_mesh.*(ones(N_f,1)*(RGrad_y + 2*Bi_Range.*RRGrad_y/c_sol));
F_z = zeros(N_f,N_t);

%%%%
%Move-stop-Move
% G_x = k_const*freq_fm_mesh.*(ones(N_f,1)*RGrad_x);
% G_y = k_const*freq_fm_mesh.*(ones(N_f,1)*RGrad_y);
% G_z = zeros(N_f,N_t);

%Const Fast-Time Vel
G_x = k_const*freq_fm_mesh.*(ones(N_f,1)*(RGrad_x + 2*Bi_Range.*RRGrad_x/c_sol));
G_y = k_const*freq_fm_mesh.*(ones(N_f,1)*(RGrad_y + 2*Bi_Range.*RRGrad_y/c_sol));
G_z = zeros(N_f,N_t);


%Move-stop-Move
% Ft_x = k_const*freq_mesh.*(ones(N_f,1)*RRGrad_x);
% Ft_y = k_const*freq_mesh.*(ones(N_f,1)*RRGrad_y);
% Ft_z = zeros(N_f,N_t);

%Const Fast-Time Vel
Ft_x = k_const*freq_mesh.*(ones(N_f,1)*(RRGrad_x + (2/c_sol)*(Bi_RRange.*RRGrad_x + Bi_Range.*RRRGrad_x)));
Ft_y = k_const*freq_mesh.*(ones(N_f,1)*(RRGrad_y + (2/c_sol)*(Bi_RRange.*RRGrad_y + Bi_Range.*RRRGrad_y)));
Ft_z = zeros(N_f,N_t);

NFt_x = Ft_x./sqrt(Ft_x.^2+Ft_y.^2);
NFt_y = Ft_y./sqrt(Ft_x.^2+Ft_y.^2);
NFt_z = zeros(N_f,N_t);


%Move-stop-Move
% Gt_x = k_const*( ones(N_f,1)*(d_fc_fm.*RGrad_x) + freq_fm_mesh.*(ones(N_f,1)*RRGrad_x));
% Gt_y = k_const*( ones(N_f,1)*(d_fc_fm.*RGrad_y) + freq_fm_mesh.*(ones(N_f,1)*RRGrad_y));
% Gt_z = zeros(N_f,N_t);


%Const Fast-Time Vel
Gt_x = k_const*dt_freq_fm_mesh.*(ones(N_f,1)*(RGrad_x + 2*Bi_Range.*RRGrad_x/c_sol)) +...
k_const*freq_fm_mesh.*(ones(N_f,1)*(RRGrad_x + 2*Bi_RRange.*RRGrad_x/c_sol + 2*Bi_Range.*RRRGrad_x/c_sol));
Gt_y = k_const*dt_freq_fm_mesh.*(ones(N_f,1)*(RGrad_y + 2*Bi_Range.*RRGrad_y/c_sol)) +...
k_const*freq_fm_mesh.*(ones(N_f,1)*(RRGrad_y + 2*Bi_RRange.*RRGrad_y/c_sol + 2*Bi_Range.*RRRGrad_y/c_sol));
Gt_z = zeros(N_f,N_t);

NGt_x = Gt_x./sqrt(Gt_x.^2+Gt_y.^2);
NGt_y = Gt_y./sqrt(Gt_x.^2+Gt_y.^2);
NGt_z = zeros(N_f,N_t);

%Partial derivative w.r.t freq f

%Move-stop-Move
% Ff_x = k_const*ones(N_f,N_t).*(ones(N_f,1)*RGrad_x);
% Ff_y = k_const*ones(N_f,N_t).*(ones(N_f,1)*RGrad_y);
% Ff_z = zeros(N_f,N_t);

Ff_x = k_const*ones(N_f,1)*(RGrad_x + 2*Bi_Range.*RRGrad_x/c_sol);
Ff_y = k_const*ones(N_f,1)*(RGrad_y + 2*Bi_Range.*RRGrad_y/c_sol);
Ff_z = zeros(N_f,N_t);


NFf_x = Ff_x./sqrt(Ff_x.^2+Ff_y.^2);
NFf_y = Ff_y./sqrt(Ff_x.^2+Ff_y.^2);
NFf_z = zeros(N_f,N_t);


%Move-stop-Move
% Gf_x = k_const*ones(N_f,N_t).*(ones(N_f,1)*RGrad_x);
% Gf_y = k_const*ones(N_f,N_t).*(ones(N_f,1)*RGrad_y);
% Gf_z = zeros(N_f,N_t);

%Const Fast-Time Vel
Gf_x = k_const*ones(N_f,1)*(RGrad_x + 2*Bi_Range.*RRGrad_x/c_sol);
Gf_y = k_const*ones(N_f,1)*(RGrad_y + 2*Bi_Range.*RRGrad_y/c_sol);
Gf_z = zeros(N_f,N_t);

NGf_x = Gf_x./sqrt(Gf_x.^2+Gf_y.^2);
NGf_y = Gf_y./sqrt(Gf_x.^2+Gf_y.^2);
NGf_z = zeros(N_f,N_t);

%Should be zeros - with ground plane projection
Ft_cross_Ff_x = Ft_y.*Ff_z - Ft_z.*Ff_y;
Gt_cross_Gf_x = Gt_y.*Gf_z - Gt_z.*Gf_y;


%Should be zeros - with ground plane projection
Ft_cross_Ff_y = Ft_z.*Ff_x - Ft_x.*Ff_z;
Gt_cross_Gf_y = Gt_z.*Gf_x - Gt_x.*Gf_z;

%Should be non-zeros 
Ft_cross_Ff_z = Ft_x.*Ff_y - Ft_y.*Ff_x;
Gt_cross_Gf_z = Gt_x.*Gf_y - Gt_y.*Gf_x;

weight_F = 1./abs(Ft_cross_Ff_z);
weight_F = weight_F/sum(weight_F(:));

weight_G = 1./abs(Gt_cross_Gf_z);
weight_G = weight_G/sum(weight_G(:));

weight_uniform = ones(size(weight_F));
weight_uniform = weight_uniform/sum(weight_uniform(:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Lets play with soem other weight functions :)

%Slowtime parameterized center frequency K-space path along passband
%surface( when projected to the ground plane).
% \vec{alpha}(t)
% alpha_k_pos_fun = @(tau) k_const*fc*[RGrad_fun_x(tau);RGrad_fun_y(tau);zeros(size(tau))];
% bravo_k_pos_fun = @(tau) k_const*fc_fm_fun(tau).*[RGrad_fun_x(tau);RGrad_fun_y(tau);zeros(size(tau))];

alpha_k_pos_fun = @(tau) k_const*fc*Mod_vec_fun(tau);
bravo_k_pos_fun = @(tau) k_const*fc_fm_fun(tau).*[Mod_vec_fun_x(tau);Mod_vec_fun_y(tau);zeros(size(tau))];


%time derivative % \vec{\dot{alpha}}(t) = \frac{d}{dt} \vec{\alpha}(t)
% dot_alpha_k_pos_fun = @(tau) k_const*fc*[RRGrad_fun_x(tau);RRGrad_fun_y(tau);zeros(size(tau))];
% dot_bravo_k_pos_fun = @(tau) k_const*d_fc_fm_fun(tau).*[RGrad_fun_x(tau);RGrad_fun_y(tau);zeros(size(tau))] + ...
%     fc_fm_fun(tau).*[RRGrad_fun_x(tau);RRGrad_fun_y(tau);zeros(size(tau))];

dot_alpha_k_pos_fun = @(tau) k_const*fc*d_Mod_vec_fun(tau);
dot_bravo_k_pos_fun = @(tau) k_const*d_fc_fm_fun(tau).*Mod_vec_fun(tau) + ...
    fc_fm_fun(tau).*d_Mod_vec_fun(tau);


%speed along path % |\vec{\dot{alpha}}(t)| 
dot_alpha_k_pos_mag_fun = @(tau) sqrt(dot(dot_alpha_k_pos_fun(tau),dot_alpha_k_pos_fun(tau),1 ));
dot_bravo_k_pos_mag_fun = @(tau) sqrt(dot(dot_bravo_k_pos_fun(tau),dot_bravo_k_pos_fun(tau),1 ));

%ArcLength, distance traveled along path %\int _{t_min} ^t |\vec{\dot{alpha}}(\tau)| d \tau 
%rad/m
arcLength_alpha_k_fun = @(tau) integral(@(v) dot_alpha_k_pos_mag_fun(v),t_min,tau);    
arcLength_bravo_k_fun = @(tau) integral(@(v) dot_bravo_k_pos_mag_fun(v),t_min,tau);    

%ArcLength with N_t steps along center frequency path (of surface F)
s_F = linspace(0,arcLength_alpha_k_fun(t_max),N_t );
s_G = linspace(0,arcLength_bravo_k_fun(t_max),N_t );

%Slow-time that corresponds with the uniform spaced arcLength samples 
t_sF = zeros(size(s_F));
t_sG = zeros(size(s_G));

t_sF(1) = t_min;
t_sG(1) = t_min;

t_sF(end) = t_max;
t_sG(end) = t_max;

for ii=2:N_t-1,
    t_sF(ii) = fminbnd(@(tau) abs(arcLength_alpha_k_fun(tau) - s_F(ii)),t_min,t_max); 
    t_sG(ii) = fminbnd(@(tau) abs(arcLength_bravo_k_fun(tau) - s_G(ii)),t_min,t_max);     
end

freq_weight = taylorwin(N_f,5,-35);
freq_weight = freq_weight/sum(freq_weight);
ArcLength_weight = taylorwin(N_t,5,-35);
ArcLength_weight = ArcLength_weight/sum(ArcLength_weight);

time_weight_F = interp1(t_sF,ArcLength_weight,t);
time_weight_G = interp1(t_sG,ArcLength_weight,t);

taylorwin_2D_F = (freq_weight*time_weight_F).*weight_F;
taylorwin_2D_F = taylorwin_2D_F/sum(taylorwin_2D_F(:));
taylorwin_2D_G = (freq_weight*time_weight_G).*weight_G;
taylorwin_2D_G = taylorwin_2D_G/sum(taylorwin_2D_G(:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%averages

%K-space centriod
% AVG_F = [sum(sum(weight_F.*F_x));sum(sum(weight_F.*F_y));sum(sum(weight_F.*F_z))]; 
% AVG_G = [sum(sum(weight_G.*G_x));sum(sum(weight_G.*G_y));sum(sum(weight_G.*G_z))]; 

AVG_F = [mean(F_x(:));mean(F_y(:));mean(F_z(:))]; 
AVG_G = [mean(G_x(:));mean(G_y(:));mean(G_z(:))]; 

%Average time modulation
% AVG_Ft = [sum(sum(weight_F.*Ft_x));sum(sum(weight_F.*Ft_y));sum(sum(weight_F.*Ft_z))]; 
% AVG_Gt = [sum(sum(weight_G.*Gt_x));sum(sum(weight_G.*Gt_y));sum(sum(weight_G.*Gt_z))]; 

AVG_Ft = [mean(Ft_x(:));mean(Ft_y(:));mean(Ft_z(:))]; 
AVG_Gt = [mean(Gt_x(:));mean(Gt_y(:));mean(Gt_z(:))]; 


%Average freq modulation
% AVG_Ff = [sum(sum(weight_F.*Ff_x));sum(sum(weight_F.*Ff_y));sum(sum(weight_F.*Ff_z))]; 
% AVG_Gf = [sum(sum(weight_G.*Gf_x));sum(sum(weight_G.*Gf_y));sum(sum(weight_G.*Gf_z))]; 

AVG_Ff = [mean(Ff_x(:));mean(Ff_y(:));mean(Ff_z(:))]; 
AVG_Gf = [mean(Gf_x(:));mean(Gf_y(:));mean(Gf_z(:))]; 

%Doppler direction
D_vec = AVG_Ft/sqrt(dot(AVG_Ft,AVG_Ft));
D_fm_vec = AVG_Gt/sqrt(dot(AVG_Gt,AVG_Gt));

%Range direction
R_vec = AVG_Ff/sqrt(dot(AVG_Ff,AVG_Ff));
R_fm_vec = AVG_Gf/sqrt(dot(AVG_Gf,AVG_Gf));

%Range/Doppler surface norm - slant plane is plane that best approximates
%curved surface. N_vec is normal to slant plane surface  

N_vec = cross(R_vec,D_vec); %In the \vec{z} direction but not unity
%normalize it to force it to be unity
N_vec = N_vec/sqrt(dot(N_vec,N_vec));

N_fm_vec = cross(R_fm_vec,D_fm_vec);
N_fm_vec = N_fm_vec/sqrt(dot(N_fm_vec,N_fm_vec));

%Cross-Range direction
XR_vec = cross(R_vec,N_vec);
XR_fm_vec = cross(R_fm_vec,N_fm_vec);

%Cross-Doppler direction
XD_vec = cross(N_vec,D_vec);
XD_fm_vec = cross(N_fm_vec,D_fm_vec);

%%%%%%%%%%%%%%%%
%K-space bandwidth
BW_XR = sqrt(12*sum(sum(weight_F.*( XR_vec(1)*(F_x-AVG_F(1)) + XR_vec(2)*(F_y-AVG_F(2)) + XR_vec(3)*(F_z-AVG_F(3))).^2)));
BW_XD = sqrt(12*sum(sum(weight_F.*( XD_vec(1)*(F_x-AVG_F(1)) + XD_vec(2)*(F_y-AVG_F(2)) + XD_vec(3)*(F_z-AVG_F(3))).^2)));
BW_R  = sqrt(12*sum(sum(weight_F.*(  R_vec(1)*(F_x-AVG_F(1)) +  R_vec(2)*(F_y-AVG_F(2)) +  R_vec(3)*(F_z-AVG_F(3))).^2)));

BW_XR_fm = sqrt(12*sum(sum(weight_G.*( XR_fm_vec(1)*(G_x-AVG_G(1)) + XR_fm_vec(2)*(G_y-AVG_G(2)) + XR_fm_vec(3)*(G_z-AVG_G(3))).^2)));
BW_XD_fm = sqrt(12*sum(sum(weight_G.*( XD_fm_vec(1)*(G_x-AVG_G(1)) + XD_fm_vec(2)*(G_y-AVG_G(2)) + XD_fm_vec(3)*(G_z-AVG_G(3))).^2)));
BW_R_fm  = sqrt(12*sum(sum(weight_G.*(  R_fm_vec(1)*(G_x-AVG_G(1)) +  R_fm_vec(2)*(G_y-AVG_G(2)) +  R_fm_vec(3)*(G_z-AVG_G(3))).^2)));


%spatial resolution
%Note: I added 10% margin to BW and rotation angle
%These resolutions will be finner than commanded
%We may want to sacrifice these resolutions for sidlobe supression or PFA
%inscription.
res_XR = 2*pi/BW_XR;
res_XD = 2*pi/BW_XD;
res_R  = 2*pi/BW_R;

res_XR_fm = 2*pi/BW_XR_fm;
res_XD_fm = 2*pi/BW_XD_fm;
res_R_fm  = 2*pi/BW_R_fm;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Project surfaces F and G anlong R and XR directions

%Surface F
%%%%%%%%
F_r = F_x*R_vec(1) + F_y*R_vec(2) + F_z*R_vec(3);
F_xr = F_x*XR_vec(1) + F_y*XR_vec(2) + F_z*XR_vec(3);

Ft_r = Ft_x*R_vec(1) + Ft_y*R_vec(2) + Ft_z*R_vec(3);
Ft_xr = Ft_x*XR_vec(1) + Ft_y*XR_vec(2) + Ft_z*XR_vec(3);

Ff_r = Ff_x*R_vec(1) + Ff_y*R_vec(2) + Ff_z*R_vec(3);
Ff_xr = Ff_x*XR_vec(1) + Ff_y*XR_vec(2) + Ff_z*XR_vec(3);


%Surface G
%%%%%%%%
G_r = G_x*R_fm_vec(1) + G_y*R_fm_vec(2) + G_z*R_fm_vec(3);
G_xr = G_x*XR_fm_vec(1) + G_y*XR_fm_vec(2) + G_z*XR_fm_vec(3);

Gt_r = Gt_x*R_fm_vec(1) + Gt_y*R_fm_vec(2) + Gt_z*R_fm_vec(3);
Gt_xr = Gt_x*XR_fm_vec(1) + Gt_y*XR_fm_vec(2) + Gt_z*XR_fm_vec(3);

Gf_r = Gf_x*R_fm_vec(1) + Gf_y*R_fm_vec(2) + Gf_z*R_fm_vec(3);
Gf_xr = Gf_x*XR_fm_vec(1) + Gf_y*XR_fm_vec(2) + Gf_z*XR_fm_vec(3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dx_max = .5*min([res_XR res_R]);

N_x = 2^ceil(log(2*scene_radius/dx_max)/log(2));

r = linspace(-scene_radius,scene_radius,N_x+1);
r = r(1:end-1);
xr=r;
dr = r(2)-r(1);

[r_mesh,xr_mesh] = meshgrid(r,xr);

x_mesh = r_mesh*R_vec(1) + xr_mesh*XR_vec(1); 
y_mesh = r_mesh*R_vec(2) + xr_mesh*XR_vec(2);



%%%%%%%%%%%
dx_psf_max = dx_max/2;
% max_x_psf = 10*min([res_XR res_R]);
max_x_psf = 20*min([res_XR res_R]);
N_psf = 2^ceil(log(2*max_x_psf/dx_psf_max)/log(2));

r_psf = linspace(-max_x_psf,max_x_psf,N_psf+1);
r_psf = r_psf(1:end-1);
xr_psf = r_psf;
dr_psf = r_psf(2)-r_psf(1);

[r_mesh_psf,xr_mesh_psf] = meshgrid(r_psf,xr_psf);

x_mesh_psf = r_mesh_psf*R_vec(1) + xr_mesh_psf*XR_vec(1); 
y_mesh_psf = r_mesh_psf*R_vec(2) + xr_mesh_psf*XR_vec(2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('PSF CUDA');

%prep the CUDA Kernel
k = parallel.gpu.CUDAKernel('PSF_Kernel.ptx','PSF_Kernel.cu');
N1=512;
N2=2^ceil(log( N_psf*N_psf/N1)/log(2));
N2=max([N2 1]);
k.ThreadBlockSize = [N1,1,1];
k.GridSize=[N2,1,1];


%allocate memory on GPU for output
PSF_real = gpuArray(zeros(N_psf,N_psf));
PSF_imag = gpuArray(zeros(N_psf,N_psf));

%Push data onto GPU memory 
x_psf_gpu = gpuArray(x_mesh_psf(:));
y_psf_gpu = gpuArray(y_mesh_psf(:));
z_psf_gpu = gpuArray(zeros(size(x_mesh_psf(:))));

Pos_TX_gpu_x = gpuArray(POS_TX(1,:));
Pos_TX_gpu_y = gpuArray(POS_TX(2,:));
Pos_TX_gpu_z = gpuArray(POS_TX(3,:));

Pos_RX_gpu_x = gpuArray(POS_RX(1,:));
Pos_RX_gpu_y = gpuArray(POS_RX(2,:));
Pos_RX_gpu_z = gpuArray(POS_RX(3,:));

Vel_TX_gpu_x = gpuArray(Vel_TX(1,:));
Vel_TX_gpu_y = gpuArray(Vel_TX(2,:));
Vel_TX_gpu_z = gpuArray(Vel_TX(3,:));

Vel_RX_gpu_x = gpuArray(Vel_RX(1,:));
Vel_RX_gpu_y = gpuArray(Vel_RX(2,:));
Vel_RX_gpu_z = gpuArray(Vel_RX(3,:));

fc_gpu = gpuArray(fc*ones(1,N_t));
BW_gpu = gpuArray(BW*ones(1,N_t));

% wt_gpu = gpuArray(ones(1,N_t)/N_t);
% wf_gpu = gpuArray(ones(1,N_f)/N_f);

w2d_gpu = gpuArray(weight_uniform);
%Run the CUDA Kernel
[y1 y2] = feval(k,PSF_real,PSF_imag,...
    x_psf_gpu,y_psf_gpu,z_psf_gpu,...
    Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
    Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
    Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
    Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...    
    0,0,0,...
    fc_gpu,BW_gpu,w2d_gpu,...
    int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
psf = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
psf(:) = gather(y1)+j*gather(y2);

%Normalize the PSF
% psf = psf./max(abs(psf(:)));

%%%%%%%%
w2d_gpu = gpuArray(weight_F);
%Re-Run the CUDA Kernel for AM psf
[y1 y2] = feval(k,PSF_real,PSF_imag,...
    x_psf_gpu,y_psf_gpu,z_psf_gpu,...
    Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
    Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
    Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
    Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...    
    0,0,0,...
    fc_gpu,BW_gpu,w2d_gpu,...
    int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
psf_am = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
psf_am(:) = gather(y1)+j*gather(y2);

%Normalize the PSF
% psf_am = psf_am./max(abs(psf_am(:)));


%taylor window version

w2d_gpu = gpuArray(taylorwin_2D_F);
%re-Run the CUDA Kernel
[y1 y2] = feval(k,PSF_real,PSF_imag,...
    x_psf_gpu,y_psf_gpu,z_psf_gpu,...
    Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
    Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
    Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
    Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...    
    0,0,0,...
    fc_gpu,BW_gpu,w2d_gpu,...
    int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
psf_taylor = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
psf_taylor(:) = gather(y1)+j*gather(y2);

%Normalize the PSF
% psf_taylor = psf_taylor./max(abs(psf_taylor(:)));

%%%%%%
%Use the frequency agile parameters
w2d_gpu = gpuArray(weight_uniform);
fc_fm_gpu = gpuArray(fc_fm);
BW_fm_gpu = gpuArray(BW_fm);

%Re-Run the CUDA Kernel with new frequencies
[y1 y2] = feval(k,PSF_real,PSF_imag,...
    x_psf_gpu,y_psf_gpu,z_psf_gpu,...
    Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
    Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
    Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
    Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...    
    0,0,0,...
    fc_fm_gpu,BW_fm_gpu,w2d_gpu,...
    int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
psf_fm = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
psf_fm(:) = gather(y1)+j*gather(y2);

%Normalize the PSF
% psf_fm = psf_fm./max(abs(psf_fm(:)));


w2d_gpu = gpuArray(weight_G);
%Re-Run the CUDA Kernel with AM/FM
[y1 y2] = feval(k,PSF_real,PSF_imag,...
    x_psf_gpu,y_psf_gpu,z_psf_gpu,...
    Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
    Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
    Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
    Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...    
    0,0,0,...
    fc_fm_gpu,BW_fm_gpu,w2d_gpu,...
    int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
psf_amfm = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
psf_amfm(:) = gather(y1)+j*gather(y2);

%Normalize the PSF
% psf_amfm = psf_amfm./max(abs(psf_amfm(:)));


%taylor window with agile frequency version
w2d_gpu = gpuArray(taylorwin_2D_G);

%Re-Run the CUDA Kernel with new frequencies
[y1 y2] = feval(k,PSF_real,PSF_imag,...
    x_psf_gpu,y_psf_gpu,z_psf_gpu,...
    Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
    Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
    Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
    Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...    
    0,0,0,...
    fc_fm_gpu,BW_fm_gpu,w2d_gpu,...
    int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
psf_fm_taylor = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
psf_fm_taylor(:) = gather(y1)+j*gather(y2);

%Normalize the PSF
% psf_fm_taylor = psf_fm_taylor./max(abs(psf_fm_taylor(:)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('IPR CUDA');

%clear the GPU - No big deal for PSFs/IPRs but matters for Phase history 
% and SAR images 
reset(device);

%prep the CUDA Kernel
k = parallel.gpu.CUDAKernel('IPR_Kernel.ptx','IPR_Kernel.cu');
N1=512;
N2=2^ceil(log( N_psf*N_psf/N1)/log(2));
N2=max([N2 1]);
k.ThreadBlockSize = [N1,1,1];
k.GridSize=[N2,1,1];


%allocate memory on GPU for output
IPR_real = gpuArray(zeros(N_psf,N_psf));
IPR_imag = gpuArray(zeros(N_psf,N_psf));

%Push data onto GPU memory
x_psf_gpu = gpuArray(x_mesh_psf(:));
y_psf_gpu = gpuArray(y_mesh_psf(:));
z_psf_gpu = gpuArray(zeros(size(x_mesh_psf(:))));

RGrad_gpu_x = gpuArray(RGrad(1,:));
RGrad_gpu_y = gpuArray(RGrad(2,:));
RGrad_gpu_z = gpuArray(RGrad(3,:));

RRGrad_gpu_x = gpuArray(RRGrad(1,:));
RRGrad_gpu_y = gpuArray(RRGrad(2,:));
RRGrad_gpu_z = gpuArray(RRGrad(3,:));

RO_gpu = gpuArray( sqrt(dot(POS_TX,POS_TX,1)) + sqrt(dot(POS_RX,POS_RX,1)));

fc_gpu = gpuArray(fc*ones(1,N_t));
BW_gpu = gpuArray(BW*ones(1,N_t));
% wt_gpu = gpuArray(ones(1,N_t)/N_t);
% wf_gpu = gpuArray(ones(1,N_f)/N_f);

w2d_gpu = gpuArray(weight_uniform);


%Run the CUDA Kernel
[y1 y2] = feval(k,IPR_real,IPR_imag,...
x_psf_gpu,y_psf_gpu,z_psf_gpu,...
RGrad_gpu_x,RGrad_gpu_y,RGrad_gpu_z,...
RRGrad_gpu_x,RRGrad_gpu_y,RRGrad_gpu_z,...
RO_gpu,fc_gpu,BW_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
ipr = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
ipr(:) = gather(y1)+j*gather(y2);

%Normalize the PSF
% ipr = ipr./max(abs(ipr(:)));

%AM IPR
w2d_gpu = gpuArray(weight_F);
%Run the CUDA Kernel
[y1 y2] = feval(k,IPR_real,IPR_imag,...
x_psf_gpu,y_psf_gpu,z_psf_gpu,...
RGrad_gpu_x,RGrad_gpu_y,RGrad_gpu_z,...
RRGrad_gpu_x,RRGrad_gpu_y,RRGrad_gpu_z,...
RO_gpu,fc_gpu,BW_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
ipr_am = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
ipr_am(:) = gather(y1)+j*gather(y2);

%Normalize the IPR
% ipr_am = ipr_am./max(abs(ipr_am(:)));

%taylor window version
w2d_gpu = gpuArray(taylorwin_2D_F);

%Re-Run the CUDA Kernel
[y1 y2] = feval(k,IPR_real,IPR_imag,...
x_psf_gpu,y_psf_gpu,z_psf_gpu,...
RGrad_gpu_x,RGrad_gpu_y,RGrad_gpu_z,...
RRGrad_gpu_x,RRGrad_gpu_y,RRGrad_gpu_z,...
RO_gpu,fc_gpu,BW_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
ipr_taylor = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
ipr_taylor(:) = gather(y1)+j*gather(y2);

%Normalize the PSF
% ipr_taylor = ipr_taylor./max(abs(ipr_taylor(:)));


%%%%%%
%Use the frequency agile parameters
w2d_gpu = gpuArray(weight_uniform);
fc_fm_gpu = gpuArray(fc_fm);
BW_fm_gpu = gpuArray(BW_fm);

%Re-Run the CUDA Kernel with new frequencies
[y1 y2] = feval(k,IPR_real,IPR_imag,...
x_psf_gpu,y_psf_gpu,z_psf_gpu,...
RGrad_gpu_x,RGrad_gpu_y,RGrad_gpu_z,...
RRGrad_gpu_x,RRGrad_gpu_y,RRGrad_gpu_z,...
RO_gpu,fc_fm_gpu,BW_fm_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
ipr_fm = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
ipr_fm(:) = gather(y1)+j*gather(y2);

%Normalize the PSF
% ipr_fm = ipr_fm./max(abs(ipr_fm(:)));

%AMFM IPR
w2d_gpu = gpuArray(weight_G);

%Re-Run the CUDA Kernel with new frequencies
[y1 y2] = feval(k,IPR_real,IPR_imag,...
x_psf_gpu,y_psf_gpu,z_psf_gpu,...
RGrad_gpu_x,RGrad_gpu_y,RGrad_gpu_z,...
RRGrad_gpu_x,RRGrad_gpu_y,RRGrad_gpu_z,...
RO_gpu,fc_fm_gpu,BW_fm_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
ipr_amfm = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
ipr_amfm(:) = gather(y1)+j*gather(y2);

%Normalize the PSF
% ipr_amfm = ipr_amfm./max(abs(ipr_amfm(:)));

%taylor window with agile frequency version
w2d_gpu = gpuArray(taylorwin_2D_G);

%Re-Run the CUDA Kernel with new frequencies
[y1 y2] = feval(k,IPR_real,IPR_imag,...
x_psf_gpu,y_psf_gpu,z_psf_gpu,...
RGrad_gpu_x,RGrad_gpu_y,RGrad_gpu_z,...
RRGrad_gpu_x,RRGrad_gpu_y,RRGrad_gpu_z,...
RO_gpu,fc_fm_gpu,BW_fm_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_psf*N_psf));

%allocate system memory for output 
ipr_fm_taylor = zeros(N_psf,N_psf);

%Pull the output data from the GPU's memory
ipr_fm_taylor(:) = gather(y1)+j*gather(y2);

%Normalize the PSF
% ipr_fm_taylor = ipr_fm_taylor./max(abs(ipr_fm_taylor(:)));

%%%%%%%%%%%%%%%%%%%%%%%%%%

display('CUDA Phase History');

%Clear the GPU
reset(device);

% Problem size
Ntot = N_f * N_t;

%prep the kernel
k = parallel.gpu.CUDAKernel('PhaseHist_Kernel.ptx','PhaseHist_Kernel.cu');
cap = min(device.MaxThreadsPerBlock, k.MaxThreadsPerBlock);  % cap = 896 here
warp = 32;

% Choose a good block size (portable, efficient)
N1 = min(256, cap);                     % 256 is a safe default
N1 = warp * floor(N1/warp);             % enforce multiple of 32
if N1 == 0, N1 = warp; end

% Grid size to cover everything; kernel already strides by blockDim*gridDim
N2 = ceil(Ntot / N1);

k.ThreadBlockSize = [N1, 1, 1];
k.GridSize        = [N2, 1, 1];

%allocate Memory for output
PH_real = gpuArray(zeros(N_f*N_t,1));
PH_imag = gpuArray(zeros(N_f*N_t,1));

%Upload data to GPU memory
tgt_gpu_x = gpuArray(tgt_x);
tgt_gpu_y = gpuArray(tgt_y);
tgt_gpu_z = gpuArray(tgt_z);
tgt_gpu_rho = gpuArray(tgt_rho);

Pos_TX_gpu_x = gpuArray(POS_TX(1,:));
Pos_TX_gpu_y = gpuArray(POS_TX(2,:));
Pos_TX_gpu_z = gpuArray(POS_TX(3,:));

Pos_RX_gpu_x = gpuArray(POS_RX(1,:));
Pos_RX_gpu_y = gpuArray(POS_RX(2,:));
Pos_RX_gpu_z = gpuArray(POS_RX(3,:));

Vel_TX_gpu_x = gpuArray(Vel_TX(1,:));
Vel_TX_gpu_y = gpuArray(Vel_TX(2,:));
Vel_TX_gpu_z = gpuArray(Vel_TX(3,:));

Vel_RX_gpu_x = gpuArray(Vel_RX(1,:));
Vel_RX_gpu_y = gpuArray(Vel_RX(2,:));
Vel_RX_gpu_z = gpuArray(Vel_RX(3,:));

fc_gpu = gpuArray(fc*ones(1,N_t));
BW_gpu = gpuArray(BW*ones(1,N_t));

%Don't normalize the weights for phase history
%We will when we generate the SAR image
% wt_gpu = gpuArray(ones(1,N_t));
% wf_gpu = gpuArray(ones(1,N_f));
w2d_gpu = gpuArray(ones(N_f,N_t));
%Run the kernel
[y1 y2] = feval(k,PH_real,PH_imag,...
tgt_gpu_x,tgt_gpu_y,tgt_gpu_z,tgt_gpu_rho,...
Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...
fc_gpu,BW_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(length(tgt_x)));

Phase_hist = zeros(N_t,N_f);
Phase_hist(:) = gather(y1)+j*gather(y2);

%Change the frequency profile
fc_fm_gpu = gpuArray(fc_fm);
BW_fm_gpu = gpuArray(BW_fm);

%Run the kernel again
[y1 y2] = feval(k,PH_real,PH_imag,...
tgt_gpu_x,tgt_gpu_y,tgt_gpu_z,tgt_gpu_rho,...
Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...
fc_fm_gpu,BW_fm_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(length(tgt_x)));

Phase_hist_fm = zeros(N_t,N_f);
Phase_hist_fm(:) = gather(y1)+j*gather(y2);

%%%%%%%%%%%%%%%%%%%%%%%%%%

display('CUDA MF SAR Image Formation');

%Clear the GPU
reset(device);

%prep the kernel
k = parallel.gpu.CUDAKernel('MF_Kernel.ptx','MF_Kernel.cu');
N1=512;
N2=2^ceil(log( N_x*N_x/N1)/log(2));
N2=max([N2 1]);
k.ThreadBlockSize = [N1,1,1];
k.GridSize=[N2,1,1];

%allocate memory on GPU for output
SAR_real = gpuArray(zeros(N_x,N_x));
SAR_imag = gpuArray(zeros(N_x,N_x));

%push data onto GPU memory
PH_real = gpuArray(real(Phase_hist(:)));
PH_imag = gpuArray(imag(Phase_hist(:)));

x_gpu = gpuArray(x_mesh(:));
y_gpu = gpuArray(y_mesh(:));
z_gpu = gpuArray(zeros(size(x_mesh(:))));

Pos_TX_gpu_x = gpuArray(POS_TX(1,:));
Pos_TX_gpu_y = gpuArray(POS_TX(2,:));
Pos_TX_gpu_z = gpuArray(POS_TX(3,:));

Pos_RX_gpu_x = gpuArray(POS_RX(1,:));
Pos_RX_gpu_y = gpuArray(POS_RX(2,:));
Pos_RX_gpu_z = gpuArray(POS_RX(3,:));

Vel_TX_gpu_x = gpuArray(Vel_TX(1,:));
Vel_TX_gpu_y = gpuArray(Vel_TX(2,:));
Vel_TX_gpu_z = gpuArray(Vel_TX(3,:));

Vel_RX_gpu_x = gpuArray(Vel_RX(1,:));
Vel_RX_gpu_y = gpuArray(Vel_RX(2,:));
Vel_RX_gpu_z = gpuArray(Vel_RX(3,:));


fc_gpu = gpuArray(fc*ones(1,N_t));
BW_gpu = gpuArray(BW*ones(1,N_t));

% wt_gpu = gpuArray(ones(1,N_t)/N_t);
% wf_gpu = gpuArray(ones(1,N_f)/N_f);

w2d_gpu = gpuArray(weight_uniform);


%run the kernel
[y1 y2] = feval(k,SAR_real,SAR_imag,PH_real,PH_imag,...
x_gpu,y_gpu,z_gpu,...
Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...
fc_gpu,BW_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_x*N_x));

sar = zeros(N_x,N_x);
sar(:) = gather(y1)+j*gather(y2);
% sar = sar/max(abs(sar(:)));

%AM SAR
w2d_gpu = gpuArray(weight_F);

%Re-run the kernel
[y1 y2] = feval(k,SAR_real,SAR_imag,PH_real,PH_imag,...
x_gpu,y_gpu,z_gpu,...
Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...
fc_gpu,BW_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_x*N_x));

sar_am = zeros(N_x,N_x);
sar_am(:) = gather(y1)+j*gather(y2);
% sar_am = sar_am/max(abs(sar_am(:)));

%%%%%%%
w2d_gpu = gpuArray(taylorwin_2D_F);

%run the kernel
[y1 y2] = feval(k,SAR_real,SAR_imag,PH_real,PH_imag,...
x_gpu,y_gpu,z_gpu,...
Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...
fc_gpu,BW_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_x*N_x));

sar_taylor = zeros(N_x,N_x);
sar_taylor(:) = gather(y1)+j*gather(y2);
% sar_taylor = sar_taylor/max(abs(sar_taylor(:)));

%Update phase history and frequencies
PH_real = gpuArray(real(Phase_hist_fm(:)));
PH_imag = gpuArray(imag(Phase_hist_fm(:)));
fc_fm_gpu = gpuArray(fc_fm);
BW_fm_gpu = gpuArray(BW_fm);
w2d_gpu = gpuArray(weight_uniform);


%run the kernel
[y1 y2] = feval(k,SAR_real,SAR_imag,PH_real,PH_imag,...
x_gpu,y_gpu,z_gpu,...
Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...
fc_fm_gpu,BW_fm_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_x*N_x));

sar_fm = zeros(N_x,N_x);
sar_fm = gather(y1)+j*gather(y2);
% sar_fm = sar_fm/max(abs(sar_fm(:)));

%AMFM SAR image
w2d_gpu = gpuArray(weight_G);
%re-run the kernel
[y1 y2] = feval(k,SAR_real,SAR_imag,PH_real,PH_imag,...
x_gpu,y_gpu,z_gpu,...
Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...
fc_fm_gpu,BW_fm_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_x*N_x));

sar_amfm = zeros(N_x,N_x);
sar_amfm = gather(y1)+j*gather(y2);
% sar_amfm = sar_amfm/max(abs(sar_amfm(:)));


%%%%%%%%%
w2d_gpu = gpuArray(taylorwin_2D_G);

%run the kernel
[y1 y2] = feval(k,SAR_real,SAR_imag,PH_real,PH_imag,...
x_gpu,y_gpu,z_gpu,...
Pos_TX_gpu_x,Pos_TX_gpu_y,Pos_TX_gpu_z,...
Pos_RX_gpu_x,Pos_RX_gpu_y,Pos_RX_gpu_z,...
Vel_TX_gpu_x,Vel_TX_gpu_y,Vel_TX_gpu_z,...
Vel_RX_gpu_x,Vel_RX_gpu_y,Vel_RX_gpu_z,...
fc_fm_gpu,BW_fm_gpu,w2d_gpu,...
int32(N_t),int32(N_f),int32(N_x*N_x));

sar_fm_taylor = zeros(N_x,N_x);
sar_fm_taylor = gather(y1)+j*gather(y2);
% sar_fm_taylor = sar_fm_taylor/max(abs(sar_fm_taylor(:)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1D metrics
display('Image IPR metrics')
%Resolution
meas_res_XR = fminbnd(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)),0,1.75*res_XR);
meas_res_XR_am = fminbnd(@(d) abs(IPR_1d(weight_F,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)),0,1.75*res_XR);
meas_res_XR_taylor = fminbnd(@(d) abs(IPR_1d(taylorwin_2D_F,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)),0,2*res_XR);
meas_res_XD = fminbnd(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)),0,1.75*res_XD);
meas_res_XD_am = fminbnd(@(d) abs(IPR_1d(weight_F,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)),0,1.75*res_XD);
meas_res_XD_taylor = fminbnd(@(d) abs(IPR_1d(taylorwin_2D_F,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)),0,2*res_XD);

meas_res_XR_fm = fminbnd(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)),0,1.75*res_XR);
meas_res_XR_amfm = fminbnd(@(d) abs(IPR_1d(weight_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)),0,1.75*res_XR);
meas_res_XR_fm_taylor = fminbnd(@(d) abs(IPR_1d(taylorwin_2D_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)),0,2*res_XR);
meas_res_XD_fm = fminbnd(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)),0,1.75*res_XD);
meas_res_XD_amfm = fminbnd(@(d) abs(IPR_1d(weight_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)),0,1.75*res_XD);
meas_res_XD_fm_taylor = fminbnd(@(d) abs(IPR_1d(taylorwin_2D_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)),0,2*res_XD);


%Peak to Sidelobe ratio (PSLR)
[SL_XR_loc,PSLR_XR]                 = fminbnd(@(d) 1./abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)),meas_res_XR,2*meas_res_XR);
[SL_XR_loc_am,PSLR_XR_am]           = fminbnd(@(d) 1./abs(IPR_1d(weight_F,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)),meas_res_XR_am,2*meas_res_XR_am);
[SL_XR_loc_taylor,PSLR_XR_taylor]   = fminbnd(@(d) 1./abs(IPR_1d(taylorwin_2D_F,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)),meas_res_XR_taylor,2*meas_res_XR_taylor);
[SL_XD_loc,PSLR_XD]                 = fminbnd(@(d) 1./abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)),meas_res_XD,2*meas_res_XD);
[SL_XD_loc_am,PSLR_XD_am]           = fminbnd(@(d) 1./abs(IPR_1d(weight_F,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)),meas_res_XD_am,2*meas_res_XD_am);
[SL_XD_loc_taylor,PSLR_XD_taylor]   = fminbnd(@(d) 1./abs(IPR_1d(taylorwin_2D_F,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)),meas_res_XD_taylor,2*meas_res_XD_taylor);

[SL_XR_loc_fm,PSLR_XR_fm]                 = fminbnd(@(d) 1./abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)),meas_res_XR_fm,2*meas_res_XR_fm);
[SL_XR_loc_amfm,PSLR_XR_amfm]             = fminbnd(@(d) 1./abs(IPR_1d(weight_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)),meas_res_XR_amfm,2*meas_res_XR_amfm);
[SL_XR_loc_fm_taylor,PSLR_XR_fm_taylor]   = fminbnd(@(d) 1./abs(IPR_1d(taylorwin_2D_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)),meas_res_XR_fm_taylor,2*meas_res_XR_fm_taylor);
[SL_XD_loc_fm,PSLR_XD_fm]                 = fminbnd(@(d) 1./abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)),meas_res_XD_fm,2*meas_res_XD_fm);
[SL_XD_loc_amfm,PSLR_XD_amfm]             = fminbnd(@(d) 1./abs(IPR_1d(weight_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)),meas_res_XD_amfm,2*meas_res_XD_amfm);
[SL_XD_loc_fm_taylor,PSLR_XD_fm_taylor]   = fminbnd(@(d) 1./abs(IPR_1d(taylorwin_2D_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)),meas_res_XD_fm_taylor,2*meas_res_XD_fm_taylor);

%integrated sidelobe ratio (ISLR)
SL_int_XR = integral(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)).^2,meas_res_XR,10*meas_res_XR);
ML_int_XR = integral(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)).^2,0,meas_res_XR);
ISLR_XR_dB = 10*log10(SL_int_XR/ML_int_XR);

SL_int_XR_am = integral(@(d) abs(IPR_1d(weight_F,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)).^2,meas_res_XR_am,10*meas_res_XR_am);
ML_int_XR_am = integral(@(d) abs(IPR_1d(weight_F,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)).^2,0,meas_res_XR_am);
ISLR_XR_am_dB = 10*log10(SL_int_XR_am/ML_int_XR_am);


SL_int_XR_taylor = integral(@(d) abs(IPR_1d(taylorwin_2D_F,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)).^2,meas_res_XR_taylor,10*meas_res_XR_taylor);
ML_int_XR_taylor = integral(@(d) abs(IPR_1d(taylorwin_2D_F,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d)).^2,0,meas_res_XR_taylor);
ISLR_XR_taylor_dB = 10*log10(SL_int_XR_taylor/ML_int_XR_taylor);

SL_int_XD = integral(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)).^2,meas_res_XD,10*meas_res_XD);
ML_int_XD = integral(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)).^2,0,meas_res_XD);
ISLR_XD_dB = 10*log10(SL_int_XD/ML_int_XD);

SL_int_XD_am = integral(@(d) abs(IPR_1d(weight_F,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)).^2,meas_res_XD_am,10*meas_res_XD_am);
ML_int_XD_am = integral(@(d) abs(IPR_1d(weight_F,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)).^2,0,meas_res_XD_am);
ISLR_XD_am_dB = 10*log10(SL_int_XD_am/ML_int_XD_am);

SL_int_XD_taylor = integral(@(d) abs(IPR_1d(taylorwin_2D_F,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)).^2,meas_res_XD_taylor,10*meas_res_XD_taylor);
ML_int_XD_taylor = integral(@(d) abs(IPR_1d(taylorwin_2D_F,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d)).^2,0,meas_res_XD_taylor);
ISLR_XD_taylor_dB = 10*log10(SL_int_XD_taylor/ML_int_XD_taylor);

SL_int_XR_fm = integral(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)).^2,meas_res_XR_fm,10*meas_res_XR_fm);
ML_int_XR_fm = integral(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)).^2,0,meas_res_XR_fm);
ISLR_XR_fm_dB = 10*log10(SL_int_XR_fm/ML_int_XR_fm);

SL_int_XR_amfm = integral(@(d) abs(IPR_1d(weight_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)).^2,meas_res_XR_amfm,10*meas_res_XR_amfm);
ML_int_XR_amfm = integral(@(d) abs(IPR_1d(weight_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)).^2,0,meas_res_XR_amfm);
ISLR_XR_amfm_dB = 10*log10(SL_int_XR_amfm/ML_int_XR_amfm);

SL_int_XR_fm_taylor = integral(@(d) abs(IPR_1d(taylorwin_2D_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)).^2,meas_res_XR_fm_taylor,10*meas_res_XR_fm_taylor);
ML_int_XR_fm_taylor = integral(@(d) abs(IPR_1d(taylorwin_2D_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d)).^2,0,meas_res_XR_fm_taylor);
ISLR_XR_fm_taylor_dB = 10*log10(SL_int_XR_fm_taylor/ML_int_XR_fm_taylor);


SL_int_XD_fm = integral(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)).^2,meas_res_XD_fm,10*meas_res_XD_fm);
ML_int_XD_fm = integral(@(d) abs(IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)).^2,0,meas_res_XD_fm);
ISLR_XD_fm_dB = 10*log10(SL_int_XD_fm/ML_int_XD_fm);

SL_int_XD_amfm = integral(@(d) abs(IPR_1d(weight_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)).^2,meas_res_XD_amfm,10*meas_res_XD_amfm);
ML_int_XD_amfm = integral(@(d) abs(IPR_1d(weight_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)).^2,0,meas_res_XD_amfm);
ISLR_XD_amfm_dB = 10*log10(SL_int_XD_amfm/ML_int_XD_amfm);


SL_int_XD_fm_taylor = integral(@(d) abs(IPR_1d(taylorwin_2D_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)).^2,meas_res_XD_fm_taylor,10*meas_res_XD_fm_taylor);
ML_int_XD_fm_taylor = integral(@(d) abs(IPR_1d(taylorwin_2D_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d)).^2,0,meas_res_XD_fm_taylor);
ISLR_XD_fm_taylor_dB = 10*log10(SL_int_XD_fm_taylor/ML_int_XD_fm_taylor);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save some 1D examples for plots
display('1D plots')

%the taylor windows are going to have the lowest resolution.
%I want to generate 1D PSFs/IPRs with the same spacing.
max_d = max([meas_res_XR_taylor meas_res_XD_taylor meas_res_XR_fm_taylor meas_res_XD_fm_taylor]);

d = linspace(0,10*max_d,2^12);

%IPRs
ipr_XR          = IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d);
ipr_XR_am       = IPR_1d(weight_F,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d);
ipr_XR_taylor   = IPR_1d(taylorwin_2D_F,RGrad,RRGrad,Bi_Range,freq_mesh,XR_vec,d);
ipr_XD          = IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d);
ipr_XD_am       = IPR_1d(weight_F,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d);
ipr_XD_taylor   = IPR_1d(taylorwin_2D_F,RGrad,RRGrad,Bi_Range,freq_mesh,XD_vec,d);

ipr_XR_fm          = IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d);
ipr_XR_amfm        = IPR_1d(weight_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d);
ipr_XR_fm_taylor   = IPR_1d(taylorwin_2D_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XR_fm_vec,d);
ipr_XD_fm          = IPR_1d(weight_uniform,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d);
ipr_XD_amfm        = IPR_1d(weight_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d);
ipr_XD_fm_taylor   = IPR_1d(taylorwin_2D_G,RGrad,RRGrad,Bi_Range,freq_fm_mesh,XD_fm_vec,d);

%PSFs
psf_XR          = PSF_1d(weight_uniform,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_mesh,XR_vec,[0;0;0],d);
psf_XR_am       = PSF_1d(weight_F,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_mesh,XR_vec,[0;0;0],d);
psf_XR_taylor   = PSF_1d(taylorwin_2D_F,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_mesh,XR_vec,[0;0;0],d);
psf_XD          = PSF_1d(weight_uniform,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_mesh,XD_vec,[0;0;0],d);
psf_XD_am       = PSF_1d(weight_F,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_mesh,XD_vec,[0;0;0],d);
psf_XD_taylor   = PSF_1d(taylorwin_2D_F,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_mesh,XD_vec,[0;0;0],d);

psf_XR_fm          = PSF_1d(weight_uniform,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_fm_mesh,XR_fm_vec,[0;0;0],d);
psf_XR_amfm        = PSF_1d(weight_G,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_fm_mesh,XR_fm_vec,[0;0;0],d);
psf_XR_fm_taylor   = PSF_1d(taylorwin_2D_G,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_fm_mesh,XR_fm_vec,[0;0;0],d);
psf_XD_fm          = PSF_1d(weight_uniform,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_fm_mesh,XD_fm_vec,[0;0;0],d);
psf_XD_amfm        = PSF_1d(weight_G,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_fm_mesh,XD_fm_vec,[0;0;0],d);
psf_XD_fm_taylor   = PSF_1d(taylorwin_2D_G,POS_TX,POS_RX,Vel_TX,Vel_RX,freq_fm_mesh,XD_fm_vec,[0;0;0],d);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Save simulation results:

%save initial sim parms
data.parms = parms;

%Slow-time 
data.T_c = T_c;
data.t_max = t_max;
data.t_min = t_min;
data.t = t;


%waveform frequency 
data.freq = freq;
data.fc = fc;
data.lambda_c = lambda_c;
data.BW = BW;
data.fc_fm = fc_fm;
data.BW_fm = BW_fm;
data.d_fc_fm = d_fc_fm;
data.d_BW_fm = d_BW_fm;
data.freq_mesh = freq_mesh;
data.freq_fm_mesh = freq_fm_mesh;

%platform geometry
data.RGrad = RGrad;
data.RGrad_GP = RGrad_GP;
data.RRGrad = RRGrad;
data.RRGrad_GP = RRGrad_GP;


data.POS_TX = POS_TX;
data.POS_RX = POS_RX;
data.Vel_TX = Vel_TX;
data.Vel_RX = Vel_RX;
data.LOS_TX = LOS_TX;
data.LOS_RX = LOS_RX;
data.d_LOS_TX = d_LOS_TX;
data.d_LOS_RX = d_LOS_RX;

data.LOS_TX_0 = LOS_TX_0;
data.POS_TX_0 = POS_TX_0;
data.LOS_RX_0 = LOS_RX_0;
data.POS_RX_0 = POS_RX_0;


%K-space surface positions
data.F_x = F_x;
data.F_y = F_y;
data.F_z = F_z;
data.G_x = G_x;
data.G_y = G_y;
data.G_z = G_z;

%K-space modulation vectors
data.Ft_x = Ft_x;
data.Ft_y = Ft_y;
data.Ft_z = Ft_z;
data.Gt_x = Gt_x;
data.Gt_y = Gt_y;
data.Gt_z = Gt_z;
data.Ff_x = Ff_x;
data.Ff_y = Ff_y;
data.Ff_z = Ff_z;
data.Gf_x = Gf_x;
data.Gf_y = Gf_y;
data.Gf_z = Gf_z;

%Project surfaces in R and XR directions
data.F_r = F_r;
data.F_xr = F_xr;
data.Ft_r = Ft_r;
data.Ft_xr = Ft_xr;
data.Ff_r = Ff_r;
data.Ff_xr = Ff_xr;

data.G_r   = G_r;
data.G_xr  = G_xr;
data.Gt_r  = Gt_r;
data.Gt_xr = Gt_xr;
data.Gf_r  = Gf_r;
data.Gf_xr = Gf_xr;

%Weight functions
data.weight_uniform = weight_uniform;
data.weight_F = weight_F;
data.weight_G = weight_G;
data.taylorwin_2D_F = taylorwin_2D_F;
data.taylorwin_2D_G = taylorwin_2D_G;


%K-space averages
data.AVG_F = AVG_F;
data.AVG_G = AVG_G; 
data.AVG_Ft = AVG_Ft;
data.AVG_Gt = AVG_Gt;
data.AVG_Ff = AVG_Ff;
data.AVG_Gf = AVG_Gf;
data.R_vec = R_vec;
data.R_fm_vec = R_fm_vec;
data.D_vec = D_vec;
data.D_fm_vec = D_fm_vec;
data.N_vec = N_vec;
data.N_fm_vec = N_fm_vec;
data.XR_vec = XR_vec;
data.XR_fm_vec = XR_fm_vec;
data.XD_vec = XD_vec;
data.XD_fm_vec = XD_fm_vec;
data.BW_XR = BW_XR;
data.BW_XD = BW_XD;
data.BW_R = BW_R;
data.BW_XR_fm = BW_XR_fm;
data.BW_XD_fm = BW_XD_fm;
data.BW_R_fm = BW_R_fm;
data.res_XR = res_XR;
data.res_XD = res_XD;
data.res_R = res_R;
data.res_XR_fm = res_XR_fm;
data.res_XD_fm = res_XD_fm;
data.res_R_fm  = res_R_fm;

%SAR image pixel positions 
data.r = r;
data.xr = xr;

%PSF/IPR pixel positions 
data.r_psf = r_psf;
data.xr_psf = xr_psf;

%PSFs from CUDA sim
data.psf = psf;
data.psf_am = psf_am;
data.psf_taylor = psf_taylor;
data.psf_fm = psf_fm;
data.psf_amfm = psf_amfm;
data.psf_fm_taylor = psf_fm_taylor;

%IPRs from CUDA sim
data.ipr = ipr;
data.ipr_am = ipr_am;
data.ipr_taylor = ipr_taylor;
data.ipr_fm = ipr_fm;
data.ipr_amfm = ipr_amfm;
data.ipr_fm_taylor = ipr_fm_taylor;

%Uncompressed phase history data (s_{RX}(t,f))
data.Phase_hist = Phase_hist;
data.Phase_hist_fm = Phase_hist_fm;

%SAR images
data.sar = sar;
data.sar_am = sar_am;
data.sar_taylor = sar_taylor;
data.sar_fm = sar_fm;
data.sar_amfm = sar_amfm;
data.sar_fm_taylor = sar_fm_taylor;

%Imaging performace metrics 
%Resolution
data.meas_res_XR = meas_res_XR;
data.meas_res_XR_am = meas_res_XR_am;
data.meas_res_XR_taylor = meas_res_XR_taylor;
data.meas_res_XD = meas_res_XD;
data.meas_res_XD_am = meas_res_XD_am;
data.meas_res_XD_taylor = meas_res_XD_taylor;
data.meas_res_XR_fm = meas_res_XR_fm;
data.meas_res_XR_amfm = meas_res_XR_amfm;
data.meas_res_XR_fm_taylor = meas_res_XR_fm_taylor;
data.meas_res_XD_fm = meas_res_XD_fm;
data.meas_res_XD_amfm = meas_res_XD_amfm;
data.meas_res_XD_fm_taylor = meas_res_XD_fm_taylor;

%Peak to sidelobe ratio (PSLR) 
data.SL_XR_loc = SL_XR_loc;
data.PSLR_XR_dB = 20*log10(PSLR_XR);
data.SL_XR_loc_am = SL_XR_loc_am;
data.PSLR_XR_am_dB = 20*log10(PSLR_XR_am);
data.SL_XR_loc_taylor = SL_XR_loc_taylor;
data.PSLR_XR_taylor_dB = 20*log10(PSLR_XR_taylor);
data.SL_XD_loc = SL_XD_loc;
data.PSLR_XD_dB = 20*log10(PSLR_XD);
data.SL_XD_loc_am = SL_XD_loc_am;
data.PSLR_XD_am_dB = 20*log10(PSLR_XD_am);
data.SL_XD_loc_taylor = SL_XD_loc_taylor;
data.PSLR_XD_taylor_dB = 20*log10(PSLR_XD_taylor);
data.SL_XR_loc_fm = SL_XR_loc_fm;
data.PSLR_XR_fm_dB = 20*log10(PSLR_XR_fm);
data.SL_XR_loc_amfm = SL_XR_loc_amfm;
data.PSLR_XR_amfm_dB = 20*log10(PSLR_XR_amfm);
data.SL_XR_loc_fm_taylor = SL_XR_loc_fm_taylor;
data.PSLR_XR_fm_taylor_dB = 20*log10(PSLR_XR_fm_taylor);
data.SL_XD_loc_fm = SL_XD_loc_fm;
data.PSLR_XD_fm_dB = 20*log10(PSLR_XD_fm);
data.SL_XD_loc_amfm = SL_XD_loc_amfm;
data.PSLR_XD_amfm_dB = 20*log10(PSLR_XD_amfm);
data.SL_XD_loc_fm_taylor = SL_XD_loc_fm_taylor;
data.PSLR_XD_fm_taylor_dB = 20*log10(PSLR_XD_fm_taylor);

%Integrated sidelobe ratio
data.ISLR_XR_dB = ISLR_XR_dB;
data.ISLR_XR_am_dB = ISLR_XR_am_dB;
data.ISLR_XR_taylor_dB = ISLR_XR_taylor_dB;
data.ISLR_XD_dB = ISLR_XD_dB;
data.ISLR_XD_am_dB = ISLR_XD_am_dB;
data.ISLR_XD_taylor_dB = ISLR_XD_taylor_dB;
data.ISLR_XR_fm_dB = ISLR_XR_fm_dB;
data.ISLR_XR_amfm_dB = ISLR_XR_amfm_dB;
data.ISLR_XR_fm_taylor_dB = ISLR_XR_fm_taylor_dB;
data.ISLR_XD_fm_dB = ISLR_XD_fm_dB;
data.ISLR_XD_amfm_dB = ISLR_XD_amfm_dB;
data.ISLR_XD_fm_taylor_dB = ISLR_XD_fm_taylor_dB;


data.ipr_XR = ipr_XR;
data.ipr_XR_am = ipr_XR_am;
data.ipr_XR_taylor = ipr_XR_taylor;
data.ipr_XD = ipr_XD;
data.ipr_XD_am = ipr_XD_am;
data.ipr_XD_taylor = ipr_XD_taylor;

data.ipr_XR_fm = ipr_XR_fm;
data.ipr_XR_amfm = ipr_XR_amfm;
data.ipr_XR_fm_taylor = ipr_XR_fm_taylor;
data.ipr_XD_fm = ipr_XD_fm;
data.ipr_XD_amfm = ipr_XD_amfm;
data.ipr_XD_fm_taylor = ipr_XD_fm_taylor;

data.psf_XR = psf_XR;
data.psf_XR_am = psf_XR_am;
data.psf_XR_taylor = psf_XR_taylor;
data.psf_XD = psf_XD;
data.psf_XD_am = psf_XD_am;
data.psf_XD_taylor = psf_XD_taylor;

data.psf_XR_fm = psf_XR_fm;
data.psf_XR_amfm = psf_XR_amfm;
data.psf_XR_fm_taylor = psf_XR_fm_taylor;
data.psf_XD_fm = psf_XD_fm;
data.psf_XD_amfm = psf_XD_amfm;
data.psf_XD_fm_taylor = psf_XD_fm_taylor;

data.d_ipr = d;

%%%%%%%%%%%%%
% Longer flight paths for plotting
data.t2 =  t2;
data.POS_TX2 = POS_TX2;
data.POS_RX2 = POS_RX2;

data.Range_TX = Range_TX;
data.Range_RX = Range_RX;
data.Bi_Range = Bi_Range;

data.RRange_TX = RRange_TX;
data.RRange_RX = RRange_RX;
data.Bi_RRange = Bi_RRange;

%save the output
save(dataFileName, '-struct', 'data');

