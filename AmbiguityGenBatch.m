% AmbiguityGenBatch.m
% Author: John Summerfield
% Note: this is an updated version of the simulation I developed during my
% PhD studies at UMass Dartmouth.  I'm just trying to get all of the
% flightpath types that I have worked with into a single simulation.  I'm 
% only going to simulate point targets.  Also this is a move-stop-move 
% model. I'm going to use a stepped FM waveform in this model.  We can 
% manipulate the frequency steps and add a weighting function that varies
% with frequency steps and slow-time steps.
%Flight Paths:
%I'm going to do two types of flight paths for this simulation set, const
%velocity straight and level, const altitude circular flight paths, conical
%log-spiral (constant squint), and direct constant zero or pi squint 
% (landing profile).

% I'm deleting all of the CUDA Parallel Thread Execution (PTX) files
% Type: Text-based assembly-like code (not binary).
% Purpose: Acts as a virtual instruction set architecture (ISA) for CUDA.
delete('*.ptx');

% I'm building a fresh set of Parallel Thread Execution (PTX) files
% Note: I'm using CUDA 12.4, I had to change stuff to get CUDA 12.5+ to work 
system('/usr/local/cuda-12.4/bin/nvcc -ptx -arch=compute_61 MF_Kernel.cu -o MF_Kernel.ptx')
system('/usr/local/cuda-12.4/bin/nvcc -ptx -arch=compute_61 PhaseHist_Kernel.cu -o PhaseHist_Kernel.ptx')
system('/usr/local/cuda-12.4/bin/nvcc -ptx -arch=compute_61 IPR_Kernel.cu -o IPR_Kernel.ptx')
system('/usr/local/cuda-12.4/bin/nvcc -ptx -arch=compute_61 PSF_Kernel.cu -o PSF_Kernel.ptx')

%Using Cuda 11.8 on the desktop
% system('/usr/local/cuda-11.8/bin/nvcc -ptx -arch=compute_61 MF_Kernel.cu -o MF_Kernel.ptx')
% system('/usr/local/cuda-11.8/bin/nvcc -ptx -arch=compute_61 PhaseHist_Kernel.cu -o PhaseHist_Kernel.ptx')
% system('/usr/local/cuda-11.8/bin/nvcc -ptx -arch=compute_61 IPR_Kernel.cu -o IPR_Kernel.ptx')
% system('/usr/local/cuda-11.8/bin/nvcc -ptx -arch=compute_61 PSF_Kernel.cu -o PSF_Kernel.ptx')

%If you build using CUDA 12.5, I had to manipulate the .ptx files to work
%with MATLAB (Ugg!!!!!!!!)
% system("sed -i 's/\.version 8\.5/.version 8\.4/' MF_Kernel.ptx");
% system("sed -i 's/\.version 8\.5/.version 8\.4/' PhaseHist_Kernel.ptx");
% system("sed -i 's/\.version 8\.5/.version 8\.4/' IPR_Kernel.ptx");
% system("sed -i 's/\.version 8\.5/.version 8\.4/' PSF_Kernel.ptx");



%Profile names
Profile_name_vec = {'Test_1','Test_2','Test_3','Test_4','Test_5','Test_6'}

%geometry parameters
%0 = const vel, 1 = circle path, 2 = log spiral, 3 = Direct
TX_path_type_vec = [1 1 1 1 0 2]; 
RX_path_type_vec = [0 1 2 3 0 2]; 

%squints: I'm goint to use an XYZ coordinate system (East-North-Up if you
%want).  Increasing az angle is going to be a counter-clockwise rotation
%when looking down into the XY plane.  Squint angle is the angle between
%the platform LOS vector realtive to scene center and the platform velocity
%vector.  A positive squint angle results in a left-looking scenario or 
% counter-clockwise rotation. I relize that there are inconsistancies in 
% the definition of squint angle.  
TX_squint_ang_deg_vec   = [90 90 90 90 10 60]; 
RX_squint_ang_deg_vec   = [40 90 30 0 10 20]; 

%Subsonic - 100 mps (Airborne stuff for this round)
%I'm using a move-stop-move model here
%I'm going to make a updated version for higher speed later
TX_speed_mps_vec = [100 100 100 100 100 100];
RX_speed_mps_vec = [100 100 100 100 100 100];

TX_el_ang_deg_vec   = [15 15 15 15 5 15]; 
RX_el_ang_deg_vec   = [30 30 30 30 5 30]; 

TX_range_m_vec   = [20 20 20 20 10 20]*1e3; 
RX_range_m_vec   = [8 8 8 8 10 8]*1e3; 

%I'm going to do a difference in bearing to scene center rather than a
%true bistatic angle (slant plane bistatic angle).  The difference in
%bearing is the ground plane component in bistatic angle.

bearing_diff_deg_vec = [15 45 60 90 0 60];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Center Frequency and Resolution

center_freq_Hz_vec = [10 10 10 10 10 10]*1e9;

Cmd_Res_m = .3048

%Commanded Range Res in the ground plane, meters
%Let the simulation figure out the waveform bandwidth
Cmd_Range_Res_m_GP_vec = [1 1 1 1 1 1]*Cmd_Res_m; %1 ft Res in the ground Plane

%Commanded Cross-Range Res in the ground plane, meters
%Let the simulation figure out the slow-time duration
Cmd_XRange_Res_m_GP_vec = [1 1 1 1 1 1]*Cmd_Res_m; %1 ft Res in the ground Plane


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Scene made of point targets
tgt_ang =  60*pi/180;

tgt(1).r = [0;0;0];
tgt(1).rho = 1;

tgt(2).r = 10*Cmd_Res_m*[cos(-.5*tgt_ang);sin(-.5*tgt_ang);0];
tgt(2).rho = 1;

tgt(3).r = 20*Cmd_Res_m*[cos(-.5*tgt_ang);sin(-.5*tgt_ang);0];
tgt(3).rho = 1;


tgt(4).r = 10*Cmd_Res_m*[cos(.5*tgt_ang);sin(.5*tgt_ang);0];
tgt(4).rho = 1;

tgt(5).r = 20*Cmd_Res_m*[cos(.5*tgt_ang);sin(.5*tgt_ang);0];
tgt(5).rho = 1;

mean_r = [0;0;0];
for ii=1:length(tgt),
    mean_r = mean_r +tgt(ii).r;
end
mean_r = mean_r/length(tgt);
for ii=1:length(tgt),
    tgt(ii).r = tgt(ii).r-mean_r;
end



% for ii= 1:length(TX_path_type_vec),
% for ii= 1:3,
for ii= 4:6,
    %build a parmeter set
    parms.tgt = tgt;
    parms.TX_path_type = TX_path_type_vec(ii);
    parms.RX_path_type = RX_path_type_vec(ii);
    parms.TX_squint_ang_deg = TX_squint_ang_deg_vec(ii);
    parms.RX_squint_ang_deg = RX_squint_ang_deg_vec(ii);
    parms.TX_speed_mps = TX_speed_mps_vec(ii);
    parms.RX_speed_mps = RX_speed_mps_vec(ii);
    parms.TX_el_ang_deg = TX_el_ang_deg_vec(ii);
    parms.RX_el_ang_deg = RX_el_ang_deg_vec(ii);
    parms.TX_range_m = TX_range_m_vec(ii);
    parms.RX_range_m = RX_range_m_vec(ii);
    parms.bearing_diff_deg = bearing_diff_deg_vec(ii);
    parms.center_freq_Hz = center_freq_Hz_vec(ii);
    parms.Cmd_Range_Res_m_GP = Cmd_Range_Res_m_GP_vec(ii);
    parms.Cmd_XRange_Res_m_GP = Cmd_XRange_Res_m_GP_vec(ii);    

    %output filename
    dataFileName = [Profile_name_vec{ii} '.mat']
    
    %run the sim, the output will be saved to the output file
    Sim_Main(dataFileName,parms);
    
        
end
