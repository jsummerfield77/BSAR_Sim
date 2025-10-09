% GenPlots.m
% Author: John Summerfield
% This will generate some plots of the bistatci SAR simulation

close all;
clear all;

%I increased the scene size of the IPR/PSF simulations so that I could get
%better K-space resolution when I used a 2D-FFT to mesure the K-space
%energy density. Set the ipr_scale to some number less than one to zoom in
%on the mainlobe and first few sidelobes of the IPRs/PSFs
ipr_scale = .5; 

% outputPath = './Figures/scenarios_1_2_3';
outputPath = './Figures/scenarios_4_5_6';
if ~exist(outputPath),
    mkdir(outputPath);
end

%Profile names
% Profile_filenames = {'Test_1','Test_2','Test_3'};
Profile_filenames = {'Test_4','Test_5','Test_6'};

for ii=1:length(Profile_filenames),
    profile(ii).dataset = load([Profile_filenames{ii} '.mat']);
end


%Geometry - Flight Paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig_geom1 = figure('units','normalized','outerposition',[0 0 1 1]);
pathWidth1 = .5;
pathWidth2 = 2;
pointSize = 12;

for ii=1:length(Profile_filenames),
    subplot(1,3,ii);
    hold on;

    h1 = plot3(profile(ii).dataset.POS_TX2(1,:)*1e-3,...
        profile(ii).dataset.POS_TX2(2,:)*1e-3,...
        profile(ii).dataset.POS_TX2(3,:)*1e-3,...
        '--k','LineWidth',pathWidth1);

    h1a = plot3(profile(ii).dataset.POS_TX2(1,1)*1e-3,...
        profile(ii).dataset.POS_TX2(2,1)*1e-3,...
        profile(ii).dataset.POS_TX2(3,1)*1e-3,...
        'sk','MarkerSize',pointSize);
    
    h1b = plot3(profile(ii).dataset.POS_TX2(1,end)*1e-3,...
        profile(ii).dataset.POS_TX2(2,end)*1e-3,...
        profile(ii).dataset.POS_TX2(3,end)*1e-3,...
        '^k','MarkerSize',pointSize);

    h2 = plot3(profile(ii).dataset.POS_TX(1,:)*1e-3,...
        profile(ii).dataset.POS_TX(2,:)*1e-3,...
        profile(ii).dataset.POS_TX(3,:)*1e-3,...
        'Color','g','LineWidth',pathWidth2);
    
    h3 = plot3(profile(ii).dataset.POS_RX2(1,:)*1e-3,...
        profile(ii).dataset.POS_RX2(2,:)*1e-3,...
        profile(ii).dataset.POS_RX2(3,:)*1e-3,...
        '-.k','LineWidth',pathWidth1);

    h3a = plot3(profile(ii).dataset.POS_RX2(1,1)*1e-3,...
        profile(ii).dataset.POS_RX2(2,1)*1e-3,...
        profile(ii).dataset.POS_RX2(3,1)*1e-3,...
        'sk','MarkerSize',pointSize);

    h3b = plot3(profile(ii).dataset.POS_RX2(1,end)*1e-3,...
        profile(ii).dataset.POS_RX2(2,end)*1e-3,...
        profile(ii).dataset.POS_RX2(3,end)*1e-3,...
        '^k','MarkerSize',pointSize);

    h4 = plot3(profile(ii).dataset.POS_RX(1,:)*1e-3,...
        profile(ii).dataset.POS_RX(2,:)*1e-3,...
        profile(ii).dataset.POS_RX(3,:)*1e-3,...
        'Color','r','LineWidth',pathWidth2,'LineStyle','--');

    h5 = plot3(0,0,0,'.k','MarkerSize',pointSize);

    axis equal;
    grid on

    legend([h2 h4 h5 h1a h1b],'TX Path','RX Path','Scene center','Start Pos','End Pos');

    xlabel('X km')
    ylabel('Y km')
    title(Profile_filenames{ii});

end

exportgraphics(fig_geom1, [outputPath '/Geometry_FlightPaths.png'], 'Resolution', 300);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%
% More Geometry Stuff
fig_geom2 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

    TX_az = atan2(profile(ii).dataset.LOS_TX(2,:),profile(ii).dataset.LOS_TX(1,:));
    TX_el = atan2(profile(ii).dataset.LOS_TX(3,:),sqrt(profile(ii).dataset.LOS_TX(1,:).^2+profile(ii).dataset.LOS_TX(2,:).^2));

    RX_az = atan2(profile(ii).dataset.LOS_RX(2,:),profile(ii).dataset.LOS_RX(1,:));
    RX_el = atan2(profile(ii).dataset.LOS_RX(3,:),sqrt(profile(ii).dataset.LOS_RX(1,:).^2+profile(ii).dataset.LOS_RX(2,:).^2));  

    RGrad_mag = sqrt(dot(profile(ii).dataset.RGrad,profile(ii).dataset.RGrad,1));
    RGrad_GP_mag = sqrt(dot(profile(ii).dataset.RGrad_GP,profile(ii).dataset.RGrad_GP,1));
    
    RGrad_az = atan2(profile(ii).dataset.RGrad(2,:),profile(ii).dataset.RGrad(1,:));
    RGrad_el = atan2(profile(ii).dataset.RGrad(3,:),sqrt(profile(ii).dataset.RGrad(1,:).^2+profile(ii).dataset.RGrad(2,:).^2));

    RRGrad_mag = sqrt(dot(profile(ii).dataset.RRGrad,profile(ii).dataset.RRGrad,1));
    RRGrad_GP_mag = sqrt(dot(profile(ii).dataset.RRGrad_GP,profile(ii).dataset.RRGrad_GP,1));

    d_LOS_TX_GP_mag = sqrt(profile(ii).dataset.d_LOS_TX(1,:).^2+profile(ii).dataset.d_LOS_TX(2,:).^2);
    d_LOS_RX_GP_mag = sqrt(profile(ii).dataset.d_LOS_RX(1,:).^2+profile(ii).dataset.d_LOS_RX(2,:).^2);

    subplot(3,3,ii);
    hold on;
    h1 = plot(profile(ii).dataset.t,TX_az *180/pi);
    h2 = plot(profile(ii).dataset.t,RX_az *180/pi);
    h3 = plot(profile(ii).dataset.t,RGrad_az*180/pi);

    grid on;
    axis square;
    legend([h1 h2 h3],'TX','RX','$\vec{\nabla R}(t)$','interpreter','latex');
    xlabel('Time s')
    ylabel('Angle Deg')
    title( 'Az Angles vs Time');

    subplot(3,3,ii+3);
    hold on;
    h1 = plot(profile(ii).dataset.t,TX_el *180/pi);
    h2 = plot(profile(ii).dataset.t,RX_el *180/pi);
    h3 = plot(profile(ii).dataset.t,RGrad_el*180/pi);

    grid on;
    axis square;
    legend([h1 h2 h3],'TX','RX','$\vec{\nabla R}(t)$','interpreter','latex');
    xlabel('Time s')
    ylabel('Angle Deg')
    title( 'El Angles vs Time');


    subplot(3,3,ii+6);
    hold on;
    h1 = plot(profile(ii).dataset.t,d_LOS_TX_GP_mag);
    h2 = plot(profile(ii).dataset.t,d_LOS_RX_GP_mag);
    h3 = plot(profile(ii).dataset.t,RRGrad_GP_mag);

    grid on;
    axis square;
    legend([h1 h2 h3],'$|\vec{LOS}_{TX}(t)\times \vec{z}|$','$|\vec{LOS}_{RX}(t)\times \vec{z}|$','$|\vec{\nabla \dot{R}}(t)\times \vec{z}|$','interpreter','latex');
    xlabel('Time s')
    ylabel('Mag')
    title( 'Range Rate Grad Mag vs Time');

end

exportgraphics(fig_geom2, [outputPath '/Geometry_LOS_angles_RRGrad_Mag.png'], 'Resolution', 300);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig_kSpace1 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

subplot(3,3,ii);
psd = 1./profile(ii).dataset.weight_F;
psd = psd/max(psd(:));
contourf(profile(ii).dataset.F_r,profile(ii).dataset.F_xr,psd);
caxis([0 1]);
axis equal;
colorbar;
colormap jet;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF(\vec{K}) \right|$','Interpreter','latex');

subplot(3,3,3+ii);
psd = ones(size(profile(ii).dataset.weight_F));
psd(1,1) = 1+eps*1e9*randn(1,1);
psd(1,end) = 1+eps*1e9*randn(1,1);
psd(end,1) = 1+eps*1e9*randn(1,1);
psd(end,end) = 1+eps*1e9*randn(1,1);
psd = psd/max(abs(psd(:)));
contourf(profile(ii).dataset.F_r,profile(ii).dataset.F_xr,psd);
caxis([0 1]);
axis equal;
colorbar;
colormap jet;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF_{AM}(\vec{K}) \right|$','Interpreter','latex');

subplot(3,3,6+ii);
psd = profile(ii).dataset.taylorwin_2D_F./profile(ii).dataset.weight_F;
psd = psd/max(abs(psd(:)));
contourf(profile(ii).dataset.F_r,profile(ii).dataset.F_xr,psd);
caxis([0 1]);
axis equal;
colorbar;
colormap jet;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF_{Taylor}(\vec{K}) \right|$','Interpreter','latex');

end

exportgraphics(fig_kSpace1, [outputPath '/KSpace_TF.png'], 'Resolution', 300);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig_kSpace2 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

subplot(3,3,ii);
psd = 1./profile(ii).dataset.weight_G;
psd = psd/max(psd(:));
contourf(profile(ii).dataset.G_r,profile(ii).dataset.G_xr,psd);
caxis([0 1]);
axis equal;
colorbar;
colormap jet;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF_{AF}(\vec{K}) \right|$','Interpreter','latex');

subplot(3,3,3+ii);
psd = ones(size(profile(ii).dataset.weight_G));
psd(1,1) = 1+eps*1e9*randn(1,1);
psd(1,end) = 1+eps*1e9*randn(1,1);
psd(end,1) = 1+eps*1e9*randn(1,1);
psd(end,end) = 1+eps*1e9*randn(1,1);
psd = psd/max(abs(psd(:)));
contourf(profile(ii).dataset.G_r,profile(ii).dataset.G_xr,psd);
caxis([0 1]);
axis equal;
colorbar;
colormap jet;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF_{AF,AM}(\vec{K}) \right|$','Interpreter','latex');

subplot(3,3,6+ii);
psd = profile(ii).dataset.taylorwin_2D_G./profile(ii).dataset.weight_G;
psd = psd/max(abs(psd(:)));
contourf(profile(ii).dataset.G_r,profile(ii).dataset.G_xr,psd);
caxis([0 1]);
axis equal;
colorbar;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF_{AF,Taylor}(\vec{K}) \right|$','Interpreter','latex');


end

exportgraphics(fig_kSpace2, [outputPath '/KSpace_AF_TF.png'], 'Resolution', 300);


% Transfer Functions from FFTs
fig_kSpace3 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

subplot(3,3,ii);

N_r = length(profile(ii).dataset.r);
dr = profile(ii).dataset.r_psf(2)-profile(ii).dataset.r_psf(1);
dxr = profile(ii).dataset.xr_psf(2)-profile(ii).dataset.xr_psf(1);
d_kr = 2*pi/dr/(N_r-1);
d_kxr = 2*pi/dxr/(N_r-1);

kr = [-N_r/2:N_r/2-1]*d_kr + ...
    dot(profile(ii).dataset.AVG_F,profile(ii).dataset.R_vec);
kxr = [-N_r/2:N_r/2-1]*d_kxr + ....
    dot(profile(ii).dataset.AVG_F,profile(ii).dataset.XR_vec); %<- This is approx 0

[r_mesh,xr_mesh] = meshgrid(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf);
bb_phasor = exp(-j*(r_mesh*dot(profile(ii).dataset.AVG_F,profile(ii).dataset.R_vec) + xr_mesh* dot(profile(ii).dataset.AVG_F,profile(ii).dataset.XR_vec) ) );

TF = fftshift(fft2(profile(ii).dataset.ipr.*bb_phasor));
TF = TF/max(abs(TF(:)));

%This is for setting the axis
Fr_width = max(profile(ii).dataset.F_r(:))-min(profile(ii).dataset.F_r(:));
Fxr_width = max(profile(ii).dataset.F_xr(:))-min(profile(ii).dataset.F_xr(:));
k_width = 1.05*max([Fr_width Fxr_width]);
axis_TF(1) = dot(profile(ii).dataset.AVG_F,profile(ii).dataset.R_vec)-.5*k_width;
axis_TF(2) = dot(profile(ii).dataset.AVG_F,profile(ii).dataset.R_vec)+.5*k_width;
axis_TF(3) = dot(profile(ii).dataset.AVG_F,profile(ii).dataset.XR_vec)-.5*k_width;
axis_TF(4) = dot(profile(ii).dataset.AVG_F,profile(ii).dataset.XR_vec)+.5*k_width;

imagesc(kr,kxr,abs(TF),[0 1]);
axis equal xy tight;
axis(axis_TF);
colorbar;
colormap jet;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF(\vec{K}) \right|$','Interpreter','latex');

subplot(3,3,3+ii);

TF_AM = fftshift(fft2(profile(ii).dataset.ipr_am.*bb_phasor));
TF_AM = TF_AM/max(abs(TF_AM(:)));

imagesc(kr,kxr,abs(TF_AM),[0 1]);
axis equal xy tight;
axis(axis_TF);
colorbar;
colormap jet;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF_{AM}(\vec{K}) \right|$','Interpreter','latex');

subplot(3,3,6+ii);
TF_taylor = fftshift(fft2(profile(ii).dataset.ipr_taylor .*bb_phasor));
TF_taylor = TF_taylor/max(abs(TF_taylor(:)));

imagesc(kr,kxr,abs(TF_taylor),[0 1]);
axis equal xy tight;
axis(axis_TF);
colorbar;
colormap jet;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF_{taylor}(\vec{K}) \right|$','Interpreter','latex');

end

exportgraphics(fig_kSpace3, [outputPath '/TFs_from_FFTs.png'], 'Resolution', 300);

%%%%%%%%%%%%%%

% AF Transfer Functions from FFTs
fig_kSpace4 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

subplot(3,3,ii);

N_r = length(profile(ii).dataset.r);
dr = profile(ii).dataset.r_psf(2)-profile(ii).dataset.r_psf(1);
dxr = profile(ii).dataset.xr_psf(2)-profile(ii).dataset.xr_psf(1);
d_kr = 2*pi/dr/(N_r-1);
d_kxr = 2*pi/dxr/(N_r-1);

kr = [-N_r/2:N_r/2-1]*d_kr + ...
    dot(profile(ii).dataset.AVG_G,profile(ii).dataset.R_fm_vec);
kxr = [-N_r/2:N_r/2-1]*d_kxr + ....
    dot(profile(ii).dataset.AVG_G,profile(ii).dataset.XR_fm_vec); %<- This is approx 0

[r_mesh,xr_mesh] = meshgrid(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf);
bb_phasor = exp(-j*(r_mesh*dot(profile(ii).dataset.AVG_G,profile(ii).dataset.R_fm_vec) + xr_mesh* dot(profile(ii).dataset.AVG_G,profile(ii).dataset.XR_fm_vec) ) );

TF_fm = fftshift(fft2(profile(ii).dataset.ipr_fm.*bb_phasor));
TF_fm = TF_fm/max(abs(TF_fm(:)));

%This is for setting the axis
Gr_width = max(profile(ii).dataset.G_r(:))-min(profile(ii).dataset.G_r(:));
Gxr_width = max(profile(ii).dataset.G_xr(:))-min(profile(ii).dataset.G_xr(:));
k_width = 1.05*max([Gr_width Gxr_width]);
axis_TF(1) = dot(profile(ii).dataset.AVG_G,profile(ii).dataset.R_fm_vec)-.5*k_width;
axis_TF(2) = dot(profile(ii).dataset.AVG_G,profile(ii).dataset.R_fm_vec)+.5*k_width;
axis_TF(3) = dot(profile(ii).dataset.AVG_G,profile(ii).dataset.XR_fm_vec)-.5*k_width;
axis_TF(4) = dot(profile(ii).dataset.AVG_G,profile(ii).dataset.XR_fm_vec)+.5*k_width;

imagesc(kr,kxr,abs(TF_fm),[0 1]);
axis equal xy tight;
axis(axis_TF);
colorbar;
colormap jet;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF_{fm}(\vec{K}) \right|$','Interpreter','latex');

subplot(3,3,3+ii);

TF_AMFM = fftshift(fft2(profile(ii).dataset.ipr_amfm.*bb_phasor));
TF_AMFM = TF_AMFM/max(abs(TF_AMFM(:)));

imagesc(kr,kxr,abs(TF_AMFM),[0 1]);
axis equal xy tight;
axis(axis_TF);
colorbar;
colormap jet;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF_{AF,AM}(\vec{K}) \right|$','Interpreter','latex');

subplot(3,3,6+ii);
TF_FM_taylor = fftshift(fft2(profile(ii).dataset.ipr_fm_taylor .*bb_phasor));
TF_FM_taylor = TF_FM_taylor/max(abs(TF_FM_taylor(:)));

imagesc(kr,kxr,abs(TF_FM_taylor),[0 1]);
axis equal xy tight;
axis(axis_TF);
colorbar;
colormap jet;

xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF_{AF,taylor}(\vec{K}) \right|$','Interpreter','latex');

end

exportgraphics(fig_kSpace4, [outputPath '/TFs_AF_from_FFTs.png'], 'Resolution', 300);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Transfer Functions and modulation vecs
fig_kSpace5 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

N_t = length(profile(ii).dataset.t);
N_skeep_t = 2^(log(N_t)/log(2) - 2);
N_f = length(profile(ii).dataset.freq);
N_skeep_f = 2^(log(N_f)/log(2)-2);

max_Ft_mag = max(sqrt( profile(ii).dataset.Ft_r(:).^2 + profile(ii).dataset.Ft_xr(:).^2));
max_Ff_mag = max(sqrt( profile(ii).dataset.Ff_r(:).^2 + profile(ii).dataset.Ff_xr(:).^2));

scale_Ft = profile(ii).dataset.BW_XR*N_skeep_t/(N_t*max_Ft_mag);
scale_Ff = profile(ii).dataset.BW_R*N_skeep_f/(N_f*max_Ff_mag);

max_Gt_mag = max(sqrt( profile(ii).dataset.Gt_r(:).^2 + profile(ii).dataset.Gt_xr(:).^2));
max_Gf_mag = max(sqrt( profile(ii).dataset.Gf_r(:).^2 + profile(ii).dataset.Gf_xr(:).^2));

scale_Gt = profile(ii).dataset.BW_XR_fm*N_skeep_t/(N_t*max_Gt_mag);
scale_Gf = profile(ii).dataset.BW_R_fm*N_skeep_f/(N_f*max_Gf_mag);


r1 =  [profile(ii).dataset.F_r(1:end,1).'  profile(ii).dataset.F_r(end,1:end)  profile(ii).dataset.F_r(end:-1:1,end).'  profile(ii).dataset.F_r(1,end:-1:1) ];
xr1 = [profile(ii).dataset.F_xr(1:end,1).' profile(ii).dataset.F_xr(end,1:end) profile(ii).dataset.F_xr(end:-1:1,end).' profile(ii).dataset.F_xr(1,end:-1:1)];

r2 =  [profile(ii).dataset.G_r(1:end,1).'  profile(ii).dataset.G_r(end,1:end)  profile(ii).dataset.G_r(end:-1:1,end).'  profile(ii).dataset.G_r(1,end:-1:1)];
xr2 = [profile(ii).dataset.G_xr(1:end,1).' profile(ii).dataset.G_xr(end,1:end) profile(ii).dataset.G_xr(end:-1:1,end).' profile(ii).dataset.G_xr(1,end:-1:1)];

jj_list = [1:N_skeep_t:N_t N_t];
kk_list = [1:N_skeep_f:N_f N_f];


subplot(2,3,ii);
fill(r1,xr1,[1 0 0],'FaceAlpha',0.5,'EdgeColor','k');
hold on;
for jj_idx=1:length(jj_list),
for kk_idx=1:length(kk_list),
    jj = jj_list(jj_idx);
    kk = kk_list(kk_idx);
     
    F = [profile(ii).dataset.F_r(kk,jj);profile(ii).dataset.F_xr(kk,jj);0];
    Ft =[profile(ii).dataset.Ft_r(kk,jj);profile(ii).dataset.Ft_xr(kk,jj);0];
    Ff =[profile(ii).dataset.Ff_r(kk,jj);profile(ii).dataset.Ff_xr(kk,jj);0];
    quiver(F(1),F(2),scale_Ft*Ft(1),scale_Ft*Ft(2),'color','m');
    quiver(F(1),F(2),scale_Ff*Ff(1),scale_Ff*Ff(2),'color','b');    
end
end
    
axis equal;
grid on;
xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF(\vec{K}) \right|$','Interpreter','latex');

subplot(2,3,ii+3);
fill(r2,xr2,[1 0 0],'FaceAlpha',0.5,'EdgeColor','k');
hold on;

for jj_idx=1:length(jj_list),
for kk_idx=1:length(kk_list),
    jj = jj_list(jj_idx);
    kk = kk_list(kk_idx);
    
    G = [profile(ii).dataset.G_r(kk,jj);profile(ii).dataset.G_xr(kk,jj);0];
    Gt =[profile(ii).dataset.Gt_r(kk,jj);profile(ii).dataset.Gt_xr(kk,jj);0];
    Gf =[profile(ii).dataset.Gf_r(kk,jj);profile(ii).dataset.Gf_xr(kk,jj);0];
    quiver(G(1),G(2),scale_Gt*Gt(1),scale_Gt*Gt(2),'color','m');
    quiver(G(1),G(2),scale_Gf*Gf(1),scale_Gf*Gf(2),'color','b');    
end
end
axis equal;
grid on;
xlabel('K_r rad/m')
ylabel('K_x_r rad/m')
title( '$\left| TF_{fm}(\vec{K}) \right|$','Interpreter','latex');

end

exportgraphics(fig_kSpace5, [outputPath '/K_Space_Vecs.png'], 'Resolution', 300);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IPRs
fig_ipr1 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

subplot(3,3,ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.ipr)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;
colormap jet;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| ipr(\vec{r}) \right|$ dB','Interpreter','latex');

subplot(3,3,3+ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.ipr_am)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| ipr_{AM}(\vec{r}) \right|$ dB','Interpreter','latex');

subplot(3,3,6+ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.ipr_taylor)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;
colormap jet;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| ipr_{Taylor}(\vec{r}) \right|$ dB','Interpreter','latex');

end

exportgraphics(fig_ipr1, [outputPath '/IPRs.png'], 'Resolution', 300);

%%%%%%%%%%
% AF IPRs
fig_ipr2 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

subplot(3,3,ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.ipr_fm)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;
colormap jet;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| ipr_{AF}(\vec{r}) \right|$ dB','Interpreter','latex');

subplot(3,3,3+ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.ipr_amfm)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| ipr_{AF,AM}(\vec{r}) \right|$ dB','Interpreter','latex');

subplot(3,3,6+ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.ipr_fm_taylor)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;
colormap jet;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| ipr_{AF,Taylor}(\vec{r}) \right|$ dB','Interpreter','latex');

end

exportgraphics(fig_ipr2, [outputPath '/IPRs_AF.png'], 'Resolution', 300);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%PSFs
%%%%%%%%%%%%%%%%%%

fig_psf1 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

subplot(3,3,ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.psf)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;
colormap jet;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| psf \left( \vec{O}-\frac{\vec{r}}{2}, \vec{O}+\frac{\vec{r}}{2} \right) \right|$ dB','Interpreter','latex');

subplot(3,3,3+ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.psf_am)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| psf_{AM} \left( \vec{O}-\frac{\vec{r}}{2}, \vec{O}+\frac{\vec{r}}{2} \right) \right|$ dB','Interpreter','latex');

subplot(3,3,6+ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.psf_taylor)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;
colormap jet;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| psf_{Taylor} \left( \vec{O}-\frac{\vec{r}}{2}, \vec{O}+\frac{\vec{r}}{2} \right) \right|$ dB','Interpreter','latex');

end

exportgraphics(fig_psf1, [outputPath '/PSFs.png'], 'Resolution', 300);

%%%%%%%%%%
% AF PSFs
fig_psf2 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

subplot(3,3,ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.psf_fm)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;
colormap jet;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| psf_{AF} \left( \vec{O}-\frac{\vec{r}}{2}, \vec{O}+\frac{\vec{r}}{2} \right) \right|$ dB','Interpreter','latex');

subplot(3,3,3+ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.psf_amfm)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| psf_{AF,AM} \left( \vec{O}-\frac{\vec{r}}{2}, \vec{O}+\frac{\vec{r}}{2} \right) \right|$ dB','Interpreter','latex');

subplot(3,3,6+ii);
imagesc(profile(ii).dataset.r_psf,profile(ii).dataset.xr_psf,20*log10(abs(profile(ii).dataset.psf_fm_taylor)),[-60 0]);
axis equal xy tight;
temp = ipr_scale*axis;
axis(temp);
colorbar;
colormap jet;

xlabel('Range m')
ylabel('XRange m')
title( '$\left| psf_{AF,Taylor} \left( \vec{O}-\frac{\vec{r}}{2}, \vec{O}+\frac{\vec{r}}{2} \right) \right|$ dB','Interpreter','latex');

end

exportgraphics(fig_psf2, [outputPath '/PSFs_AF.png'], 'Resolution', 300);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SAR Images
%%%%%%%%%%

fig_sar1 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

subplot(3,3,ii);
imagesc(profile(ii).dataset.r,profile(ii).dataset.xr,20*log10(abs(profile(ii).dataset.sar)),[-60 0]);
axis equal xy tight;
colorbar;
colormap jet;


xlabel('Range m')
ylabel('XRange m')
title( '$\left| \tilde{\rho} (\vec{r})\right|$ dB','Interpreter','latex');

subplot(3,3,ii+3);
imagesc(profile(ii).dataset.r,profile(ii).dataset.xr,20*log10(abs(profile(ii).dataset.sar_am )),[-60 0]);
axis equal xy tight;
colorbar;
colormap jet;


xlabel('Range m')
ylabel('XRange m')
title( '$\left| \tilde{\rho}_{AM} (\vec{r})\right|$ dB','Interpreter','latex');

subplot(3,3,ii+6);
imagesc(profile(ii).dataset.r,profile(ii).dataset.xr,20*log10(abs(profile(ii).dataset.sar_taylor )),[-60 0]);
axis equal xy tight;
colorbar;
colormap jet;


xlabel('Range m')
ylabel('XRange m')
title( '$\left| \tilde{\rho}_{Taylor} (\vec{r})\right|$ dB','Interpreter','latex');

end
exportgraphics(fig_sar1, [outputPath '/SAR.png'], 'Resolution', 300);


%%%%%%%%
fig_sar2 = figure('units','normalized','outerposition',[0 0 1 1]);
%K-space passbands
for ii=1:length(Profile_filenames),

subplot(3,3,ii);
imagesc(profile(ii).dataset.r,profile(ii).dataset.xr,20*log10(abs(profile(ii).dataset.sar_fm)),[-60 0]);
axis equal xy tight;
colorbar;
colormap jet;


xlabel('Range m')
ylabel('XRange m')
title( '$\left| \tilde{\rho}_{AF} (\vec{r})\right|$ dB','Interpreter','latex');

subplot(3,3,ii+3);
imagesc(profile(ii).dataset.r,profile(ii).dataset.xr,20*log10(abs(profile(ii).dataset.sar_amfm )),[-60 0]);
axis equal xy tight;
colorbar;
colormap jet;


xlabel('Range m')
ylabel('XRange m')
title( '$\left| \tilde{\rho}_{AF,AM} (\vec{r})\right|$ dB','Interpreter','latex');

subplot(3,3,ii+6);
imagesc(profile(ii).dataset.r,profile(ii).dataset.xr,20*log10(abs(profile(ii).dataset.sar_fm_taylor)),[-60 0]);
axis equal xy tight;
colorbar;
colormap jet;


xlabel('Range m')
ylabel('XRange m')
title( '$\left| \tilde{\rho}_{AF,Taylor} (\vec{r})\right|$ dB','Interpreter','latex');

end
exportgraphics(fig_sar2, [outputPath '/SAR_AF.png'], 'Resolution', 300);
