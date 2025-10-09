% PSF_1d.m
function psf =  PSF_1d(weight,Pos_TX,Pos_RX,Freq_mesh,v_vec,r_mid,r),
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
    
    R_1 = sqrt(dot(Pos_TX-r_1*ones(1,nt),Pos_TX-r_1*ones(1,nt),1)) + sqrt(dot(Pos_RX-r_1*ones(1,nt),Pos_RX-r_1*ones(1,nt),1));
    R_2 = sqrt(dot(Pos_TX-r_2*ones(1,nt),Pos_TX-r_2*ones(1,nt),1)) + sqrt(dot(Pos_RX-r_2*ones(1,nt),Pos_RX-r_2*ones(1,nt),1));
    
    % phase = (2*pi/c_sol)*f_n.'*(R_1-R_2);   
    phase = (2*pi/c_sol)*Freq_mesh.*(ones(nf,1)*(R_1-R_2))
    psf(ii) = sum(weight(:).*exp(j*phase(:)));
end