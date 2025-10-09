% IPR_1d.m
function ipr =  IPR_1d(weight,RGrad,Freq_mesh,u_vec,r),
% nt = size(RGrad,2);
nf = size(Freq_mesh,1);
% nt = size(Freq_mesh,2);
n_r = length(r);
c_sol = 299792458;
ipr = zeros(size(r));
parfor ii=1:n_r,
    r_ii = r(ii);    
    RGrad_dot_r = RGrad(1,:)*u_vec(1) + RGrad(2,:)*u_vec(2) + RGrad(3,:)*u_vec(3);
    phase = (2*pi*r_ii/c_sol)*Freq_mesh.*(ones(nf,1)*RGrad_dot_r);    
    ipr(ii) = sum(weight(:).*exp(j*phase(:)));
end