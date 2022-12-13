function [nl1,nl2,c,vf,Ss] = get_nulls_fps2(model,mu,coh,see,sei,sie,nuext,f,nu0I,eps,do_nulls,do_vf)

% switch model
%     case 'XJW'
%         gE_rec_nmda = 1.5e-4;
%         gI_rec_nmda = 1.29e-4;
%         gE_rec_gaba = 1.3e-3;
%         gI_rec_gaba = 1.e-3;
%     case 'Ours'
%         gE_rec_nmda = 1.9500e-04;
%         gI_rec_nmda = 1.0200e-04;
%         gE_rec_gaba = 0.0130;
%         gI_rec_gaba = 0.0084;
%     case 'ZMN'
%         gE_rec_nmda = 1.5000e-04;
%         gI_rec_nmda = 1.2700e-04;
%         gE_rec_gaba = 0.0106;
%         gI_rec_gaba = 0.0086;
% end
% mu  = 40;
% coh = 0;
% sei = 0.25;
% sie = -0.25;

if strcmp(model,'Ours')
    uu = 'Ours';
elseif strcmp(model,'OURS')
    uu = 'Ours';
else
    uu = model;
end
w_e = 1;
w_i = 1;

% marco = genMODELalphas(uu,sei,sie,nuext,gE_rec_nmda,gI_rec_nmda,gE_rec_gaba,gI_rec_gaba);
% [alpha1,alpha2,I0E1,I0E2,Tnmda,Tampa,alpha,a1_ih,a2_ih,I0I1,I0I2] = gen_alphas(w_e,w_i,see,sei,sie,nuext,model,f,nu0I);
[alpha1,alpha2,I0E1,I0E2,Tnmda,Tampa,alpha,a1_ih,a2_ih,I0I1,I0I2] = gen_alphas(w_e,w_i,see,sei,sie,nuext,model,f,nu0I);


S = 0:0.05:1; % i had at 0.025
S = round(S,4);
fp = zeros(length(S),length(S),2);
fl = zeros(length(S),length(S));
tol = 1e-6;
options = optimoptions('fsolve','Algorithm','Levenberg-Marquardt','FunctionTolerance',tol,'Display','off','OptimalityTolerance',tol);
fun  = @(x) 100*dsdt_noAMPA(x(1),x(2),alpha1,alpha2,I0E1,I0E2,mu,coh,100,0.641);
for ii=1:length(S)
    for jj=1:length(S)
        [fp(ii,jj,:),~,fl(ii,jj)] = fsolve(fun,[S(ii),S(jj)],options);
    end
end

scat_s1 = reshape(fp(:,:,1),1,length(S).^2);
scat_s2 = reshape(fp(:,:,2),1,length(S).^2);
scat_fl = reshape(fl(:,:),1,length(S).^2);
% plot(scat_s1(scat_fl==1),scat_s2(scat_fl==1),'ro','MarkerSize',16)
% plot(scat_s1(scat_fl~=1),scat_s2(scat_fl~=1),'r.','MarkerSize',16)

fpts = [scat_s1',scat_s2']; 
fpts = round(fpts,eps);
dd = unique(fpts,'rows');
frac = zeros(size(dd,1),1);
for ii = 1:size(dd,1)
    [a b]=find(fpts==dd(ii,1));
    [c d]=find(fpts==dd(ii,2));
    output = intersect(a,c);
%     index = c.*NaN;
%     for kk=1:length(c)
%       check =  a(a==c(kk))';
%       if ~isempty ( check )
%         index(kk) = check;
%       end
%     end
%     output = index(~isnan(index));
    if sum(scat_fl(output) == 1) == 0
        frac(ii) = 1;
    end
end
c = dd(frac==0,:);
a = 270;
b = 108;
d = 0.1540;
for ii = 1:size(c,1)
    J = noampa_jac(c(ii,1),c(ii,2),.100,a,b,d,0.641,alpha1,alpha2,I0E1,I0E2,(5.2e-4*mu*(1+coh/100)),(5.2e-4*mu*(1-coh/100)));
    [~,y] = eig(J);
    c(ii,4) = y(1,1);
    c(ii,5) = y(2,2);
    if y(1,1) < 0 && y(2,2) < 0
        c(ii,3) = 1;
    else
        c(ii,3) = 0;
    end
end
%%
if do_nulls
    S  = 0:0.05:1;
    S  = round(S,4); 
    options = optimoptions('fsolve','Algorithm','Levenberg-Marquardt','FunctionTolerance',tol,'Display','off','OptimalityTolerance',tol);
    fun1  = @(x) ds1dt_noAMPA(x(1),x(2),alpha1,alpha2,I0E1,mu,coh,100,0.641);
    fun2  = @(x) ds2dt_noAMPA(x(1),x(2),alpha1,alpha2,I0E2,mu,coh,100,0.641);
    null1 = zeros(length(S),length(S),2);
    null2 = zeros(length(S),length(S),2);
    for ii=1:length(S)
        for jj=1:length(S)
            null1(ii,jj,:) = fsolve(fun1,[S(ii),S(jj)],options);
            null2(ii,jj,:) = fsolve(fun2,[S(ii),S(jj)],options);
        end
    end
    
    nl1 = [reshape(null1(:,:,1),1,length(S)*length(S));reshape(null1(:,:,2),1,length(S)*length(S))];
    nl2 = [reshape(null2(:,:,1),1,length(S)*length(S));reshape(null2(:,:,2),1,length(S)*length(S))];
    [~,x] = sort(nl1(1,:),'ascend');
    nl1 = nl1(:,x);
    [~,x] = sort(nl2(2,:),'ascend');
    nl2 = nl2(:,x);
else
    nl1 = [];
    nl2 = [];
end

if do_vf
    S  = 0:0.05:1;
    S  = round(S,4);
    Ss = zeros(length(S),length(S),2);
    vf = zeros(length(S),length(S),2);
    for ii=1:length(S)
        for jj=1:length(S)
            Ss(ii,jj,1) = S(ii);
            Ss(ii,jj,2) = S(jj);
            vf(ii,jj,1) = ds1dt_noAMPA(S(ii),S(jj),alpha1,alpha2,I0E1,mu,coh,100,0.641);
            vf(ii,jj,2) = ds2dt_noAMPA(S(ii),S(jj),alpha1,alpha2,I0E1,mu,coh,100,0.641);
        end
    end
end
% P = InterX(nl1,nl2);
% %%
% plot(nl1(1,:),nl1(2,:),'b','LineWidth',2), hold on
% plot(nl2(1,:),nl2(2,:),'r','LineWidth',2)


% nl1 % s1 null cline
% nl2 % s2 null cline
% c % fixed points
