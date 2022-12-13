function F = dsdt_noAMPA(x1,x2,alpha1,alpha2,I0E1,I0E2,mu,coh,Tnmda,alpha)

conv_f1 = 1.6719;
conv_f2 = 1.8844;
        
conv_I  = 0.9229;

I_stim_1 = (5.2e-4*mu*(1+coh/100));
I_stim_2 = (5.2e-4*mu*(1-coh/100));

s1 = conv_f1*alpha1*x1 + conv_f2*alpha2*x2 + conv_I*I0E1 + I_stim_1;
s2 = conv_f1*alpha1*x2 + conv_f2*alpha2*x1 + conv_I*I0E2 + I_stim_2;

a = 270;
b = 108;
d = 0.1540;


phi1 = (a.*(s1)-b)./(1-exp(-d.*(a.*(s1)-b)));
phi2 = (a.*(s2)-b)./(1-exp(-d.*(a.*(s2)-b)));

F(1) = -(x1/Tnmda) + (1-x1)*alpha*phi1/1000;
F(2) = -(x2/Tnmda) + (1-x2)*alpha*phi2/1000;