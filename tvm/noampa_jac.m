function J = noampa_jac(s1,s2,tauS,a,b,d,gamma,a1,a2,Ie1,Ie2,Is1,Is2)
    
    conv_f1 = 1.6719;
    conv_f2 = 1.8844;

    conv_I  = 0.9229;

    alpha1 = conv_f1*a1;
    alpha2 = conv_f2*a2;
    It1    = conv_I*Ie1 + Is1;
    It2    = conv_I*Ie2 + Is2; 

    g11 =   (a*alpha1*gamma*(s1 - 1))/(exp(d*(b - a*(It1 + alpha1*s1 + alpha2*s2))) - 1) - ...
            (gamma*(b - a*(It1 + alpha1*s1 + alpha2*s2)))/(exp(d*(b - a*(It1 + alpha1*s1 + alpha2*s2))) - 1)...
            - 1/tauS - (a*alpha1*d*gamma*exp(d*(b - a*(It1 + alpha1*s1 + alpha2*s2)))*(b - a*(It1 + alpha1*s1 + alpha2*s2))*(s1 - 1))/(exp(d*(b - a*(It1 + alpha1*s1 + alpha2*s2))) - 1)^2;
        
    g12 =   (a*alpha2*gamma*(s1 - 1))/(exp(d*(b - a*(It1 + alpha1*s1 + alpha2*s2))) - 1)...
            - (a*alpha2*d*gamma*exp(d*(b - a*(It1 + alpha1*s1 + alpha2*s2)))*(b - a*(It1 + alpha1*s1 + alpha2*s2))*(s1 - 1))/(exp(d*(b - a*(It1 + alpha1*s1 + alpha2*s2))) - 1)^2;
        
    g21 =   (a*alpha2*gamma*(s2 - 1))/(exp(d*(b - a*(It2 + alpha1*s2 + alpha2*s1))) - 1)...
            - (a*alpha2*d*gamma*exp(d*(b - a*(It2 + alpha1*s2 + alpha2*s1)))*(b - a*(It2 + alpha1*s2 + alpha2*s1))*(s2 - 1))/(exp(d*(b - a*(It2 + alpha1*s2 + alpha2*s1))) - 1)^2;
        
    g22 =   (a*alpha1*gamma*(s2 - 1))/(exp(d*(b - a*(It2 + alpha1*s2 + alpha2*s1))) - 1)...
            - (gamma*(b - a*(It2 + alpha1*s2 + alpha2*s1)))/(exp(d*(b - a*(It2 + alpha1*s2 + alpha2*s1))) - 1)...
            - 1/tauS - (a*alpha1*d*gamma*exp(d*(b - a*(It2 + alpha1*s2 + alpha2*s1)))*(b - a*(It2 + alpha1*s2 + alpha2*s1))*(s2 - 1))/(exp(d*(b - a*(It2 + alpha1*s2 + alpha2*s1))) - 1)^2;
        
    J = [g11,g12;g21,g22];