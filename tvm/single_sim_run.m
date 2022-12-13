function results = single_sim_run(see,sei,sie,nuext,f,model,w_e,w_i, m, ch,nu0Ib,perturb,which_p,st_noise,cn_level,time_vector,time_vector2,db_flag)
            if size(time_vector,2)~=3
                error('wrong number of times need [stim on, stim off, tfinal]')
            end
            nu0I = nu0Ib*ones(2,3);
            switch which_p
                case 'all'
                    nu0I(2,:) = nu0I(2,:)*perturb;
                case 'sel'
                    nu0I(2,2:3) = nu0I(2,2:3)*perturb;
                case 'one'
                    nu0I(2,2) = nu0I(2,2)*perturb;
            end
            
           [alpha1,alpha2,I0E1,I0E2,Tnmda,Tampa,alpha,a1_ih,a2_ih,I0I1,I0I2] = gen_alphas(w_e,w_i,see,sei,sie,nuext,model,f,nu0I(1,:));
            %noise_amp = 0.02; 
            noise_amp = st_noise; 
            nu1_in = 2; 
            nu2_in = 2;
            psi1_in = alpha*Tnmda*(nu1_in/1000)/(1+alpha*Tnmda*nu1_in/1000);
            psi2_in = alpha*Tnmda*(nu2_in/1000)/(1+alpha*Tnmda*nu2_in/1000);
            
            conv_f1 = 1.6719;
            conv_f2 = 1.8844;
            conv_I  = 0.9229;
            

            
            
            if db_flag == 1
                tol = 1e-6;
                options = optimoptions('fsolve','Algorithm','Levenberg-Marquardt','FunctionTolerance',tol,'Display','off','OptimalityTolerance',tol);
                %fun1  = @(x) ds1dt_noAMPA(x(1),x(2),marco(1),marco(2),marco(3),mu,coh,100,0.641);
                
                fun1 = @(x) freq_diff_s(x(1),x(2),alpha1,alpha2,I0E1,I0E2,m,ch) - 15; %s2 wins
                fun2 = @(x) freq_diff_s(x(1),x(2),alpha1,alpha2,I0E1,I0E2,m,ch) + 15; %s1 wins
                
                S = 0:0.05:1;
                
                bound_2win = zeros(length(S),length(S),2);
                bound_1win = zeros(length(S),length(S),2);
                
                for ii = 1:length(S)
                    for jj = 1:length(S)
                        bound_2win(ii,jj,:) = fsolve(fun1,[S(ii),S(jj)],options);
                        bound_1win(ii,jj,:) = fsolve(fun2,[S(ii),S(jj)],options);
                    end
                end
                
                
                
                
                bw1 = [reshape(bound_1win(:,:,1),1,length(S)*length(S));reshape(bound_1win(:,:,2),1,length(S)*length(S))];
                bw2 = [reshape(bound_2win(:,:,1),1,length(S)*length(S));reshape(bound_2win(:,:,2),1,length(S)*length(S))];
                [~,x] = sort(bw1(1,:),'ascend');
                bw1 = bw1(:,x);
                [~,x] = sort(bw2(2,:),'ascend');
                bw2 = bw2(:,x);
            else
                bw1 = [];
                bw2 = [];
            end

            %---- Initial conditions and clearing variables -----------

            s1_in = 0.1*rand; s2_in = 0.1*rand;
            I_eta1_in = noise_amp*randn ; I_eta2_in = noise_amp*randn ;
            I_etaI_in = noise_amp*randn ;
            x1_in = 0.3 ; x2_in = 0.3 ;
            f1_in = x1_in*s1_in + x2_in*s2_in ;
            f2_in = x1_in*s2_in + x2_in*s1_in ;
            %----------- Time conditions----------------------------------------------------------------
            dt = 2.;           % Time step in msec
            T_total = time_vector(3)/dt;  % Total number of steps
            %----------- Intialise and vectorise variables to be used in loops below ------------------

            s1 = s1_in.*ones(1,T_total); s2 = s2_in.*ones(1,T_total);
            nu1 = nu1_in.*ones(1,T_total); nu2 = nu2_in.*ones(1,T_total);
            nuI1 = nu1_in.*ones(1,T_total); nuI2 = nu2_in.*ones(1,T_total);
            phi1 = nu1_in.*ones(1,T_total); phi2 = nu2_in.*ones(1,T_total);
            psi1 = psi1_in.*ones(1,T_total); psi2 = psi2_in.*ones(1,T_total);
            x1 = x1_in.*ones(1,T_total); x2 = x2_in.*ones(1,T_total); 
            input_1 = (conv_I*I0E1+I_eta1_in).*ones(1,T_total+1); input_2 = (conv_I*I0E2+I_eta2_in).*ones(1,T_total+1); 
            f1 = f1_in.*ones(1,T_total); f2 = f2_in.*ones(1,T_total); 
            I_eta1 = I_eta1_in.*ones(1,T_total); I_eta2 = I_eta2_in.*ones(1,T_total);
            I_etaI = I_etaI_in.*ones(1,T_total);


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%% Time (or trial) loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            
            
%             cn_level   = 0.5;
            rho = [1,cn_level;cn_level,1];
            [V,D] = eig(rho);
            if(any(diag(D) <= 0))
                results = [];
                error('Rpp must be a positive definite');
                return;
            end
            W = V*sqrt(D);
            
            for t = 1:T_total
                
                %------------ perturbation ----------
                if (time_vector2(1)/dt<t && t<time_vector2(2)/dt)
                    [alpha1,alpha2,I0E1,I0E2,Tnmda,Tampa,alpha,a1_ih,a2_ih,I0I1,I0I2] = gen_alphas(w_e,w_i,see,sei,sie,nuext,model,f,nu0I(2,:));
                else
                    [alpha1,alpha2,I0E1,I0E2,Tnmda,Tampa,alpha,a1_ih,a2_ih,I0I1,I0I2] = gen_alphas(w_e,w_i,see,sei,sie,nuext,model,f,nu0I(1,:));
                end
                
                %------------External stimulus ----------------------------------------------------------------

                I_stim_1 = (time_vector(1)/dt<t & t<time_vector(2)/dt)*(5.2e-4*m*(1+ch/100));
                I_stim_2 = (time_vector(1)/dt<t & t<time_vector(2)/dt)*(5.2e-4*m*(1-ch/100));

                %------------Steady-states of NMDA dynamical gating variables------------------------------------

                psi1(t) = alpha*Tnmda*(nu1(t)/1000)/(1 + alpha*Tnmda*nu1(t)/1000);
                psi2(t) = alpha*Tnmda*(nu2(t)/1000)/(1 + alpha*Tnmda*nu2(t)/1000);
                
                input_1(t) = conv_I*I0E1 + I_stim_1 + I_eta1(t);
                input_2(t) = conv_I*I0E2 + I_stim_2 + I_eta2(t);
                %----------- X variable -------------------
                x1(t) = conv_f1*alpha1*s1(t) + conv_f2*alpha2*s2(t) + input_1(t);
                x2(t) = conv_f1*alpha1*s2(t) + conv_f2*alpha2*s1(t) + input_2(t);

                %---- Shift functions --------------------------------------------------------------

                %---- Gain functions ---------------------------------------

                % Assume linear gain

                a = 270 ;
                b = 108 ;
                d = 0.1540 ;

                %======= Resonse function of competiting excitatory population E1 =============

                phi1(t)  = (a.*(x1(t))-b)./(1-exp(-d.*(a.*(x1(t))-b)));

                %======= Response fucntion of competiting excitatory population E2 =============

                phi2(t)  = (a.*(x2(t))-b)./(1-exp(-d.*(a.*(x2(t))-b)));

                %================ Dynamical equations ===========================================================

                % Mean NMDA-receptor dynamics
                s1(t+1) = s1(t) + dt*(-(s1(t)/Tnmda) + (1-s1(t))*alpha*nu1(t)/1000);
                s2(t+1) = s2(t) + dt*(-(s2(t)/Tnmda) + (1-s2(t))*alpha*nu2(t)/1000);

                %Noise through synaptic currents of pop1 and 2
                cn_temp = randn(2,1);
                corr_noise = W*cn_temp;
                I_eta1(t+1) = I_eta1(t) + (dt/Tampa)*(-I_eta1(t)) + sqrt(dt/Tampa)*noise_amp*corr_noise(1);
                I_eta2(t+1) = I_eta2(t) + (dt/Tampa)*(-I_eta2(t)) + sqrt(dt/Tampa)*noise_amp*corr_noise(2);
                
%                 xI1 = conv_f1*a1_ih*s1(t) + conv_f2*a1_ih*s2(t) + conv_I*I0I1;
%                 xI2 = conv_f1*a2_ih*s1(t) + conv_f2*a1_ih*s2(t) + conv_I*I0I2;
                
%                 nuI1(t+1) = conv_f1*a1_ih*s1(t) + conv_f2*a2_ih*s2(t) + conv_I*I0I1;
%                 nuI2(t+1) = conv_f2*a2_ih*s1(t) + conv_f1*a1_ih*s2(t) + conv_I*I0I2;
                
                nuI1(t+1) = a1_ih*s1(t) + a2_ih*s2(t) + conv_I*I0I1;
                nuI2(t+1) = a2_ih*s1(t) + a1_ih*s2(t) + conv_I*I0I2;
                
                if phi1(t) < 0
                    nu1(t+1) = 0;
                    phi1(t) = 0;
                else
                    nu1(t+1) = phi1(t);
                end
                if phi2(t) < 0
                    nu2(t+1) = 0;
                    phi2(t) = 0;
                else
                    nu2(t+1) = phi2(t);
                end

                %==================================================================================================           
            end  %---- End of time loop --------
            % 4 step moving averge filter of frequency
            n_filt = 20;
            filt      = (1.0/n_filt)*ones(1,n_filt);
            cinput1    = conv(input_1,filt,'same');
            cinput2    = conv(input_2,filt,'same');
            cf1    = conv(nu1,filt,'same');
            cf2    = conv(nu2,filt,'same');
            cfI1    = conv(nuI1,filt,'same');
            cfI2    = conv(nuI2,filt,'same');
            results.time = (0:dt:time_vector(3))/1000;
            results.input_1   = input_1;
            results.input_2   = input_2;
            results.sinput_1   = cinput1;
            results.sinput_2   = cinput2;
            results.n1   = nu1;
            results.n2   = nu2;
            results.nI1   = nuI1;
            results.nI2   = nuI2;
            results.f1   = cf1;
            results.f2   = cf2;
            results.fI1   = cfI1;
            results.fI2   = cfI2;
            results.s1   =  s1;
            results.s2   =  s2;
            results.bw1  = bw1;
            results.bw2  = bw2;
        end