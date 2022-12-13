function [alpha1,alpha2,I0E1,I0E2,Tnmda,Tampa,alpha,a1_ih,a2_ih,I0I1,I0I2] = gen_alphas(w_e,w_i,see,sei,sie,nuext,model,f,nu0I)
    %%%% Network parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    CE    = 1600; % Total number of recurrent excitatory synaptic connection for each E cell
    Cext  = 800;  % Total number of external excitatory synaptic connection for each E or I cell
    CI    = 400;  % Total number of recurrent inhibitory synaptic connection for each I cell
    NI    = 400;  % Total number of recurrent excitatory synaptic connection for each I cell
    %f     = 0.15; % Fraction of potentiated synapses among all E cells
    Ns    = 2;
%     nuext = 3;    % Mean firing rate of external neurons

    VE = -53.4;
    VI = -52.1;
    EE =   0.0;
    EI = -70.0;

    %%%% Synaptic decay time constants (ms) %%%%%%%%%%%%%%%%%%%%%

    Tnmda = 100; % NMDAR
    Tampa = 2;   % AMPAR
    Tgaba = 5;   % GABAR (GABA_A)

    %%%%% Synaptic coupling constants conductances in microSiemens %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    switch model 
        case 'Ours'
            gE_rec_nmda = 1.9500e-04;
            gI_rec_nmda = 1.0200e-04;
            gE_rec_gaba = 0.0130;
            gI_rec_gaba = 0.0084;

%                     gE_rec_nmda = 1.5e-4;
%                     gI_rec_nmda = 1.29e-4;
%                     gE_ext_ampa = 2.1e-3;
%                     gI_ext_ampa = 1.62e-3;
%                     gE_rec_gaba = 1.3e-3;
%                     gI_rec_gaba = 1.e-3;

%                     gE_rec_nmda = 0.000841;
%                     gI_rec_nmda = 0.000503;
%                     gE_rec_gaba = 0.006312;
%                     gI_rec_gaba = 0.006182;


            gE_ext_ampa = 2.1e-3;
            gI_ext_ampa = 1.62e-3;

            % % g*<V> bare synaptic couplings
            JgabaE    = -gE_rec_gaba*(EI-VE);     % GABA recurrent for E cells
            JgabaI    = -gI_rec_gaba*(EI-VI);     % GABA recurrent for I cells
            JampaextE = gE_ext_ampa*(EE-VE);     % AMPA external for E cells
            JampaextI = gI_ext_ampa*(EE-VI);     % AMPA external for I cells

            JnmdaeffE = gE_rec_nmda*(EE - VE)/(1+(1/3.57)*exp(-0.062*VE)); % NMDA recurrent for E cells
            JnmdaeffI = gI_rec_nmda*(EE - VI)/(1+(1/3.57)*exp(-0.062*VI)); % NMDA recurrent for I cells
        case 'XJW'
            gE_rec_nmda = 1.5e-4;
            gI_rec_nmda = 1.29e-4;
            gE_ext_ampa = 2.1e-3;
            gI_ext_ampa = 1.62e-3;
            gE_rec_gaba = 1.3e-3;
            gI_rec_gaba = 1.e-3;

            % % g*<V> bare synaptic couplings
            JgabaE    = -gE_rec_gaba*(EI-VE);     % GABA recurrent for E cells
            JgabaI    = -gI_rec_gaba*(EI-VI);     % GABA recurrent for I cells
            JampaextE = gE_ext_ampa*(EE-VE);     % AMPA external for E cells
            JampaextI = gI_ext_ampa*(EE-VI);     % AMPA external for I cells

            % g*<V>/(1+(1/3.57)*exp(-0.062*<V>)) effective synaptic couplings
            JnmdaeffE = gE_rec_nmda*(EE - VE)/(1+(1/3.57)*exp(-0.062*VE)); % NMDA recurrent for E cells
            JnmdaeffI = gI_rec_nmda*(EE - VI)/(1+(1/3.57)*exp(-0.062*VI)); % NMDA recurrent for I cells
        case 'ZMN'
            gE_rec_nmda = 1.5000e-04;
            gI_rec_nmda = 1.2700e-04; % seimens; A/V
            gE_rec_gaba = 0.0106;
            gI_rec_gaba = 0.0086;

            gE_ext_ampa = 2.1e-3;
            gI_ext_ampa = 1.62e-3;

            % % g*<V> bare synaptic couplings
            JgabaE    = -gE_rec_gaba*(EI-VE);     % GABA recurrent for E cells, A
            JgabaI    = -gI_rec_gaba*(EI-VI);     % GABA recurrent for I cells, A
            JampaextE = gE_ext_ampa*(EE-VE);     % AMPA external for E cells, A
            JampaextI = gI_ext_ampa*(EE-VI);     % AMPA external for I cells, A

            % g*<V>/(1+(1/3.57)*exp(-0.062*<V>)) effective synaptic couplings
            JnmdaeffE = gE_rec_nmda*(EE - VE)/(1+(1/3.57)*exp(-0.062*VE)); % NMDA recurrent for E cells
            JnmdaeffI = gI_rec_nmda*(EE - VI)/(1+(1/3.57)*exp(-0.062*VI)); % NMDA recurrent for I cells
    end

    %%%% Other synaptic constant %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    alpha = 0.641;

    %%%% FI curve parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Im  = 125; g  = 0.16;  c  = 310;  % Larrys functions paramters for pyr cells
    ImI = 177; gI = 0.087; cI = 615;  % Larrys functions parameters for interneurons

    gI2  = 2;    % Dimensionless
%     nu0I = 11.5; % In units of Hz
    eta  = 1 + (cI/gI2)*(NI*JgabaI*Tgaba/1000); % Linearised factor for I-cells

    %%%% Parameters to be varied %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %app.coh       = 100.0;               % Motion coherence index
    %app.mu        = 40.0;               % mu_0, external (absolute) stimulus strength

    Sei = sei;
    Sie = sie; 
    See = see;
    nu3_in = 2;
    psi3_in = alpha*Tnmda*(nu3_in/1000)/(1+alpha*Tnmda*nu3_in/1000); 

    %------------External current input -----------------------------------------------------------           
    IampaextE1 = JampaextE*Tampa*Cext*nuext/1000;     %E-cells
    IampaextE2 = JampaextE*Tampa*Cext*nuext/1000; 
    IampaextI  = JampaextI*Tampa*Cext*nuext/1000;                 %I-cells
    %------------Coupling constants------------------------------------------------------------------
    switch model                              
        case 'Ours'
            what_ee = Ns*w_e/(Ns + See*(2-Ns));
            what_ei = Ns*w_e/(Ns + Sei*(2 - Ns));
            what_ie = Ns*w_i/(Ns + Sie*(2 - Ns));

            wpee = what_ee + See*what_ee;
            wmee = what_ee - See*what_ee;
            w0ee = w_e; 

            wpei = what_ei + Sei*what_ei;
            wmei = what_ei - Sei*what_ei;
            w0ei = w_e; 

            wpie = what_ie + Sie*what_ie;
            wmie = what_ie - Sie*what_ie;
            w0ie = w_i;

            alpha1_a = f*CE*wpee*JnmdaeffE ; % Ex->Ex
            alpha1_b = - (1/(eta*gI2))*(cI* f*CE*wpei*JnmdaeffI)*( f*wpie*CI*JgabaE*Tgaba/1000); %Ex->Ix->Ex
            alpha1_c = - (1/(eta*gI2))*(cI* f*CE*wmei*JnmdaeffI)*( f*wmie*CI*JgabaE*Tgaba/1000); %Ex->Iy->Ex
            alpha1_d = - (1/(eta*gI2))*(cI* f*CE*w0ei*JnmdaeffI)*((1-Ns* f)*w0ie*CI*JgabaE*Tgaba/1000); %Ex -> I0 -> Ex

            alpha1   = alpha1_a + alpha1_b + alpha1_c + alpha1_d;
            
            alpha2_a =  f*CE*wmee*JnmdaeffE ; %Ey -> Ex
            alpha2_b = - (1/(eta*gI2))*(cI* f*CE*wmei*JnmdaeffI)*( f*wpie*CI*JgabaE*Tgaba/1000); %Ey -> Ix -> Ex
            alpha2_c = - (1/(eta*gI2))*(cI* f*CE*wpei*JnmdaeffI)*( f*wmie*CI*JgabaE*Tgaba/1000); %Ey -> Iy -> Ex
            alpha2_d = - (1/(eta*gI2))*(cI* f*CE*w0ei*JnmdaeffI)*((1-Ns* f)*w0ie*CI*JgabaE*Tgaba/1000); %Ey -> I0 -> Ex

            alpha2   = alpha2_a + alpha2_b + alpha2_c + alpha2_d;
            
%             a1_ih = (1/(eta*gI2))*(cI* f*CE*wpei*JnmdaeffI); % for Ex -> Ix
%             a2_ih = (1/(eta*gI2))*(cI* f*CE*wmei*JnmdaeffI); % for Ey -> Iy
            
            a1_ih = (cI* f*CE*wpei*JnmdaeffI)/gI2; % for Ex -> Ix
            a2_ih = (cI* f*CE*wmei*JnmdaeffI)/gI2; % for Ey -> Iy
            %----------- Constant effective external inputs and inhibition----------------------------------------------------

            I0inh_sel   = IampaextI + JnmdaeffI*w0ei*(1-Ns* f)*CE*psi3_in ; %E0 -> Ix/y
            I0inh_nsel  = IampaextI + JnmdaeffI*w0ei*(1-Ns* f)*CE*psi3_in ; %E0 -> I0
            
            I0E1_a  = (1-Ns* f)*CE*w0ee*JnmdaeffE*psi3_in; % E0 -> E1
            I0E1_b  = IampaextE1 - (1-Ns* f)*w0ie*NI*JgabaE*(Tgaba/1000)*(nu0I(1) + (cI*I0inh_nsel-ImI)/gI2)/eta; % I0 -> E1
            I0E1_c  = -  f*wpie*NI*JgabaE*(Tgaba/1000)*(nu0I(2) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I1 -> E1
            I0E1_d  = -  f*wmie*NI*JgabaE*(Tgaba/1000)*(nu0I(3) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I2 -> E1
            I0E1    = I0E1_a + I0E1_b + I0E1_c + I0E1_d;

            I0E2_a  = (1-Ns*f)*CE*w0ee*JnmdaeffE*psi3_in; % E0 -> E2
            I0E2_b  = IampaextE2 - (1-Ns*f)*w0ie*NI*JgabaE*(Tgaba/1000)*(nu0I(1) + (cI*I0inh_nsel-ImI)/gI2)/eta; % I0 -> E2
            I0E2_c  = -  f*wmie*NI*JgabaE*(Tgaba/1000)*(nu0I(2) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I1 -> E2
            I0E2_d  = -  f*wpie*NI*JgabaE*(Tgaba/1000)*(nu0I(3) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I2 -> E2
            I0E2    = I0E2_a + I0E2_b + I0E2_c + I0E2_d;
            
%             I0I1 = (nu0I + (cI*I0inh_sel-ImI)/gI2)/eta;
%             I0I2 = (nu0I + (cI*I0inh_sel-ImI)/gI2)/eta;
            
            I0I1 = (nu0I(2) + (cI*I0inh_sel-ImI)/gI2);
            I0I2 = (nu0I(3) + (cI*I0inh_sel-ImI)/gI2);
%             if Sei == 0 && Sie == 0
%                 alpha1, alpha2, I0E1, wpee,wpie,wpei
%             end
            %alpha1 = 0.1560; alpha2 = -0.0264; I0E1 = 0.3535; I0E2 = 0.3535;
        case 'XJW'
            if See >=0
                what_ee = w_e/(f*(1+See) + f*(Ns - 1)*(1 - See) + (1 - f*Ns)*(1 - See));
            else
                what_ee = w_e/(f*(1+See) + f*(Ns - 1)*(1 - See) + (1 - f*Ns)*(1 + See));
            end
            if Sei >= 0
                what_ei = w_e/(f*(1+Sei) + f*(Ns - 1)*(1 - Sei) + (1 - f*Ns)*(1 - Sei));
            else
                what_ei = w_e/(f*(1+Sei) + f*(Ns - 1)*(1 - Sei) + (1 - f*Ns)*(1 + Sei));
            end
            if Sie >=0
                what_ie = w_i/(f*(1+Sie) + f*(Ns - 1)*(1 - Sie) + (1 - f*Ns)*(1 - Sie));
            else
                what_ie = w_i/(f*(1+Sie) + f*(Ns - 1)*(1 - Sie) + (1 - f*Ns)*(1 + Sie));
            end


            wpee = what_ee + See *what_ee;
            wmee = what_ee - See *what_ee;
            w0ee = w_e; 

            wpei = what_ei + Sei*what_ei;
            wmei = what_ei - Sei*what_ei;
            w0ei = w_e; 

            wpie = what_ie + Sie*what_ie;
            wmie = what_ie - Sie*what_ie;
            w0ie = w_i;

            alpha1_a = f*CE*wpee*JnmdaeffE ; % Ex->Ex
            alpha1_b = - (1/(eta*gI2))*(cI*f*CE*wpei*JnmdaeffI)*(f*wpie*CI*JgabaE*Tgaba/1000); %Ex->Ix->Ex
            alpha1_c = - (1/(eta*gI2))*(cI*f*CE*wmei*JnmdaeffI)*(f*wmie*CI*JgabaE*Tgaba/1000); %Ex->Iy->Ex
            if Sie >= 0
                alpha1_d = - (1/(eta*gI2))*(cI*f*CE*w0ei*JnmdaeffI)*((1-Ns*f)*wmie*CI*JgabaE*Tgaba/1000); %Ex -> I0 -> Ex
            else
                alpha1_d = - (1/(eta*gI2))*(cI*f*CE*w0ei*JnmdaeffI)*((1-Ns*f)*wpie*CI*JgabaE*Tgaba/1000); %Ex -> I0 -> Ex
            end
            alpha1   = alpha1_a + alpha1_b + alpha1_c + alpha1_d;

            alpha2_a = f*CE*wmee*JnmdaeffE ; %Ey -> Ex
            alpha2_b = - (1/(eta*gI2))*(cI*f*CE*wmei*JnmdaeffI)*(f*wpie*CI*JgabaE*Tgaba/1000); %Ey -> Ix -> Ex
            alpha2_c = - (1/(eta*gI2))*(cI*f*CE*wpei*JnmdaeffI)*(f*wmie*CI*JgabaE*Tgaba/1000); %Ey -> Iy -> Ex
            if Sie >= 0
                alpha2_d = - (1/(eta*gI2))*(cI*f*CE*w0ei*JnmdaeffI)*((1-Ns*f)*wmie*CI*JgabaE*Tgaba/1000); %Ey -> I0 -> Ex
            else
                alpha2_d = - (1/(eta*gI2))*(cI*f*CE*w0ei*JnmdaeffI)*((1-Ns*f)*wpie*CI*JgabaE*Tgaba/1000); %Ey -> I0 -> Ex
            end
            alpha2   = alpha2_a + alpha2_b + alpha2_c + alpha2_d;
            
            a1_ih = (cI* f*CE*wpei*JnmdaeffI)/gI2; % for Ex -> Ix
            a2_ih = (cI* f*CE*wmei*JnmdaeffI)/gI2; % for Ey -> Iy
            
            %----------- Constant effective external inputs and inhibition----------------------------------------------------
            if Sei >=0 
                I0inh_sel   = IampaextI + JnmdaeffI*wmei*(1-Ns*f)*CE*psi3_in ; %E0 -> Ix/y
            else
                I0inh_sel   = IampaextI + JnmdaeffI*wpei*(1-Ns*f)*CE*psi3_in ; %E0 -> Ix/y
            end
            I0inh_nsel  = IampaextI + JnmdaeffI*w0ei*(1-Ns*f)*CE*psi3_in ; %E0 -> I0
            if See >=0
                I0E1_a  = (1-Ns*f)*CE*wmee*JnmdaeffE*psi3_in; % E0 -> E1
            else
                I0E1_a  = (1-Ns*f)*CE*wpee*JnmdaeffE*psi3_in; % E0 -> E1
            end
            if Sie >=0
                I0E1_b  = IampaextE1 - (1-Ns*f)*wmie*NI*JgabaE*(Tgaba/1000)*(nu0I(1) + (cI*I0inh_nsel-ImI)/gI2)/eta; % I0 -> E1
            else
                I0E1_b  = IampaextE1 - (1-Ns*f)*wpie*NI*JgabaE*(Tgaba/1000)*(nu0I(1) + (cI*I0inh_nsel-ImI)/gI2)/eta; % I0 -> E1
            end
            I0E1_c  = - f*wpie*NI*JgabaE*(Tgaba/1000)*(nu0I(2) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I1 -> E1
            I0E1_d  = - f*wmie*NI*JgabaE*(Tgaba/1000)*(nu0I(3) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I2 -> E1
            I0E1    = I0E1_a + I0E1_b + I0E1_c + I0E1_d;

            if See >=0
                I0E2_a  = (1-Ns*f)*CE*wmee*JnmdaeffE*psi3_in; % E0 -> E2
            else
                I0E2_a  = (1-Ns*f)*CE*wpee*JnmdaeffE*psi3_in; % E0 -> E2
            end
            if Sie >=0
                I0E2_b  = IampaextE1 - (1-Ns*f)*wmie*NI*JgabaE*(Tgaba/1000)*(nu0I(1) + (cI*I0inh_nsel-ImI)/gI2)/eta; % I0 -> E1
            else
                I0E2_b  = IampaextE1 - (1-Ns*f)*wpie*NI*JgabaE*(Tgaba/1000)*(nu0I(1) + (cI*I0inh_nsel-ImI)/gI2)/eta; % I0 -> E1
            end
            I0E2_c  = - f*wmie*NI*JgabaE*(Tgaba/1000)*(nu0I(2) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I1 -> E2
            I0E2_d  = - f*wpie*NI*JgabaE*(Tgaba/1000)*(nu0I(3) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I2 -> E2
            I0E2    = I0E2_a + I0E2_b + I0E2_c + I0E2_d;
            
            I0I1 = (nu0I(2) + (cI*I0inh_sel-ImI)/gI2);
            I0I2 = (nu0I(3) + (cI*I0inh_sel-ImI)/gI2);
            
        case 'ZMN'

            if See >=0
                what_ee = w_e/(f*(1+ See ) +  f*(Ns - 1)*(1 -  See ) + (1 -  f*Ns)*(1 -  See ));
            else
                what_ee = w_e/(f*(1+ See ) +  f*(Ns - 1)*(1 -  See ) + (1 -  f*Ns)*(1 +  See ));
            end
            if Sei >= 0
                what_ei =  w_e /( f*(1+Sei) +  f*(Ns - 1)*(1 - Sei) + (1 -  f*Ns)*(1 - Sei));
            else
                what_ei =  w_e /( f*(1+Sei) +  f*(Ns - 1)*(1 - Sei) + (1 -  f*Ns)*(1 + Sei));
            end
            if Sie >=0
                what_ie =  w_i /( f*(1+Sie) +  f*(Ns - 1)*(1 - Sie) + (1 -  f*Ns)*(1 - Sie));
            else
                what_ie =  w_i /( f*(1+Sie) +  f*(Ns - 1)*(1 - Sie) + (1 -  f*Ns)*(1 + Sie));
            end

            if  See  >=0
                wbar_ee = ( w_e /(1 - Ns* f)) - ( w_e * f*Ns/(1-Ns* f))*((1- See )/( f*(1+ See ) +  f*(Ns-1)*(1- See ) + (1- f*Ns)*(1 -  See )));
            else
                wbar_ee = ( w_e /(1 - Ns* f)) - ( w_e * f*Ns/(1-Ns* f))*((1+ See )/( f*(1+ See ) +  f*(Ns-1)*(1- See ) + (1- f*Ns)*(1 +  See )));
            end
            if Sei >=0
                wbar_ei = ( w_e /(1 - Ns* f)) - ( w_e * f*Ns/(1-Ns* f))*((1-Sei)/( f*(1+Sei) +  f*(Ns-1)*(1-Sei) + (1- f*Ns)*(1 - Sei)));
            else
                wbar_ei = ( w_e /(1 - Ns* f)) - ( w_e * f*Ns/(1-Ns* f))*((1+Sei)/( f*(1+Sei) +  f*(Ns-1)*(1-Sei) + (1- f*Ns)*(1 + Sei)));
            end
            if Sie >=0
                wbar_ie = ( w_i /(1 - Ns* f)) - ( w_i * f*Ns/(1-Ns* f))*((1-Sie)/( f*(1+Sie) +  f*(Ns-1)*(1-Sie) + (1- f*Ns)*(1 - Sie)));
            else
                wbar_ie = ( w_i /(1 - Ns* f)) - ( w_i * f*Ns/(1-Ns* f))*((1+Sie)/( f*(1+Sie) +  f*(Ns-1)*(1-Sie) + (1- f*Ns)*(1 + Sie)));
            end

            wpee = what_ee +  See *what_ee;
            wmee = what_ee -  See *what_ee;
            w0ee =  w_e ; 

            wpei = what_ei + Sei*what_ei;
            wmei = what_ei - Sei*what_ei;
            w0ei =  w_e ; 

            wpie = what_ie + Sie*what_ie;
            wmie = what_ie - Sie*what_ie;
            w0ie =  w_i ;

            alpha1_a =  f*CE*wpee*JnmdaeffE ; % Ex->Ex
            alpha1_b = - (1/(eta*gI2))*(cI* f*CE*wpei*JnmdaeffI)*( f*wpie*CI*JgabaE*Tgaba/1000); %Ex->Ix->Ex
            alpha1_c = - (1/(eta*gI2))*(cI* f*CE*wmei*JnmdaeffI)*( f*wmie*CI*JgabaE*Tgaba/1000); %Ex->Iy->Ex
            if Sei >=0 && Sie >=0
                alpha1_d = - (1/(eta*gI2))*(cI* f*CE*wmei*JnmdaeffI)*((1-Ns* f)*wmie*CI*JgabaE*Tgaba/1000); %Ex -> I0 -> Ex
            elseif Sei >=0 && Sie <0
                alpha1_d = - (1/(eta*gI2))*(cI* f*CE*wmei*JnmdaeffI)*((1-Ns* f)*wpie*CI*JgabaE*Tgaba/1000); %Ex -> I0 -> Ex
            elseif Sei <0 && Sie >=0
                alpha1_d = - (1/(eta*gI2))*(cI* f*CE*wpei*JnmdaeffI)*((1-Ns* f)*wmie*CI*JgabaE*Tgaba/1000); %Ex -> I0 -> Ex
            else
                alpha1_d = - (1/(eta*gI2))*(cI* f*CE*wpei*JnmdaeffI)*((1-Ns* f)*wpie*CI*JgabaE*Tgaba/1000); %Ex -> I0 -> Ex
            end
            alpha1   = alpha1_a + alpha1_b + alpha1_c + alpha1_d;

            alpha2_a =  f*CE*wmee*JnmdaeffE ; %Ey -> Ex
            alpha2_b = - (1/(eta*gI2))*(cI* f*CE*wmei*JnmdaeffI)*( f*wpie*CI*JgabaE*Tgaba/1000); %Ey -> Ix -> Ex
            alpha2_c = - (1/(eta*gI2))*(cI* f*CE*wpei*JnmdaeffI)*( f*wmie*CI*JgabaE*Tgaba/1000); %Ey -> Iy -> Ex
            if Sei >=0 && Sie >=0
                alpha2_d = - (1/(eta*gI2))*(cI* f*CE*wmei*JnmdaeffI)*((1-Ns* f)*wmie*CI*JgabaE*Tgaba/1000); %Ey -> I0 -> Ex
            elseif Sei >=0 && Sie <0
                alpha2_d = - (1/(eta*gI2))*(cI* f*CE*wmei*JnmdaeffI)*((1-Ns* f)*wpie*CI*JgabaE*Tgaba/1000); %Ey -> I0 -> Ex
            elseif Sei <0 && Sie >=0
                alpha2_d = - (1/(eta*gI2))*(cI* f*CE*wpei*JnmdaeffI)*((1-Ns* f)*wmie*CI*JgabaE*Tgaba/1000); %Ey -> I0 -> Ex
            else
                alpha2_d = - (1/(eta*gI2))*(cI* f*CE*wpei*JnmdaeffI)*((1-Ns* f)*wpie*CI*JgabaE*Tgaba/1000); %Ey -> I0 -> Ex
            end
            alpha2   = alpha2_a + alpha2_b + alpha2_c + alpha2_d;
            
            a1_ih = (cI* f*CE*wpei*JnmdaeffI)/gI2; % for Ex -> Ix
            a2_ih = (cI* f*CE*wmei*JnmdaeffI)/gI2; % for Ey -> Iy
            
            %----------- Constant effective external inputs and inhibition----------------------------------------------------
            if Sei >=0
                I0inh_sel   = IampaextI + JnmdaeffI*wmei*(1-Ns* f)*CE*psi3_in ; %E0 -> Ix/y
            else
                I0inh_sel   = IampaextI + JnmdaeffI*wpei*(1-Ns* f)*CE*psi3_in ; %E0 -> Ix/y    
            end
            I0inh_nsel  = IampaextI + JnmdaeffI*wbar_ei*(1-Ns* f)*CE*psi3_in ; %E0 -> I0
            if  See  >=0
                I0E1_a  = (1-Ns* f)*CE*wmee*JnmdaeffE*psi3_in ; % E0 -> E1
            else
                I0E1_a  = (1-Ns* f)*CE*wpee*JnmdaeffE*psi3_in ; % E0 -> E1
            end
            if Sie >= 0
                I0E1_b  = IampaextE1 - (1-Ns* f)*wmie*NI*JgabaE*(Tgaba/1000)*(nu0I(1) + (cI*I0inh_nsel-ImI)/gI2)/eta; % I0 -> E1
            else
                I0E1_b  = IampaextE1 - (1-Ns* f)*wpie*NI*JgabaE*(Tgaba/1000)*(nu0I(1) + (cI*I0inh_nsel-ImI)/gI2)/eta; % I0 -> E1
            end
            I0E1_c  = -  f*wpie*NI*JgabaE*(Tgaba/1000)*(nu0I(2) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I1 -> E1
            I0E1_d  = -  f*wmie*NI*JgabaE*(Tgaba/1000)*(nu0I(3) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I2 -> E1
            I0E1    = I0E1_a + I0E1_b + I0E1_c + I0E1_d;
            if  See  >=0
                I0E2_a  = (1-Ns* f)*CE*wmee*JnmdaeffE*psi3_in ; % E0 -> E2
            else
                I0E2_a  = (1-Ns* f)*CE*wpee*JnmdaeffE*psi3_in ; % E0 -> E2
            end
            if Sie >=0
                I0E2_b  = IampaextE2 - (1-Ns* f)*wmie*NI*JgabaE*(Tgaba/1000)*(nu0I(1) + (cI*I0inh_nsel-ImI)/gI2)/eta; % I0 -> E2
            else
                I0E2_b  = IampaextE2 - (1-Ns* f)*wpie*NI*JgabaE*(Tgaba/1000)*(nu0I(1) + (cI*I0inh_nsel-ImI)/gI2)/eta; % I0 -> E2
            end
            I0E2_c  = -  f*wmie*NI*JgabaE*(Tgaba/1000)*(nu0I(2) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I1 -> E2
            I0E2_d  = -  f*wpie*NI*JgabaE*(Tgaba/1000)*(nu0I(2) + (cI*I0inh_sel-ImI)/gI2)/eta;                   % I2 -> E2
            I0E2    = I0E2_a + I0E2_b + I0E2_c + I0E2_d;
            
            I0I1 = (nu0I(2) + (cI*I0inh_sel-ImI)/gI2);
            I0I2 = (nu0I(3) + (cI*I0inh_sel-ImI)/gI2);
    end
end