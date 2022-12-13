%%% Plot single trial
see = 0.32;      % Selectivity of EE connections
sei = 0.25;      % Selectivity of EI connections
sie = 0;         % Selectivity of IE connections
w_e = 1;         % Baseline excitatory synaptic weight
w_i = 1;         % Baseline inhibitory synaptic weight
f   = 0.15;      % fraction of circuit which is choice-selective
nuext = 3;       % Background input to excitatory neurons

which_p = 'all'; % Perturb all (all), both selective (sel), or one selective (one) inhibitory population
nu0Ib = 11.5;    % Baseline background input to inhibitory neurons
perturb = 1;     % Perturbation to inhibitory neurons, 1 is no perturbation. Less than one is suppression, greater than one is enhancing

t_vect  = [2000 5000 6000]; % [stim on, stim off, trial end] miliseconds
t_vect2 = [2000 5000];      % [perturb in, perturb off] miliseconds
st_noise = 0.02;            % Noise level
cn_level = 0.0;             % Set correlation for noise between E1 and E2
model    = 'Ours';          % Sets choices for how nonselective neurons connect to the circuit

m = 40; % Stimulus magnitude
ch = 0; % Stimulus strenght [-100,100]

db_flag = 0; % Option to calulate decision boundary 
% plot single trial
results = single_sim_run(see,sei,sie,nuext,f,model,w_e,w_i,...
    m, ch,nu0Ib,perturb,which_p,st_noise,cn_level,t_vect,t_vect2,db_flag);

figure, hold on
plot(results.time,results.f1,'b')
plot(results.time,results.f2,'r')
xlabel('time (s)'), ylabel('Firing rate (Hz)')

%% plot phase planes
figure
subplot(1,2,1), hold on
do_nulls = 1; % Find nullclines
do_vf    = 0; % plot vector field
eps = 6; %deciaml places to round to

mu=0; % set stimulus to zero
[nl1,nl2,c] = get_nulls_fps2(model,mu,ch,see,sei,sie,nuext,f,[nu0Ib,nu0Ib,nu0Ib],eps,do_nulls,do_vf);
plot(nl1(1,:),nl1(2,:),'b')
plot(nl2(1,:),nl2(2,:),'r')
for ii=1:size(c,1)
    if c(ii,3)==1 % stable
        plot(c(ii,1),c(ii,2),'s','MarkerSize',8,'Color','none','MarkerFaceColor',[0. 0. 0.])
    else
        plot(c(ii,1),c(ii,2),'s','MarkerSize',8,'Color','none','MarkerFaceColor',[0.7 0.7 0.7])
    end
end
axis([0 1 0 1]),xlabel('E_1 activity'),ylabel('E_2 activity'), title('Unstimulated')

subplot(1,2,2), hold on
mu=40; % set stimulus to zero
[nl1,nl2,c] = get_nulls_fps2(model,mu,ch,see,sei,sie,nuext,f,[nu0Ib,nu0Ib,nu0Ib],eps,do_nulls,do_vf);
plot(nl1(1,:),nl1(2,:),'b')
plot(nl2(1,:),nl2(2,:),'r')
for ii=1:size(c,1)
    if c(ii,3)==1 % stable
        plot(c(ii,1),c(ii,2),'s','MarkerSize',8,'Color','none','MarkerFaceColor',[0. 0. 0.])
    else
        plot(c(ii,1),c(ii,2),'s','MarkerSize',8,'Color','none','MarkerFaceColor',[0.7 0.7 0.7])
    end
end
egs = c(c(:,3)==0,4:5);
tau_slow = 1./(egs(egs>0));
axis([0 1 0 1]),xlabel('E_1 activity'),ylabel('E_2 activity'), title(['Unstimulated, \tau_{slow}= ',num2str(tau_slow)])
%% plot psychometric
figure
no_runs = 25;
ch_array = -20:2:20;
[results,uu,sing_trial] = gen_psychmetric(no_runs,see,sei,sie,nuext,f,mu,ch_array,...
    nu0Ib,perturb,which_p,t_vect,t_vect2,w_e,w_i,st_noise,cn_level,model);

subplot(3,1,1), plot(ch_array,results.psych,'k')
xlabel('Stimulus strength'), ylabel('P(choose 1)')
subplot(3,1,2), plot(ch_array,results.rt_raw,'k')
xlabel('Stimulus strength'), ylabel('Decision time (s)')
subplot(3,1,3), plot(ch_array,results.ntrial,'k')
xlabel('Stimulus strength'), ylabel('Fraction completed trials')


