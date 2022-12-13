function [results,uu,sing_trial] = gen_psychmetric(no_runs,see,sei,sie,nuext,f,mu,ch,nu0Ib,perturb,which_p,t_vect,t_vect2,w_e,w_i,st_noise,cn_level,model)
    %mu = 40;
%     ch = -50:1:50;
    %ch = 1;
    
%     t_vect = [4000,6000,8000];
    
%     no_runs = 2000;
    choose = zeros(length(ch),no_runs);
    rt_corr = zeros(length(ch),no_runs);
    rt_icor = zeros(length(ch),no_runs);
    rt_raww = zeros(length(ch),no_runs);
    wtb = waitbar(0,['Running for \Sigma^{EE} = ',num2str(see),', \Sigma^{EI} = ',num2str(sei),', \Sigma^{IE} = ',num2str(sie)]);
    for ii = 1:length(ch)
        for jj = 1:no_runs
%                          single_sim_run(see,sei,sie,nuext,f,model,w_e,w_i, m, ch,st_noise,cn_level,db_flag)
%             sing_trial = single_sim_run(see,sei,sie,nuext,f,model,w_e,w_i, mu, ch(ii),st_noise,cn_level,0);
            sing_trial = single_sim_run(see,sei,sie,nuext,f,model,w_e,w_i, mu, ch(ii),nu0Ib,perturb,which_p,st_noise,cn_level,t_vect,t_vect2,0);
            dt = sing_trial.time(2)-sing_trial.time(1);
            b4_stim  = sing_trial.time > (t_vect(1)-500)/1000 & sing_trial.time <= t_vect(1)/1000; 
            stim_per = sing_trial.time > t_vect(1)/1000 & sing_trial.time <= t_vect(2)/1000; 
            if abs(sing_trial.f1(b4_stim) - sing_trial.f2(b4_stim)) < 5
                if (sum((sing_trial.f1(stim_per) - sing_trial.f2(stim_per) > 15)) > 1)...
                        && ((sing_trial.f1(sing_trial.time == t_vect(2)/1000) - sing_trial.f2(sing_trial.time == t_vect(2)/1000)) > 15)
                    choose(ii,jj) = 1;
                    uu = find( (sing_trial.f1(stim_per) - sing_trial.f2(stim_per)) > 15);
                    vv = diff(uu);
                    ww = find(vv>1);
                    if ~isempty(ww)
                        rt_raww(ii,jj) = uu(ww(end)+1)*dt;
                    else
                        rt_raww(ii,jj) = uu(1)*dt;
                    end
                    if  ch(ii) >= 0
                        rt_corr(ii,jj) = rt_raww(ii,jj);
                        rt_icor(ii,jj) = nan;
                    else
                        rt_corr(ii,jj) = nan;
                        rt_icor(ii,jj) = rt_raww(ii,jj);
                    end
                elseif (sum((sing_trial.f1(stim_per) - sing_trial.f2(stim_per) < -15)) > 1)...
                        && ((sing_trial.f1(sing_trial.time == t_vect(2)/1000) - sing_trial.f2(sing_trial.time == t_vect(2)/1000)) < -15)
                    uu = find( (sing_trial.f1(stim_per) - sing_trial.f2(stim_per)) < -15);
                    choose(ii,jj) = 0;
                    vv = diff(uu);
                    ww = find(vv>1);
                    if ~isempty(ww)
                        rt_raww(ii,jj) = uu(ww(end)+1)*dt;
                    else
                        rt_raww(ii,jj) = uu(1)*dt;
                    end
                    if  ch(ii) <= 0
                        rt_corr(ii,jj) = rt_raww(ii,jj);
                        rt_icor(ii,jj) = nan;
                    else
                        rt_corr(ii,jj) = nan;
                        rt_icor(ii,jj) = rt_raww(ii,jj);
                    end
                else
                    choose(ii,jj) = nan;
                    rt_raww(ii,jj) = nan;
                    rt_corr(ii,jj) = nan;
                    rt_icor(ii,jj) = nan;
                end
            else
                choose(ii,jj) = nan;
                rt_raww(ii,jj) = nan;
                rt_corr(ii,jj) = nan;
                rt_icor(ii,jj) = nan;                          
            end
        end
        waitbar(ii/length(ch),wtb)
    end
    close(wtb)
    if ~exist('uu','var')
        uu = [];
    end
    %sum(choose,2)/no_runs
    %disp(num2str(size(   nansum(choose,2) ./ sum(~isnan(choose),2))   ))
    results.ch     = ch;
    results.choose = choose;
    results.ntrial = sum(~isnan(choose),2);
    results.psych  = nansum(choose,2)./sum(~isnan(choose),2);
    results.rt_raw = nanmean(rt_raww,2);
    results.re_raw = nanstd(rt_raww,0,2)./sqrt(sum(~isnan(rt_raww),2));
    results.rt_cor = nanmean(rt_corr,2);
    results.re_cor = nanstd(rt_corr,0,2)./sqrt(sum(~isnan(rt_corr),2));
    results.rt_ico = nanmean(rt_icor,2);
    results.re_ico = nanstd(rt_icor,0,2)./sqrt(sum(~isnan(rt_icor),2));
    results.pn     = sum(~isnan(choose),2);
    results.rn     = sum(~isnan(rt_raww),2);
end