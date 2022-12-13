clear
option = 'four';
boot_opt = 'one';
n_boot = 150;
thresh = 0.5;
f = uipickfiles('FilterSpec','/Users/roach/work/RNNs/');
delta_sel = zeros(length(f),4,2);

date_str = int2str(fix(clock));
date_str(date_str==' ') = '_';
date_str = [option,'_',boot_opt,'_',int2str(n_boot),'_',date_str];
date_str = [];

for oo = 1:length(f)
    try
        load([f{oo},'/data.mat'])
        if perfT_hist(end) > 0.85
        %     load([f{oo},'/resim_data.mat'])
            %%
            trial = 125;

            fig1 = figure;

            ax1 = axes('Parent',fig1,'Position',[0.15 0.775 0.8 0.2]);
            plot(squeeze(val_inps(trial,:,:)),'LineWidth',2.0)
            ax2 = axes('Parent',fig1,'Position',[0.15 0.55 0.8 0.2]);
            plot(squeeze(val_act_e(trial,:,:)),'LineWidth',2.0)
            ax3 = axes('Parent',fig1,'Position',[0.15 0.325 0.8 0.2]);

            xx = find(val_mask(trial,:,1)~=0);
            plot(xx,squeeze(val_outs(trial,xx,1)),'.'), hold on
            plot(xx,squeeze(val_outs(trial,xx,2)),'.')
            ax4 = axes('Parent',fig1,'Position',[0.15 0.1 0.8 0.2]);
            plot(squeeze(val_pred(trial,:,:)),'LineWidth',2.0)
            xlabel('time [AU]')
            set(ax1,'FontName','Arial','FontSize',16,'XTick',[])
            set(ax2,'FontName','Arial','FontSize',16,'XTick',[])
            set(ax3,'FontName','Arial','FontSize',16,'XTick',[])
            set(ax4,'FontName','Arial','FontSize',16)
            print(fig1,[f{oo},'/trial_',int2str(trial),'_act',date_str,'.eps'],'-depsc','-painters')

            %%
            dz_out      = zeros(size(val_pred,1),size(val_pred,2));
            trl_val_pre = zeros(size(val_pred,1),1); 
            trl_val_pst = zeros(size(val_pred,1),1); 
            h_act_e     = zeros(size(val_pred,1),size(val_act_e,3));
            h_act_i     = zeros(size(val_pred,1),size(val_act_i,3));
            choose_1    = zeros(1,size(val_pred,1)); 
            choose_2    = zeros(1,size(val_pred,1)); 
            for ii = 1:size(val_pred,1)
            %     dz_out(ii) = squeeze(val_pred(ii,trial_end_val(ii)+10,1) - val_pred(ii,trial_end_val(ii)+10,2));
                dz_out(ii,:) = squeeze(val_pred(ii,:,1) - val_pred(ii,:,2));
                h_act_e(ii,:) = val_act_e(ii,trial_end_val(ii)+1,:);
                h_act_i(ii,:) = val_act_i(ii,trial_end_val(ii)+1,:);

                trl_val_pre(ii) = sum(abs(dz_out(ii,1:(trial_starts_val(ii)))) < 0.25)/length(abs(dz_out(ii,1:(trial_starts_val(ii)))));
                trl_val_pst(ii) = sum(abs(dz_out(ii,(trial_end_val(ii)):end)) > 0.25)/length(abs(dz_out(ii,(trial_end_val(ii)):end)));

                 %----- option 2 ------%
                 switch option
        %         choose_1(ii)  = dz_out(ii,trial_end_val(ii)+1)' > 0.6;
        %         choose_2(ii)  = dz_out(ii,trial_end_val(ii)+1)' < -0.6;
                     case 'two'
                        if val_pred(ii,trial_end_val(ii)+1,1) > val_pred(ii,trial_end_val(ii)+1,2)
                            choose_1(ii)  = 1;
                        else
                            choose_2(ii)  = 1;
                        end
                 end

        %         valid_trial =  trl_val_pre >= 0.75 & trl_val_pst >= 0.5 ;
                %----- option 1 ------%
            %     choose_1 = dz_out > 15;
            %     choose_1  = dz_out(:,end) > 0.6;
            %     choose_2  = dz_out(:,end) < -0.6;
            %     valid_trial = abs(dz_out(:,end)) > 0.6;

            end
            switch option
            %----- option 2 ------%
        %     choose_1  = dz_out(:,trial_end_val)' > 0.6;
        %     choose_2  = dz_out(:,trial_end_val)' < -0.6;
                case 'two'
                    valid_trial =  trl_val_pre >= 0.75 & trl_val_pst >= 0.5 ;
                case 'one'
            %----- option 1 ------%
        %             choose_1 = dz_out > 15;
                    choose_1  = dz_out(:,end) > 0.6;
                    choose_2  = dz_out(:,end) < -0.6;
                    valid_trial = abs(dz_out(:,end)) > 0.6;
                case 'three'
                    choose_1 = dz_out(:,end) > 0.0;
                    choose_2 = dz_out(:,end) < 0.0;
        %             valid_trial =  trl_val_pre >= 0.75 & trl_val_pst >= 0.5 ;
        %             valid_trial = ones(size(trl_val_pre));
                    valid_trial = dz_out(:,end) ~= 0;
                case 'four'
                    choose_1 = choice == 0;
                    choose_2 = choice == 1;
                    valid_trial = ~isnan(choice');
            end

            choose_1 = logical(choose_1);
            choose_2 = logical(choose_2);

            p_ch1 = zeros(length(psycho_b),1);

            for ii = 1:length(psycho_b)
                xx = (val_coh == psycho_b(ii));
                x = xx & valid_trial';
                p_ch1(ii) = sum(choose_1(x))/sum(x);    
            end

            if size(choose_1,2) == 1
                choose_1 = choose_1';
                choose_2 = choose_2';
            end


            fig1 = figure;
            plot(psycho_b,p_ch1,'k','LineWidth',12)
            hold on
            plot(psycho_b,psycho,'r','LineWidth',12)
            set(gca,'FontSize',16,'FontName','Arial'), xlabel('Stim'), ylabel('P(Choose 1)')
            print(fig1,[f{oo},'/two_psychos',date_str,'.eps'],'-depsc','-painters')

        %     choose_1 = dz_out > 0.6;

            auc_e = zeros(ne,2);
            auc_i = zeros(ni,2);
            d_prime_e = zeros(ne,2);
            d_prime_i = zeros(ni,2);

        %         focus on valid trials
            vchoose_1 = logical(choose_1(valid_trial));
            vchoose_2 = logical(choose_2(valid_trial));
            vh_act_e  = h_act_e(valid_trial,:);
            vh_act_i  = h_act_i(valid_trial,:);
            shuffle_acte = reshape(vh_act_e,1,[]);
            shuffle_acti = reshape(vh_act_i,1,[]);

            for ii = 1:ne        
                [TPR,FPR] = roc(vchoose_1,vh_act_e(:,ii)');
                auc_e(ii,1) = trapz([0,FPR,1],[0,TPR,1]);
                d_prime_e(ii,1) = mean(vh_act_e(vchoose_1==1,ii)) - mean(vh_act_e(~vchoose_1,ii))/sqrt((var(vh_act_e(vchoose_1,ii))+var(vh_act_e(~vchoose_1,ii)))/2);
                auc_boot = zeros(n_boot,1);
                d_prime_boot = zeros(n_boot,1);
                for jj = 1:n_boot
                    if strcmp(boot_opt,'one')
                        xx = randperm(size(vchoose_1,2));
                        [TPR,FPR] = roc(vchoose_1(xx),vh_act_e(:,ii)');
                    elseif strcmp(boot_opt,'two')
                        xx = randperm(size(shuffle_acte,2));
                        [TPR,FPR] = roc(vchoose_1,shuffle_acte(xx(1:size(vchoose_1,2))));
                    end
                    auc_boot(jj) = trapz([0,FPR,1],[0,TPR,1]);
        %             d_prime_boot(jj) = mean(vh_act_e(vchoose_1(xx),ii)) - mean(vh_act_e(~vchoose_1(xx),ii))/sqrt((var(vh_act_e(vchoose_1(xx),ii))+var(vh_act_e(~vchoose_1(xx),ii)))/2);
                end
        %         if sum(d_prime_e(ii,1) > d_prime_boot)/50 > 0.975
        %             d_prime_e(ii,2) = 1;
        %         elseif sum(d_prime_e(ii,1) > d_prime_boot)/50 < 0.025
        %             d_prime_e(ii,2) = -1;
        %         end

                if sum(auc_e(ii,1) > auc_boot)/n_boot > 0.975 && auc_e(ii,1) > thresh
                    auc_e(ii,2) = 1;
                elseif sum(auc_e(ii,1) > auc_boot)/n_boot < 0.025 && auc_e(ii,1) ~= 0.5 && auc_e(ii,1) < (1-thresh)
                    auc_e(ii,2) = -1;
                end
            end


            for ii = 1:ni
                [TPR,FPR] = roc(vchoose_1,vh_act_i(:,ii)');
                auc_i(ii,1) = trapz([0,FPR,1],[0,TPR,1]);
                d_prime_i(ii,1) = mean(vh_act_i(vchoose_1,ii)) - mean(vh_act_i(~vchoose_1,ii))/sqrt((var(vh_act_i(vchoose_1,ii))+var(vh_act_i(~vchoose_1,ii)))/2);
                auc_boot = zeros(n_boot,1);
                d_prime_boot = zeros(n_boot,1);
                for jj = 1:n_boot
                    if strcmp(boot_opt,'one')
                        xx = randperm(size(vchoose_1,2));
                        [TPR,FPR] = roc(vchoose_1(xx),vh_act_i(:,ii)');
                    elseif strcmp(boot_opt,'two')
                        xx = randperm(size(shuffle_acti,2));
                        [TPR,FPR] = roc(vchoose_1,shuffle_acti(xx(1:size(vchoose_1,2))));
                    end
                    auc_boot(jj) = trapz([0,FPR,1],[0,TPR,1]);
        %             d_prime_boot(jj) = mean(vh_act_i(vchoose_1(xx),ii)) - mean(vh_act_i(~vchoose_1(xx),ii))/sqrt((var(vh_act_i(vchoose_1(xx),ii))+var(vh_act_i(~vchoose_1(xx),ii)))/2);
                end
        %         if sum(d_prime_i(ii,1) > d_prime_boot)/50 > 0.975
        %             d_prime_i(ii,2) = 1;
        %         elseif sum(d_prime_i(ii,1) > d_prime_boot)/50 < 0.025
        %             d_prime_i(ii,2) = -1;
        %         end

                if (sum(auc_i(ii,1) > auc_boot)/n_boot > 0.975) && auc_i(ii,1) > thresh
                    auc_i(ii,2) = 1;
                elseif sum(auc_i(ii,1) > auc_boot)/n_boot < 0.025 && auc_i(ii,1) ~= 0.5 && auc_i(ii,1) < (1-thresh)
                    auc_i(ii,2) = -1;
                end
            end
            %%
            fig1 = figure;
        %     lcoh = find(val_coh<=-0);
        %     rcoh = find(val_coh>=0);
            hold on
            plot(squeeze(mean(val_act_e(choose_1&valid_trial',:,auc_e(:,2)==-1),1)) - squeeze(mean(val_act_e(choose_2&valid_trial',:,auc_e(:,2)==-1),1)),'Color',[1 0 0],'Linewidth',2.0) % one
            plot(squeeze(mean(val_act_e(choose_1&valid_trial',:,auc_e(:,2)==1),1)) - squeeze(mean(val_act_e(choose_2&valid_trial',:,auc_e(:,2)==1),1)),'Color',[0 0 1],'Linewidth',2.0) % two   
            plot(squeeze(mean(val_act_e(choose_1&valid_trial',:,auc_e(:,2)==0),1)) - squeeze(mean(val_act_e(choose_2&valid_trial',:,auc_e(:,2)==0),1)),'Color',[0.7 0.7 0.7],'Linewidth',2.0) % none
            set(gca,'FontSize',16,'FontName','Arial'), xlabel('time (AU)'), ylabel('\Delta r_{t}')
            print(fig1,[f{oo},'/delta_acte',date_str,'.eps'],'-depsc','-painters')

            fig1 = figure;
        %     lcoh = find(val_coh<=-0);
        %     rcoh = find(val_coh>=0);
            hold on
            plot(squeeze(mean(val_act_i(choose_1&valid_trial',:,auc_i(:,2)==-1),1)) - squeeze(mean(val_act_i(choose_2&valid_trial',:,auc_i(:,2)==-1),1)),'Color',[1 0 0],'Linewidth',2.0) % one 
            plot(squeeze(mean(val_act_i(choose_1&valid_trial',:,auc_i(:,2)==1),1)) - squeeze(mean(val_act_i(choose_2&valid_trial',:,auc_i(:,2)==1),1)),'Color',[0 0 1],'Linewidth',2.0) %two
            plot(squeeze(mean(val_act_i(choose_1&valid_trial',:,auc_i(:,2)==0),1)) - squeeze(mean(val_act_i(choose_2&valid_trial',:,auc_i(:,2)==0),1)),'Color',[0.7 0.7 0.7],'Linewidth',2.0) % none
            set(gca,'FontSize',16,'FontName','Arial'), xlabel('time (AU)'), ylabel('\Delta r_{t}')
            print(fig1,[f{oo},'/delta_acti',date_str,'.eps'],'-depsc','-painters')

            %%
            fig1 = figure;
            [~,ie] = sort(auc_e(:,1));
            [~,ii] = sort(auc_i(:,1));
        %     ii = ii+100;
            subplot(2,2,1),imagesc(log10(ree_weights(ie,ie))),axis xy
            subplot(2,2,2),imagesc(log10(rei_weights(ii,ie))),axis xy
            subplot(2,2,3),imagesc(log10(rie_weights(ie,ii))),axis xy
            subplot(2,2,4),imagesc(log10(rii_weights(ii,ii))),axis xy
            cb1 = colorbar; ylabel(cb1,'W_{rec}')
            set(gca,'FontName','Arial','FontSize',16)
            xlabel('From neuron'), ylabel('To neuron')
            print(fig1,[f{oo},'/sorted_weights',date_str,'.eps'],'-depsc','-painters')
            %%
        %     fig1 = figure;
        %     [~,ie] = sort(d_prime(1:100,1));
        %     [~,ii] = sort(d_prime(101:125,1));
        %     ii = ii+100;
        %     imagesc(rnn_weights([ie;ii],[ie;ii])),axis xy
        %     imagesc(log10(abs(rnn_weights([ie;ii],[ie;ii])))),axis xy
        %     cb1 = colorbar; ylabel(cb1,'W_{rec}')
        %     set(gca,'FontName','Arial','FontSize',16)
        %     xlabel('From neuron'), ylabel('To neuron')
        %     print(fig1,[f{oo},'/sorted_log_weights.eps'],'-depsc','-painters')
            %%

            if ~exist('save_steps','var')
                save_steps = 1;
            end
            choice_tunedE = abs(auc_e(:,1)-0.5);
            choice_tunedI = abs(auc_i(:,1)-0.5);
            
            xxEE = choice_tunedE*choice_tunedE';
            xxEE = xxEE./max(max(xxEE));
            
            xxEI = choice_tunedE*choice_tunedI';
            xxEI = xxEI./max(max(xxEI));
            
            xxIE = choice_tunedI*choice_tunedE';
            xxIE = xxIE./max(max(xxIE));
            
            xxII = choice_tunedI*choice_tunedI';
            xxII = xxII./max(max(xxII));
            
            ee_same = zeros(size(wee_hist,3),4);
            ee_diff = zeros(size(wee_hist,3),4);

            ei_same = zeros(size(wei_hist,3),4);
            ei_diff = zeros(size(wei_hist,3),4);

            ie_same = zeros(size(wie_hist,3),4);
            ie_diff = zeros(size(wie_hist,3),4);

            ii_same = zeros(size(wii_hist,3),4);
            ii_diff = zeros(size(wii_hist,3),4);

            sigma_ee = zeros(size(wie_hist,3),4);
            sigma_ei = zeros(size(wie_hist,3),4);
            sigma_ie = zeros(size(wie_hist,3),4);
            sigma_ii = zeros(size(wii_hist,3),4);



            cell_id       = 1:125;
            sel_sim_ee     = (auc_e(:,2) * auc_e(:,2)').*(1-eye(ne));
            sel_sim_ei     = (auc_e(:,2) * auc_i(:,2)')';
            sel_sim_ie     = (auc_e(:,2) * auc_i(:,2)');
            sel_sim_ii     = (auc_i(:,2) * auc_i(:,2)').*(1-eye(ni));

            n_shuff = 500;

            ee_same_shuffles = zeros(size(wie_hist,3),n_shuff,2);
            ee_diff_shuffles = zeros(size(wie_hist,3),n_shuff,2);
            ei_same_shuffles = zeros(size(wie_hist,3),n_shuff,2);
            ei_diff_shuffles = zeros(size(wie_hist,3),n_shuff,2);
            ie_same_shuffles = zeros(size(wie_hist,3),n_shuff,2);
            ie_diff_shuffles = zeros(size(wie_hist,3),n_shuff,2);
            ii_same_shuffles = zeros(size(wii_hist,3),n_shuff,2);
            ii_diff_shuffles = zeros(size(wii_hist,3),n_shuff,2);

            sel_sim_eeshuff = zeros(ne,ne,n_shuff);
            sel_sim_eishuff = zeros(ni,ne,n_shuff);
            sel_sim_ieshuff = zeros(ne,ni,n_shuff);
            sel_sim_iishuff = zeros(ni,ni,n_shuff);

            shuff_sig_ee = zeros(size(wee_hist,3),n_shuff,2);
            shuff_sig_ei = zeros(size(wee_hist,3),n_shuff,2);
            shuff_sig_ie = zeros(size(wee_hist,3),n_shuff,2);
            shuff_sig_ii = zeros(size(wee_hist,3),n_shuff,2);

            for ii=1:n_shuff
                xx = randperm(ne);
                yy = randperm(ni);
                sel_sim_eeshuff(:,:,ii) = (auc_e(xx,2) * auc_e(xx,2)').*(1-eye(ne));
                sel_sim_eishuff(:,:,ii) = (auc_e(xx,2) * auc_i(yy,2)')';
                sel_sim_ieshuff(:,:,ii) = (auc_e(xx,2) * auc_i(yy,2)');
                sel_sim_iishuff(:,:,ii) = (auc_i(yy,2) * auc_i(yy,2)').*(1-eye(ni));
            end
        %     sel_sim_iishuff = auc_i(randperm(ni),2) * auc_i(randperm(ni),2)';

        %     wee = wr_hist(1:100,1:100,:);
        %     wei = wr_hist(101:125,1:100,:);
        %     wie = wr_hist(1:100,101:125,:);
        %     wii = wr_hist(101:125,101:125,:);
        % 
            w_ex = zeros(size(wee_hist,3),1);
            w_ih = zeros(size(wee_hist,3),1);

            for ii = 1:size(wee_hist,3)
                wwee = wee_hist(:,:,ii);
                wwei = wei_hist(:,:,ii);
                wwie = wie_hist(:,:,ii);
                wwii = wie_hist(:,:,ii);

                ee_same(ii,1) = mean2(wwee(sel_sim_ee==1));
                ee_diff(ii,1) = mean2(wwee(sel_sim_ee==-1));

                ei_same(ii,1) = mean2(wwei(sel_sim_ei==1));
                ei_diff(ii,1) = mean2(wwei(sel_sim_ei==-1));

                ie_same(ii,1) = mean2(wwie(sel_sim_ie==1));
                ie_diff(ii,1) = mean2(wwie(sel_sim_ie==-1));

                ii_same(ii,1) = mean2(wwie(sel_sim_ii==1));
                ii_diff(ii,1) = mean2(wwie(sel_sim_ii==-1));
                
                ee_same(ii,4) = mean2(wwee(sel_sim_ee==1).*xxEE(sel_sim_ee==1));
                ee_diff(ii,4) = mean2(wwee(sel_sim_ee==-1).*xxEE(sel_sim_ee==-1));

                ei_same(ii,4) = mean2(wwei(sel_sim_ei==1).*xxEI(sel_sim_ei==1));
                ei_diff(ii,4) = mean2(wwei(sel_sim_ei==-1).*xxEI(sel_sim_ei==-1));

                ie_same(ii,4) = mean2(wwie(sel_sim_ie==1).*xxIE(sel_sim_ie==1));
                ie_diff(ii,4) = mean2(wwie(sel_sim_ie==-1).*xxIE(sel_sim_ie==-1));

                ii_same(ii,4) = mean2(wwie(sel_sim_ii==1).*xxII(sel_sim_ii==1));
                ii_diff(ii,4) = mean2(wwie(sel_sim_ii==-1).*xxII(sel_sim_ii==-1));

                sigma_ee(ii,1) = (ee_same(ii,1)-ee_diff(ii,1))./(ee_same(ii,1)+ee_diff(ii,1));
                sigma_ei(ii,1) = (ei_same(ii,1)-ei_diff(ii,1))./(ei_same(ii,1)+ei_diff(ii,1));
                sigma_ie(ii,1) = (ie_same(ii,1)-ie_diff(ii,1))./(ie_same(ii,1)+ie_diff(ii,1));
                sigma_ii(ii,1) = (ii_same(ii,1)-ii_diff(ii,1))./(ii_same(ii,1)+ii_diff(ii,1));
                
                sigma_ee(ii,4) = (ee_same(ii,4)-ee_diff(ii,4))./(ee_same(ii,4)+ee_diff(ii,4));
                sigma_ei(ii,4) = (ei_same(ii,4)-ei_diff(ii,4))./(ei_same(ii,4)+ei_diff(ii,4));
                sigma_ie(ii,4) = (ie_same(ii,4)-ie_diff(ii,4))./(ie_same(ii,4)+ie_diff(ii,4));
                sigma_ii(ii,4) = (ii_same(ii,4)-ii_diff(ii,4))./(ii_same(ii,4)+ii_diff(ii,4));

                for jj = 1:n_shuff
                    ee_same_shuffles(ii,jj,1) = mean2(wwee(sel_sim_eeshuff(:,:,jj)==1));
                    ee_diff_shuffles(ii,jj,1) = mean2(wwee(sel_sim_eeshuff(:,:,jj)==-1));
                    ei_same_shuffles(ii,jj,1) = mean2(wwei(sel_sim_eishuff(:,:,jj)==1));
                    ei_diff_shuffles(ii,jj,1) = mean2(wwei(sel_sim_eishuff(:,:,jj)==-1));
                    ie_same_shuffles(ii,jj,1) = mean2(wwie(sel_sim_ieshuff(:,:,jj)==1));
                    ie_diff_shuffles(ii,jj,1) = mean2(wwie(sel_sim_ieshuff(:,:,jj)==-1));
                    ii_same_shuffles(ii,jj,1) = mean2(wwii(sel_sim_iishuff(:,:,jj)==1));
                    ii_diff_shuffles(ii,jj,1) = mean2(wwii(sel_sim_iishuff(:,:,jj)==-1));
                    
                    ee_same_shuffles(ii,jj,2) = mean2(wwee(sel_sim_eeshuff(:,:,jj)==1).*xxEE(sel_sim_eeshuff(:,:,jj)==1));
                    ee_diff_shuffles(ii,jj,2) = mean2(wwee(sel_sim_eeshuff(:,:,jj)==-1).*xxEE(sel_sim_eeshuff(:,:,jj)==-1));
                    ei_same_shuffles(ii,jj,2) = mean2(wwei(sel_sim_eishuff(:,:,jj)==1).*xxEI(sel_sim_eishuff(:,:,jj)==1));
                    ei_diff_shuffles(ii,jj,2) = mean2(wwei(sel_sim_eishuff(:,:,jj)==-1).*xxEI(sel_sim_eishuff(:,:,jj)==-1));
                    ie_same_shuffles(ii,jj,2) = mean2(wwie(sel_sim_ieshuff(:,:,jj)==1).*xxIE(sel_sim_ieshuff(:,:,jj)==1));
                    ie_diff_shuffles(ii,jj,2) = mean2(wwie(sel_sim_ieshuff(:,:,jj)==-1).*xxIE(sel_sim_ieshuff(:,:,jj)==-1));
                    ii_same_shuffles(ii,jj,2) = mean2(wwii(sel_sim_iishuff(:,:,jj)==1).*xxII(sel_sim_iishuff(:,:,jj)==1));
                    ii_diff_shuffles(ii,jj,2) = mean2(wwii(sel_sim_iishuff(:,:,jj)==-1).*xxII(sel_sim_iishuff(:,:,jj)==-1));
                end

                shuff_sig_ee(ii,:,1) = (ee_same_shuffles(ii,:,1) - ee_diff_shuffles(ii,:,1))./(ee_same_shuffles(ii,:,1) + ee_diff_shuffles(ii,:,1));
                shuff_sig_ei(ii,:,1) = (ei_same_shuffles(ii,:,1) - ei_diff_shuffles(ii,:,1))./(ei_same_shuffles(ii,:,1) + ei_diff_shuffles(ii,:,1));
                shuff_sig_ie(ii,:,1) = (ie_same_shuffles(ii,:,1) - ie_diff_shuffles(ii,:,1))./(ie_same_shuffles(ii,:,1) + ie_diff_shuffles(ii,:,1));
                shuff_sig_ii(ii,:,1) = (ii_same_shuffles(ii,:,1) - ii_diff_shuffles(ii,:,1))./(ii_same_shuffles(ii,:,1) + ii_diff_shuffles(ii,:,1));
                
                shuff_sig_ee(ii,:,2) = (ee_same_shuffles(ii,:,2) - ee_diff_shuffles(ii,:,2))./(ee_same_shuffles(ii,:,2) + ee_diff_shuffles(ii,:,2));
                shuff_sig_ei(ii,:,2) = (ei_same_shuffles(ii,:,2) - ei_diff_shuffles(ii,:,2))./(ei_same_shuffles(ii,:,2) + ei_diff_shuffles(ii,:,2));
                shuff_sig_ie(ii,:,2) = (ie_same_shuffles(ii,:,2) - ie_diff_shuffles(ii,:,2))./(ie_same_shuffles(ii,:,2) + ie_diff_shuffles(ii,:,2));
                shuff_sig_ii(ii,:,2) = (ii_same_shuffles(ii,:,2) - ii_diff_shuffles(ii,:,2))./(ii_same_shuffles(ii,:,2) + ii_diff_shuffles(ii,:,2));


                ee_same(ii,2) = mean(ee_same_shuffles(ii,:,1));
                ee_diff(ii,2) = mean(ee_same_shuffles(ii,:,1));

                ei_same(ii,2) = mean(ei_same_shuffles(ii,:,1));
                ei_diff(ii,2) = mean(ei_same_shuffles(ii,:,1));

                ie_same(ii,2) = mean(ie_same_shuffles(ii,:,1));
                ie_diff(ii,2) = mean(ie_same_shuffles(ii,:,1));

                ii_same(ii,2) = mean(ii_same_shuffles(ii,:,1));
                ii_diff(ii,2) = mean(ii_same_shuffles(ii,:,1));

                ee_same(ii,3) = std(ee_same_shuffles(ii,:,1))/sqrt(n_shuff);
                ee_diff(ii,3) = std(ee_same_shuffles(ii,:,1))/sqrt(n_shuff);

                ei_same(ii,3) = std(ei_same_shuffles(ii,:,1))/sqrt(n_shuff);
                ei_diff(ii,3) = std(ei_same_shuffles(ii,:,1))/sqrt(n_shuff);

                ie_same(ii,3) = std(ie_same_shuffles(ii,:,1))/sqrt(n_shuff);
                ie_diff(ii,3) = std(ie_same_shuffles(ii,:,1))/sqrt(n_shuff);

                ii_same(ii,3) = std(ii_same_shuffles(ii,:,1))/sqrt(n_shuff);
                ii_diff(ii,3) = std(ii_same_shuffles(ii,:,1))/sqrt(n_shuff);
                
                ee_same(ii,5) = mean(ee_same_shuffles(ii,:,2));
                ee_diff(ii,5) = mean(ee_same_shuffles(ii,:,2));

                ei_same(ii,5) = mean(ei_same_shuffles(ii,:,2));
                ei_diff(ii,5) = mean(ei_same_shuffles(ii,:,2));

                ie_same(ii,5) = mean(ie_same_shuffles(ii,:,2));
                ie_diff(ii,5) = mean(ie_same_shuffles(ii,:,2));

                ii_same(ii,5) = mean(ii_same_shuffles(ii,:,2));
                ii_diff(ii,5) = mean(ii_same_shuffles(ii,:,2));

                ee_same(ii,6) = std(ee_same_shuffles(ii,:,2))/sqrt(n_shuff);
                ee_diff(ii,6) = std(ee_same_shuffles(ii,:,2))/sqrt(n_shuff);

                ei_same(ii,6) = std(ei_same_shuffles(ii,:,2))/sqrt(n_shuff);
                ei_diff(ii,6) = std(ei_same_shuffles(ii,:,2))/sqrt(n_shuff);

                ie_same(ii,6) = std(ie_same_shuffles(ii,:,2))/sqrt(n_shuff);
                ie_diff(ii,6) = std(ie_same_shuffles(ii,:,2))/sqrt(n_shuff);

                ii_same(ii,6) = std(ii_same_shuffles(ii,:,2))/sqrt(n_shuff);
                ii_diff(ii,6) = std(ii_same_shuffles(ii,:,2))/sqrt(n_shuff);

                sigma_ee(ii,2) = mean(shuff_sig_ee(ii,:,1));
                sigma_ei(ii,2) = mean(shuff_sig_ei(ii,:,1));
                sigma_ie(ii,2) = mean(shuff_sig_ie(ii,:,1));
                sigma_ii(ii,2) = mean(shuff_sig_ii(ii,:,1));

                sigma_ee(ii,3) = std(shuff_sig_ee(ii,:,1))/sqrt(n_shuff);
                sigma_ei(ii,3) = std(shuff_sig_ei(ii,:,1))/sqrt(n_shuff);
                sigma_ie(ii,3) = std(shuff_sig_ie(ii,:,1))/sqrt(n_shuff);
                sigma_ii(ii,3) = std(shuff_sig_ii(ii,:,1))/sqrt(n_shuff);
                
                sigma_ee(ii,5) = mean(shuff_sig_ee(ii,:,2));
                sigma_ei(ii,5) = mean(shuff_sig_ei(ii,:,2));
                sigma_ie(ii,5) = mean(shuff_sig_ie(ii,:,2));
                sigma_ii(ii,5) = mean(shuff_sig_ii(ii,:,2));

                sigma_ee(ii,6) = std(shuff_sig_ee(ii,:,2))/sqrt(n_shuff);
                sigma_ei(ii,6) = std(shuff_sig_ei(ii,:,2))/sqrt(n_shuff);
                sigma_ie(ii,6) = std(shuff_sig_ie(ii,:,2))/sqrt(n_shuff);
                sigma_ii(ii,6) = std(shuff_sig_ii(ii,:,2))/sqrt(n_shuff);

        %         ww = wii_hist(:,:,ii);
        %         ii_same(ii,1) = mean2(ww(sel_sim_ii==1));
        %         ii_diff(ii,1) = mean2(ww(sel_sim_ii==-1));
        % 
        %         ii_same(ii,2) = mean2(ww(sel_sim_iishuff==1));
        %         ii_diff(ii,2) = mean2(ww(sel_sim_iishuff==-1));

                w_ex(ii) = mean([mean2(wee_hist(:,:,ii)),mean2(wei_hist(:,:,ii))]);
                w_ih(ii) = mean([mean2(wie_hist(:,:,ii)),wii_hist(ii)]);
            end
            t = double(save_steps)*(0:(size(wee_hist,3)-1));
            fig1 = figure; hold on

        %     plot(t,w_ex,'LineWidth',2.0,'Color',[0.6 0.6 0.6])
        %     plot(t,w_ih,'LineWidth',2.0,'Color',[0.3 0.3 0.3])

            plot(t,ee_same(:,1),'r','LineWidth',2.0)
            plot(t,ee_diff(:,1),'r--','LineWidth',2.0)

            plot(t,ei_same(:,1),'k','LineWidth',2.0)
            plot(t,ei_diff(:,1),'k--','LineWidth',2.0)

            plot(t,ie_same(:,1),'b','LineWidth',2.0)
            plot(t,ie_diff(:,1),'b--','LineWidth',2.0)

            plot(t,ii_same(:,1),'c','LineWidth',2.0)
            plot(t,ii_diff(:,1),'c--','LineWidth',2.0)
            xlabel('traning epoch'), ylabel('Mean Synaptic Strength'), set(gca,'FontSize',16,'FontName','Arial')

            dtstr = fix(clock);
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/evo_synsel',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/evo_synsel',date_str,'.eps'],'-depsc','-painters')
            end
            %%
            t = double(save_steps)*(0:(size(wee_hist,3)-1));
            fig1 = figure; hold on

            plot(t,w_ex,'LineWidth',2.0,'Color',[0.6 0.6 0.6])
            plot(t,w_ih,'LineWidth',2.0,'Color',[0.3 0.3 0.3])
            xlabel('traning epoch'), ylabel('Mean Synaptic Strength'), set(gca,'FontSize',16,'FontName','Arial')
            dtstr = fix(clock);
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/evo_meansyn',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/evo_meansyn',date_str,'.eps'],'-depsc','-painters')
            end

            %%
            fig1 = figure; hold on

            % plot(t,w_ex,'LineWidth',2.0,'Color',[0.6 0.6 0.6])
            % plot(t,w_ih,'LineWidth',2.0,'Color',[0.3 0.3 0.3])
        %     clear sigma_ee sigma_ei sigma_ie sigma_ii
        %     sigma_ee(:,1) = (ee_same(:,1)-ee_diff(:,1))./(ee_same(:,1)+ee_diff(:,1));
        %     sigma_ee(:,2) = (ee_same(:,2)-ee_diff(:,2))./(ee_same(:,2)+ee_diff(:,2));
        %     
        %     sigma_ei(:,1) = (ei_same(:,1)-ei_diff(:,1))./(ei_same(:,1)+ei_diff(:,1));
        %     sigma_ei(:,2) = (ei_same(:,2)-ei_diff(:,2))./(ei_same(:,2)+ei_diff(:,2));
        %     
        %     sigma_ie(:,1) = (ie_same(:,1)-ie_diff(:,1))./(ie_same(:,1)+ie_diff(:,1));
        %     sigma_ie(:,2) = (ie_same(:,2)-ie_diff(:,2))./(ie_same(:,2)+ie_diff(:,2));

        %     sigma_ii(:,1) = (ii_same(:,1)-ii_diff(:,1))./(ii_same(:,1)+ii_diff(:,1));
        %     sigma_ii(:,2) = (ii_same(:,2)-ii_diff(:,2))./(ii_same(:,2)+ii_diff(:,2));
        %     plot(t,(ii_same(:,1)-ii_diff(:,1))./(ii_same(:,1)+ii_diff(:,1)),'c','LineWidth',2.0)

        %     plot(t,sigma_ee(:,2) + sigma_ee(:,3),'r--','LineWidth',2.0)
        % 
        %     plot(t,sigma_ei(:,2) + sigma_ei(:,3),'k--','LineWidth',2.0)
        % 
        %     plot(t,sigma_ie(:,2) + sigma_ie(:,3),'b--','LineWidth',2.0)
        % 
        %     plot(t,sigma_ee(:,2) - sigma_ee(:,3),'r--','LineWidth',2.0)
        % 
        %     plot(t,sigma_ei(:,2) - sigma_ei(:,3),'k--','LineWidth',2.0)
        % 
        %     plot(t,sigma_ie(:,2) - sigma_ie(:,3),'b--','LineWidth',2.0)    

            shadedErrorBar(t,sigma_ee(:,2),sigma_ee(:,3),'lineProps','r') 
            shadedErrorBar(t,sigma_ei(:,2),sigma_ei(:,3),'lineProps','k')
            shadedErrorBar(t,sigma_ie(:,2),sigma_ie(:,3),'lineProps','b')
            shadedErrorBar(t,sigma_ii(:,2),sigma_ii(:,3),'lineProps','c')

            plot(t,sigma_ee(:,1),'r','LineWidth',2.0)
            plot(t,sigma_ei(:,1),'k','LineWidth',2.0)
            plot(t,sigma_ie(:,1),'b','LineWidth',2.0)
            plot(t,sigma_ii(:,1),'c','LineWidth',2.0)

        %     plot(t,(ii_same(:,2)-ii_diff(:,2))./(ii_same(:,2)+ii_diff(:,2)),'c--','LineWidth',2.0)

            xlabel('traning epoch'), ylabel('\Sigma'), set(gca,'FontSize',16,'FontName','Arial')

            dtstr = fix(clock);
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/evo_sigmasel',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/evo_sigmasel',date_str,'.eps'],'-depsc','-painters')
            end
            %%
            fig1 = figure; hold on

            % plot(t,w_ex,'LineWidth',2.0,'Color',[0.6 0.6 0.6])
            % plot(t,w_ih,'LineWidth',2.0,'Color',[0.3 0.3 0.3])
        %     clear sigma_ee sigma_ei sigma_ie sigma_ii
        %     sigma_ee(:,1) = (ee_same(:,1)-ee_diff(:,1))./(ee_same(:,1)+ee_diff(:,1));
        %     sigma_ee(:,2) = (ee_same(:,2)-ee_diff(:,2))./(ee_same(:,2)+ee_diff(:,2));
        %     
        %     sigma_ei(:,1) = (ei_same(:,1)-ei_diff(:,1))./(ei_same(:,1)+ei_diff(:,1));
        %     sigma_ei(:,2) = (ei_same(:,2)-ei_diff(:,2))./(ei_same(:,2)+ei_diff(:,2));
        %     
        %     sigma_ie(:,1) = (ie_same(:,1)-ie_diff(:,1))./(ie_same(:,1)+ie_diff(:,1));
        %     sigma_ie(:,2) = (ie_same(:,2)-ie_diff(:,2))./(ie_same(:,2)+ie_diff(:,2));

        %     sigma_ii(:,1) = (ii_same(:,1)-ii_diff(:,1))./(ii_same(:,1)+ii_diff(:,1));
        %     sigma_ii(:,2) = (ii_same(:,2)-ii_diff(:,2))./(ii_same(:,2)+ii_diff(:,2));
        %     plot(t,(ii_same(:,1)-ii_diff(:,1))./(ii_same(:,1)+ii_diff(:,1)),'c','LineWidth',2.0)

        %     plot(t,sigma_ee(:,2) + sigma_ee(:,3),'r--','LineWidth',2.0)
        % 
        %     plot(t,sigma_ei(:,2) + sigma_ei(:,3),'k--','LineWidth',2.0)
        % 
        %     plot(t,sigma_ie(:,2) + sigma_ie(:,3),'b--','LineWidth',2.0)
        % 
        %     plot(t,sigma_ee(:,2) - sigma_ee(:,3),'r--','LineWidth',2.0)
        % 
        %     plot(t,sigma_ei(:,2) - sigma_ei(:,3),'k--','LineWidth',2.0)
        % 
        %     plot(t,sigma_ie(:,2) - sigma_ie(:,3),'b--','LineWidth',2.0)    

            shadedErrorBar(t,sigma_ee(:,5),sigma_ee(:,6),'lineProps','r') 
            shadedErrorBar(t,sigma_ei(:,5),sigma_ei(:,6),'lineProps','k')
            shadedErrorBar(t,sigma_ie(:,5),sigma_ie(:,6),'lineProps','b')
            shadedErrorBar(t,sigma_ii(:,5),sigma_ii(:,6),'lineProps','c')

            plot(t,sigma_ee(:,4),'r','LineWidth',2.0)
            plot(t,sigma_ei(:,4),'k','LineWidth',2.0)
            plot(t,sigma_ie(:,4),'b','LineWidth',2.0)
            plot(t,sigma_ii(:,4),'c','LineWidth',2.0)

        %     plot(t,(ii_same(:,2)-ii_diff(:,2))./(ii_same(:,2)+ii_diff(:,2)),'c--','LineWidth',2.0)

            xlabel('traning epoch'), ylabel('w\Sigma'), set(gca,'FontSize',16,'FontName','Arial')

            dtstr = fix(clock);
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/evo_wsigmasel',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/evo_wsigmasel',date_str,'.eps'],'-depsc','-painters')
            end
            %% Cellwise sigmas (from outputs)
            EE_sigma = zeros(length(auc_e(:,2)),2);
            EI_sigma = zeros(length(auc_e(:,2)),2);
            IE_sigma = zeros(length(auc_i(:,2)),2);
            II_sigma = zeros(length(auc_i(:,2)),2);

            wwee = wee_hist(:,:,1);
            wwei = wei_hist(:,:,1);
            wwie = wie_hist(:,:,1);
            wwii = wie_hist(:,:,1);

            for ii = 1:length(auc_e(:,2))
                if auc_e(ii,2) ~= 0
                    EE_sigma(ii,1) = (mean(wwee(sel_sim_ee(:,ii)==1,ii))-mean(wwee(sel_sim_ee(:,ii)==-1,ii)))/...
                        (mean(wwee(sel_sim_ee(:,ii)==1,ii))+mean(wwee(sel_sim_ee(:,ii)==-1,ii)));
                    EI_sigma(ii,1) = (mean(wwei(sel_sim_ei(:,ii)==1,ii))-mean(wwei(sel_sim_ei(:,ii)==-1,ii)))/...
                        (mean(wwei(sel_sim_ei(:,ii)==1,ii))+mean(wwei(sel_sim_ei(:,ii)==-1,ii)));
                end
            end

            for ii = 1:length(auc_i(:,2))
                if auc_i(ii,2) ~= 0
                    IE_sigma(ii,1) = (mean(wwie(sel_sim_ie(:,ii)==1,ii))-mean(wwie(sel_sim_ie(:,ii)==-1,ii)))/...
                        (mean(wwie(sel_sim_ie(:,ii)==1,ii))+mean(wwie(sel_sim_ie(:,ii)==-1,ii)));
                    II_sigma(ii,1) = (mean(wwii(sel_sim_ii(:,ii)==1,ii))-mean(wwii(sel_sim_ii(:,ii)==-1,ii)))/...
                        (mean(wwii(sel_sim_ii(:,ii)==1,ii))+mean(wwii(sel_sim_ii(:,ii)==-1,ii)));
                end
            end

            wwee = wee_hist(:,:,end);
            wwei = wei_hist(:,:,end);
            wwie = wie_hist(:,:,end);
            wwii = wie_hist(:,:,end);

            for ii = 1:length(auc_e(:,2))
                if auc_e(ii,2) ~= 0
                    EE_sigma(ii,2) = (mean(wwee(sel_sim_ee(:,ii)==1,ii))-mean(wwee(sel_sim_ee(:,ii)==-1,ii)))/...
                        (mean(wwee(sel_sim_ee(:,ii)==1,ii))+mean(wwee(sel_sim_ee(:,ii)==-1,ii)));
                    EI_sigma(ii,2) = (mean(wwei(sel_sim_ei(:,ii)==1,ii))-mean(wwei(sel_sim_ei(:,ii)==-1,ii)))/...
                        (mean(wwei(sel_sim_ei(:,ii)==1,ii))+mean(wwei(sel_sim_ei(:,ii)==-1,ii)));
                end
            end

            for ii = 1:length(auc_i(:,2))
                if auc_i(ii,2) ~= 0
                    IE_sigma(ii,2) = (mean(wwie(sel_sim_ie(:,ii)==1,ii))-mean(wwie(sel_sim_ie(:,ii)==-1,ii)))/...
                        (mean(wwie(sel_sim_ie(:,ii)==1,ii))+mean(wwie(sel_sim_ie(:,ii)==-1,ii)));
                    II_sigma(ii,2) = (mean(wwii(sel_sim_ii(:,ii)==1,ii))-mean(wwii(sel_sim_ii(:,ii)==-1,ii)))/...
                        (mean(wwii(sel_sim_ii(:,ii)==1,ii))+mean(wwii(sel_sim_ii(:,ii)==-1,ii)));
                end
            end
            %%
            fig1 = figure; hold on
            plot(abs(auc_e(auc_e(:,2)~=0,1)-0.5),squeeze(mean(mean(val_act_e(:,:,auc_e(:,2)~=0),2),1)),'r.','MarkerSize',16)
            plot(abs(auc_e(auc_e(:,2)==0,1)-0.5),squeeze(mean(mean(val_act_e(:,:,auc_e(:,2)==0),2),1)),'k.','MarkerSize',16)
            set(gca,'FontName','Hevetica','FontSize',12)
            xlabel('|AUC-0.5|'), ylabel('Mean Activity')
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/aucXact',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/aucXact',date_str,'.eps'],'-depsc','-painters')
            end
            %%    
            fig1=figure; hold on
            for pp = 1:length(auc_e(:,2))
                if auc_e(pp,2) ~=0
                    plot([1,1.5],abs([EE_sigma(pp,1),EE_sigma(pp,2)]),'.-r','MarkerSize',16,'LineWidth',2.0)
                    plot([2,2.5],abs([EI_sigma(pp,1),EI_sigma(pp,2)]),'.-k','MarkerSize',16,'LineWidth',2.0)
                end
            end
            for pp = 1:length(auc_i(:,2))
                if auc_i(pp,2) ~=0
                    plot([3,3.5],abs([IE_sigma(pp,1),IE_sigma(pp,2)]),'.-b','MarkerSize',16,'LineWidth',2.0)
                    plot([4,4.5],abs([II_sigma(pp,1),II_sigma(pp,2)]),'.-c','MarkerSize',16,'LineWidth',2.0)
                end
            end
            plot([1 4.25],[0 0],'--','Color',[0.5 0.5 0.5],'LineWidth',2)
            xlim([0.90 4.85])
            set(gca,'XTick',[1,2,3,4],'XTickLabel',{'EE','EI','IE','II'})
            ylabel('|\Sigma|'), set(gca,'FontSize',16,'FontName','Arial')
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/cellwise_sigmas',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/cellwise_ABSsigmas',date_str,'.eps'],'-depsc','-painters')
            end
            %%    
            fig1=figure; hold on
            cmap = parula(ne+5);
            [~,ll] = sort(abs(auc_e(:,1)),'descend');
            for pp = 1:length(auc_e(:,2))
                if auc_e(ll(pp),2) ~=0
                    plot([1,1.5],([EE_sigma(ll(pp),1),EE_sigma(ll(pp),2)]),'.-','MarkerSize',16,'LineWidth',2.0,'Color',cmap(pp,:))
                    plot([2,2.5],([EI_sigma(ll(pp),1),EI_sigma(ll(pp),2)]),'.-','MarkerSize',16,'LineWidth',2.0,'Color',cmap(pp,:))
                end
            end
            cmap = parula(ni+2);
            [~,ll] = sort(abs(auc_i(:,1)),'descend');
            for pp = 1:length(auc_i(:,2))
                if auc_i(ll(pp),2) ~=0
                    plot([3,3.5],([IE_sigma(ll(pp),1),IE_sigma(ll(pp),2)]),'.-','MarkerSize',16,'LineWidth',2.0,'Color',cmap(pp,:))
                    plot([4,4.5],([II_sigma(ll(pp),1),II_sigma(ll(pp),2)]),'.-','MarkerSize',16,'LineWidth',2.0,'Color',cmap(pp,:))
                end
            end
            plot([1 4.25],[0 0],'--','Color',[0.5 0.5 0.5],'LineWidth',2)
            xlim([0.90 4.85])
            set(gca,'XTick',[1,2,3,4],'XTickLabel',{'EE','EI','IE','II'})
            ylabel('\Sigma'), set(gca,'FontSize',16,'FontName','Arial')
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/cellwise_sigmas',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/cellwise_sigmas',date_str,'.eps'],'-depsc','-painters')
            end
            %%
            fig1 = figure;
            plot(sigma_ei(:,1).*sigma_ie(:,1),perf_hist,'k.','MarkerSize',16)
            xlabel('\Sigma^{EI}\Sigma^{IE}'), ylabel('Performance'), set(gca,'FontSize',16,'FontName','Arial')
            dtstr = fix(clock);
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/sigmaselXperf',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/sigmaselXperf',date_str,'.eps'],'-depsc','-painters')
            end

            fig1 = figure;
            plot(sigma_ei(:,1).*sigma_ie(:,1),sum(chrono_hist,1),'k.','MarkerSize',16)
            xlabel('\Sigma^{EI}\Sigma^{IE}'), ylabel('RT'), set(gca,'FontSize',16,'FontName','Arial')
            dtstr = fix(clock);
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/sigmaselXRT',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/sigmaselXRT',date_str,'.eps'],'-depsc','-painters')
            end
            %%
            wee_cor  = zeros(size(wee_hist,3),1);
            wei_cor  = zeros(size(wee_hist,3),1);
            wie_cor  = zeros(size(wee_hist,3),1);
            wii_cor  = zeros(size(wee_hist,3),1);
            wine_cor = zeros(size(wee_hist,3),1);
            wo_cor   = zeros(size(wee_hist,3),1);

            xee  = reshape(wee_hist(:,:,1),[],1);
            xei  = reshape(wei_hist(:,:,1),[],1);
            xie  = reshape(wie_hist(:,:,1),[],1);
            xii  = reshape(wii_hist(:,:,1),[],1);
            xine = reshape(wine_hist(:,:,1),[],1);
            xo   = reshape(wo_hist(:,:,1),[],1);

            for ii=1:size(wee_hist,3)
               yee  = reshape(wee_hist(:,:,ii),[],1);
               yei  = reshape(wei_hist(:,:,ii),[],1);
               yie  = reshape(wie_hist(:,:,ii),[],1);
               yii  = reshape(wii_hist(:,:,ii),[],1);
               yine = reshape(wine_hist(:,:,ii),[],1);
               yo   = reshape(wo_hist(:,:,ii),[],1);
               wee_cor(ii)  = corr(xee,yee);
               wei_cor(ii)  = corr(xei,yei);
               wie_cor(ii)  = corr(xie,yie);
               wii_cor(ii)  = corr(xii,yii);
               wine_cor(ii) = corr(xine,yine);
               wo_cor(ii)   = corr(xo,yo);
            end
            fig1 = figure; hold on
            plot(wee_cor,'r','LineWidth',2.0)
            plot(wei_cor,'k','LineWidth',2.0)
            plot(wie_cor,'b','LineWidth',2.0)
            plot(wii_cor,'c','LineWidth',2.0)
            plot(wine_cor,'-','Color',[0.6 0.6 0.6],'LineWidth',2.0)
            plot(wo_cor,'--','Color',[0.6 0.6 0.6],'LineWidth',2.0)

            xlabel('traning epoch'), ylabel('Weight Correlation'), set(gca,'FontSize',16,'FontName','Arial')
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/weight_corr',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/weight_corr',date_str,'.eps'],'-depsc','-painters')
            end
            %%
            fig1 = figure;
            if exist('loss_fcn','var')
                plot(log10(loss_fcn),'LineWidth',2.0)
            else
                plot(log10(loss),'LineWidth',2.0)
            end
            xlabel('traning epoch'), ylabel('log_{10}(loss)'), set(gca,'FontSize',16,'FontName','Arial')
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/loss',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/loss',date_str,'.eps'],'-depsc','-painters')
            end
            %%
            fig1 = figure;
            subplot(3,1,1),imagesc(2*(1:length(perf_hist)),psycho_b,psycho_hist), axis xy, colorbar
            subplot(3,1,2),imagesc(2*(1:length(perf_hist)),psycho_b,chrono_hist), axis xy, colorbar
            subplot(3,1,3),imagesc(2*(1:length(perf_hist)),psycho_b,ntrial_hist), axis xy, colorbar
            xlabel('traning epoch'), ylabel('stim'), set(gca,'FontSize',16,'FontName','Arial')
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/psychometrics',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/psychometrics',date_str,'.eps'],'-depsc','-painters')
            end
            %%
            fig1 = figure;
            plot(2*(1:length(perf_hist)),perf_hist,'k','LineWidth',2.0), hold on
            plot(2*(1:length(perf_hist)),perf0_hist,'r','LineWidth',2.0)
            plot(2*(1:length(perf_hist)),perf1_hist,'b','LineWidth',2.0)
            xlabel('traning epoch'), ylabel('performance'), set(gca,'FontSize',16,'FontName','Arial')
            if exist('/Users/roachjp/work/','dir')
                print(fig1,[f{oo},'/performance',date_str,'.eps'],'-depsc','-painters')
            elseif exist('/Users/roach/work/','dir')
                print(fig1,[f{oo},'/performance',date_str,'.eps'],'-depsc','-painters')
            end
            %%
            fig1 = figure; hold on
            cmap = redblue(length(psycho_b));

            pre = ceil(min(trial_starts_val)/2);
            pst = trial_end_val(1)-trial_starts_val(1) + ceil((8/8)*(size(val_pred,2)-max(trial_end_val)));
            t = -pre:pst;
            del_o = zeros(length(psycho_b),length(t));
            stim_avesE = zeros(length(psycho_b),length(t),size(val_act_e,3));
            stim_avesI = zeros(length(psycho_b),length(t),size(val_act_i,3));

            residuals_E = zeros(size(val_act_e,1),length(t),size(val_act_e,3));
            residuals_I = zeros(size(val_act_i,1),length(t),size(val_act_i,3));

            plot([t(1) t(end)],[0.25 0.25],'--','LineWidth',2.0,'Color',[0.6 0.6 0.6])
            plot([t(1) t(end)],[-0.25 -0.25],'--','LineWidth',2.0,'Color',[0.6 0.6 0.6])
            for ii = 1:length(psycho_b)
                x = find(val_coh == psycho_b(ii));

                i1 = trial_starts_val(1) - pre;
                i2 = trial_starts_val(1) + pst;

                del_o(ii,:) = squeeze(mean(val_pred(x,i1:i2,1)- val_pred(x,i1:i2,2),1));
                plot(t,del_o(ii,:),'Color',cmap(ii,:),'LineWidth',2.0)
                for jj=1:size(val_act_e,3)
                    stim_avesE(ii,:,jj) = squeeze(mean(val_act_e(x,i1:i2,jj)));
                    residuals_E(x,:,jj) = val_act_e(x,i1:i2,jj) -  stim_avesE(ii,:,jj);
                end
                for jj=1:size(val_act_i,3)
                    stim_avesI(ii,:,jj) = squeeze(mean(val_act_i(x,i1:i2,jj)));
                    residuals_I(x,:,jj) = val_act_i(x,i1:i2,jj) -  stim_avesI(ii,:,jj);
                end

            end

            xlabel('Time rel stim.'), ylabel('Z_{1} - Z_{2}'), set(gca,'FontSize',16,'FontName','Arial','Color',[0 0 0])
            colormap(cmap)
            cb1 = colorbar();
            ylabel(cb1,'Signed Stim'), set(cb1,'FontSize',16,'FontName','Arial')
            set(cb1,'YTick',[0 0.5 1],'YTickLabel',{'-20','0','20'})
            exportgraphics(fig1,[f{oo},'/ave_dz',date_str,'.pdf'],'ContentType','vector',...
                       'BackgroundColor',[0 0 0])
            %%
            noise_corrEE = nan(size(val_act_e,3),size(val_act_i,3),length(t));
            noise_corrEI = nan(size(val_act_e,3),size(val_act_i,3),length(t));
            noise_corrII = nan(size(val_act_i,3),size(val_act_i,3),length(t));

            for ii=1:length(t)
                for jj = 1:size(val_act_e,3)
                    for kk = (jj+1):size(val_act_e,3)
                        noise_corrEE(jj,kk,ii) = corr(residuals_E(:,ii,jj),residuals_E(:,ii,kk));
                    end
                    for kk = 1:size(val_act_i,3)
                        noise_corrEI(jj,kk,ii) = corr(residuals_E(:,ii,jj),residuals_I(:,ii,kk));
                    end
                end
                for jj = 1:size(val_act_i,3)
                    for kk = (jj+1):size(val_act_i,3)
                        noise_corrII(jj,kk,ii) = corr(residuals_I(:,ii,jj),residuals_I(:,ii,kk));
                    end
                end  
            end
            %%
            ee_corr_t = zeros(length(t),3,2);
            ei_corr_t = zeros(length(t),3,2);
            ii_corr_t = zeros(length(t),3,2);

            for ii=1:length(t)
                dd = noise_corrEE(:,:,ii);
                ee_corr_t(ii,1,1) = nanmean(dd(sel_sim_ee==1));
                ee_corr_t(ii,1,2) = nanstd(dd(sel_sim_ee==1))/sqrt(sum(~isnan(dd(sel_sim_ee==1))));
                dd = noise_corrEI(:,:,ii);
                ei_corr_t(ii,1,1) = nanmean(dd(sel_sim_ei==1));
                ei_corr_t(ii,1,2) = nanstd(dd(sel_sim_ei==1))/sqrt(sum(~isnan(dd(sel_sim_ei==1))));
                dd = noise_corrII(:,:,ii);
                ii_corr_t(ii,1,1) = nanmean(dd(sel_sim_ii==1));
                ii_corr_t(ii,1,2) = nanstd(dd(sel_sim_ii==1))/sqrt(sum(~isnan(dd(sel_sim_ii==1))));

                dd = noise_corrEE(:,:,ii);
                ee_corr_t(ii,2,1) = nanmean(dd(sel_sim_ee==-1));
                ee_corr_t(ii,2,2) = nanstd(dd(sel_sim_ee==-1))/sqrt(sum(~isnan(dd(sel_sim_ee==1))));
                dd = noise_corrEI(:,:,ii);
                ei_corr_t(ii,2,1) = nanmean(dd(sel_sim_ei==-1));
                ei_corr_t(ii,2,2) = nanstd(dd(sel_sim_ei==-1))/sqrt(sum(~isnan(dd(sel_sim_ei==1))));
                dd = noise_corrII(:,:,ii);
                ii_corr_t(ii,2,1) = nanmean(dd(sel_sim_ii==-1));
                ii_corr_t(ii,2,2) = nanstd(dd(sel_sim_ii==-1))/sqrt(sum(~isnan(dd(sel_sim_ii==1))));

                dd = noise_corrEE(:,:,ii);
                ee_corr_t(ii,3,1) = nanmean(dd(sel_sim_ee==0));
                ee_corr_t(ii,3,2) = nanmean(dd(sel_sim_ee==0))/sqrt(sum(~isnan(dd(sel_sim_ee==1))));
                dd = noise_corrEI(:,:,ii);
                ei_corr_t(ii,3,1) = nanmean(dd(sel_sim_ei==0));
                ei_corr_t(ii,3,2) = nanmean(dd(sel_sim_ei==0))/sqrt(sum(~isnan(dd(sel_sim_ei==1))));
                dd = noise_corrII(:,:,ii);
                ii_corr_t(ii,3,1) = nanmean(dd(sel_sim_ii==0));
                ii_corr_t(ii,3,2) = nanmean(dd(sel_sim_ii==0))/sqrt(sum(~isnan(dd(sel_sim_ii==1))));
            end
            yls = [0,0];
            fig1 = figure;
            subplot(1,3,1), shadedErrorBar(t,ee_corr_t(:,1,1),ee_corr_t(:,1,2),'lineprops',{'g','LineWidth',2.0})
            shadedErrorBar(t,ee_corr_t(:,2,1),ee_corr_t(:,2,2),'lineprops',{'m','LineWidth',2.0})
            shadedErrorBar(t,ee_corr_t(:,3,1),ee_corr_t(:,3,2),'lineprops',{'k','LineWidth',2.0})
            yls(1,:) = get(gca,'Ylim');
            subplot(1,3,2),shadedErrorBar(t,ei_corr_t(:,1,1),ei_corr_t(:,1,2),'lineprops',{'g','LineWidth',2.0})
            shadedErrorBar(t,ei_corr_t(:,2,1),ei_corr_t(:,2,2),'lineprops',{'m','LineWidth',2.0})
            shadedErrorBar(t,ei_corr_t(:,3,1),ei_corr_t(:,3,2),'lineprops',{'k','LineWidth',2.0})
            yls(2,:) = get(gca,'Ylim');

            subplot(1,3,3),shadedErrorBar(t,ii_corr_t(:,1,1),ii_corr_t(:,1,2),'lineprops',{'g','LineWidth',2.0})
            shadedErrorBar(t,ii_corr_t(:,2,1),ii_corr_t(:,2,2),'lineprops',{'m','LineWidth',2.0})
            shadedErrorBar(t,ii_corr_t(:,3,1),ii_corr_t(:,3,2),'lineprops',{'k','LineWidth',2.0})
            yls(3,:) = get(gca,'Ylim');

            subplot(1,3,1),set(gca,'YLim',[min(yls(:,1)) max(yls(:,2))],'FontName','Hevetica','FontSize',12)
            title('E-E pairs'), ylabel('Correlation'),xlabel('time to stim')
            subplot(1,3,2),set(gca,'YLim',[min(yls(:,1)) max(yls(:,2))],'FontName','Hevetica','FontSize',12)
            title('E-I pairs'),xlabel('time to stim')
            subplot(1,3,3),set(gca,'YLim',[min(yls(:,1)) max(yls(:,2))],'FontName','Hevetica','FontSize',12)
            title('I-I pairs'),xlabel('time to stim')
            exportgraphics(fig1,[f{oo},'/corrs',date_str,'.pdf'],'ContentType','vector')
            %%
            delta_sel(oo,1,1) = (ee_same(end,1)-ee_diff(end,1))./(ee_same(end,1)+ee_diff(end,1)) - (ee_same(1,1)-ee_diff(1,1))./(ee_same(1,1)+ee_diff(1,1));
            delta_sel(oo,2,1) = (ei_same(end,1)-ei_diff(end,1))./(ei_same(end,1)+ei_diff(end,1)) - (ei_same(1,1)-ei_diff(1,1))./(ei_same(1,1)+ei_diff(1,1));
            delta_sel(oo,3,1) = (ie_same(end,1)-ie_diff(end,1))./(ie_same(end,1)+ie_diff(end,1)) - (ie_same(1,1)-ie_diff(1,1))./(ie_same(1,1)+ie_diff(1,1));
            delta_sel(oo,4,1) = (ii_same(end,1)-ii_diff(end,1))./(ii_same(end,1)+ii_diff(end,1)) - (ii_same(1,1)-ii_diff(1,1))./(ii_same(1,1)+ii_diff(1,1));

            delta_sel(oo,1,2) = (ee_same(end,2)-ee_diff(end,2))./(ee_same(end,2)+ee_diff(end,2)) - (ee_same(1,2)-ee_diff(1,2))./(ee_same(1,2)+ee_diff(1,2));
            delta_sel(oo,2,2) = (ei_same(end,2)-ei_diff(end,2))./(ei_same(end,2)+ei_diff(end,2)) - (ei_same(1,2)-ei_diff(1,2))./(ei_same(1,2)+ei_diff(1,2));
            delta_sel(oo,3,2) = (ie_same(end,2)-ie_diff(end,2))./(ie_same(end,2)+ie_diff(end,2)) - (ie_same(1,2)-ie_diff(1,2))./(ie_same(1,2)+ie_diff(1,2));
            delta_sel(oo,4,2) = (ii_same(end,2)-ii_diff(end,2))./(ii_same(end,2)+ii_diff(end,2)) - (ii_same(1,2)-ii_diff(1,2))./(ii_same(1,2)+ii_diff(1,2));
            %%
            save([f{oo},'/evo_sigmasel',date_str,'.mat'],'sigma_ee','sigma_ei','sigma_ie',...
                'sigma_ii','p_ch1','auc_e','auc_i','wee_cor','wei_cor','wie_cor','wii_cor','wine_cor','wo_cor')
            save([f{oo},'/correlations',date_str,'.mat'],'ee_corr_t','ei_corr_t','ii_corr_t',...
                'noise_corrEE','noise_corrEI','noise_corrII','stim_avesE','stim_avesI','residuals_E','residuals_I','t'...
                ,'sel_sim_ee','sel_sim_ei','sel_sim_ie','sel_sim_ii')
            close all
        end
    end
end
sss = find(f{1} == '/');
flum = f{1}(1:sss(end));
save([flum,'delt_sel',date_str,'.mat'],'delta_sel')