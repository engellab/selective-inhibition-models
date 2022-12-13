clear
f = uipickfiles('FilterSpec','/Users/roach/work/RNNs/');

date_str = int2str(fix(clock));
date_str(date_str==' ') = '_';

sigmaees = cell(length(f),1);
sigmaeis = cell(length(f),1);
sigmaies = cell(length(f),1);
sigmaiis = cell(length(f),1);

auces = cell(length(f),1);
aucis = cell(length(f),1);

wee_corrs = cell(length(f),1);
wei_corrs = cell(length(f),1);
wie_corrs = cell(length(f),1);
wii_corrs = cell(length(f),1);

wine_corrs = cell(length(f),1);
wo_corrs = cell(length(f),1);

p_ch1s = cell(length(f),1);
chronos = cell(length(f),1);
perfs = zeros(length(f),1);
good_run = zeros(length(f),1);
ntrials = cell(length(f),1);

perfT_hists = cell(length(f),1);

for ii=1:length(f)
    try
        load([f{ii},'/evo_sigmasel.mat'])
        load([f{ii},'/data.mat'])
        sigmaees{ii} = sigma_ee;
        sigmaeis{ii} = sigma_ei;
        sigmaies{ii} = sigma_ie;
        sigmaiis{ii} = sigma_ii;

        auces{ii}  = auc_e;
        aucis{ii}  = auc_i;
        p_ch1s{ii} = [double(psycho_b)',psycho'];
    %     p_ch1s{ii} = [double(psycho_b)',p_ch1];
        chronos{ii} = [double(psycho_b)',chrono'];
        ntrials{ii} = [double(psycho_b)',ntrial'];
        perfs(ii) = perf;
        if perfT_hist(end) > 0.85
            good_run(ii) = 1;
        end

        wee_corrs{ii} = wee_cor;
        wei_corrs{ii} = wei_cor;
        wie_corrs{ii} = wie_cor;
        wii_corrs{ii} = wii_cor;

        wine_corrs{ii} = wine_cor;
        wo_corrs{ii} = wo_cor;

        perfT_hists{ii} = perfT_hist;
    end
    
end

%%
fig1 = figure; hold on
pch_arry = zeros(length(p_ch1s{end}(:,1)),length(p_ch1s));
uu = find(good_run==1);

for ii=1:length(p_ch1s)
    if good_run(ii) == 1
        plot(p_ch1s{ii}(:,1),p_ch1s{ii}(:,2),'-','LineWidth',1,'Color',[0.75 0.75 0.75],'MarkerSize',16)
        pch_arry(:,ii) = p_ch1s{ii}(:,2);
    end
end
shadedErrorBar(p_ch1s{uu(1)}(:,1),nanmean(pch_arry(:,good_run==1),2),nanstd(pch_arry(:,good_run==1),1,2)/sqrt(sum(good_run==1)),'lineprops','r')
ylim([-0.05 1.05])

set(gca,'FontName','Arial','FontSize',12)
print(fig1,['/Users/roach/work/perf_fig',date_str],'-dpdf','-painters')

%%
fig1 = figure; hold on
chronos_arry = zeros(length(chronos{end}(:,1)),length(chronos));

for ii=1:length(p_ch1s)
    if good_run(ii) == 1
        plot(chronos{ii}(:,1),chronos{ii}(:,2),'-','LineWidth',1,'Color',[0.75 0.75 0.75],'MarkerSize',16)
        chronos_arry(:,ii) = chronos{ii}(:,2);
    end
end
shadedErrorBar(chronos{uu(1)}(:,1),nanmean(chronos_arry(:,good_run==1),2),nanstd(chronos_arry(:,good_run==1),1,2)/sqrt(sum(good_run==1)),'lineprops','r')
%ylim([-0.05 1.05])

set(gca,'FontName','Arial','FontSize',12)
print(fig1,['/Users/roach/work/chron_fig',date_str],'-dpdf','-painters')
%%
fig1 = figure; hold on
ntrials_arry = zeros(length(ntrials{end}(:,1)),length(ntrials));

for ii=1:length(p_ch1s)
    if good_run(ii) == 1
        plot(ntrials{ii}(:,1),ntrials{ii}(:,2),'-','LineWidth',1,'Color',[0.75 0.75 0.75],'MarkerSize',16)
        ntrials_arry(:,ii) = ntrials{ii}(:,2);
    end
end
shadedErrorBar(ntrials{uu(1)}(:,1),nanmean(ntrials_arry(:,good_run==1),2),nanstd(ntrials_arry(:,good_run==1),1,2)/sqrt(sum(good_run==1)),'lineprops','r')
%ylim([-0.05 1.05])

set(gca,'FontName','Arial','FontSize',12)
print(fig1,['/Users/roach/work/ntrial_fig',date_str],'-dpdf','-painters')
%%
fig1=figure('Units','Inches','Position',[1,1,4,3],'PaperPosition',[1,1,4,3]);
ax1 = axes('Position',[0.15 0.6 0.75 0.35]); hold on
pch_arry = zeros(length(p_ch1s{end}(:,1)),length(p_ch1s));
uu = find(good_run==1);

for ii=1:length(p_ch1s)
    if good_run(ii) == 1
        plot(p_ch1s{ii}(:,1),p_ch1s{ii}(:,2),'-','LineWidth',1,'Color',[0.75 0.75 0.75],'MarkerSize',16)
        pch_arry(:,ii) = p_ch1s{ii}(:,2);
    end
end
shadedErrorBar(p_ch1s{uu(1)}(:,1),nanmean(pch_arry(:,good_run==1),2),nanstd(pch_arry(:,good_run==1),1,2)/sqrt(sum(good_run==1)),'lineprops','r')
ylim([-0.05 1.05]), set(ax1,'YTick',[0.2 0.6 1.0]),set(ax1,'XLim',[-22 22])
xlabel(''),ylabel('P(Choose 1)')
set(gca,'FontName','Arial','FontSize',11)
ax2 = axes('Position',[0.15 0.15 0.75 0.35]); hold on
ntrials_arry = zeros(length(ntrials{end}(:,1)),length(ntrials));

for ii=1:length(p_ch1s)
    if good_run(ii) == 1
        plot(ntrials{ii}(:,1),ntrials{ii}(:,2),'-','LineWidth',1,'Color',[0.75 0.75 0.75],'MarkerSize',16)
        ntrials_arry(:,ii) = ntrials{ii}(:,2);
    end
end
shadedErrorBar(ntrials{uu(1)}(:,1),nanmean(ntrials_arry(:,good_run==1),2),nanstd(ntrials_arry(:,good_run==1),1,2)/sqrt(sum(good_run==1)),'lineprops','r')
xlabel('Stimulus'),ylabel('% Trial Completed'), set(ax2,'YTick',[60 80 100]),set(ax2,'XLim',[-22 22])
set(gca,'FontName','Arial','FontSize',11)
print(fig1,['/Users/roach/work/ntrial_psycho_fig',date_str],'-dpdf','-painters')
%ylim([-0.05 1.05])
%%

%perf,rt,see,seisie
% rr = zeros(4);
% pp = zeros(4);
% 
% [rr(1,4),pp(1,4)] = corr(seisie_vect(good_run ==1)',perfs(good_run==1));
% [rr(2,4),pp(2,4)] = corr(seisie_vect(good_run ==1)',sum(chronos_arry(:,good_run==1))');
% [rr(2,3),pp(2,3)] = corr(see_vect(good_run ==1)',sum(chronos_arry(:,good_run==1))');
% [rr(1,3),pp(1,3)] = corr(see_vect(good_run ==1)',perfs(good_run==1));
% [rr(1,2),pp(1,2)] = corr(perfs(good_run==1),sum(chronos_arry(:,good_run==1))');
%%
fig1 = figure; hold on
plot(perfs(good_run==1),sum(chronos_arry(:,good_run==1)),'.k','MarkerSize',12)
xlabel('performance'),ylabel('reation time')
set(gca,'FontName','Arial','FontSize',12)
print(fig1,['/Users/roach/work/sat_fig',date_str],'-dpdf','-painters')
%%
fsele = zeros(length(auces),1);
fseli = zeros(length(auces),1);
for ii = 1:length(f)
    if good_run(ii) == 1
        fsele(ii) = sum(auces{ii}(:,2) ~= 0)/length(auces{ii}(:,2));
        fseli(ii) = sum(aucis{ii}(:,2) ~= 0)/length(aucis{ii}(:,2));
    end
end
[ne,ee] = histcounts(fsele(good_run==1),0:0.05:1);
[ni,ei] = histcounts(fseli(good_run==1),0:0.05:1);

fig1 = figure('Units','Inches','Position',[1,1,4,3],'PaperPosition',[1,1,4,3]); hold on
plot(ee(2:end),ne/sum(ne),'k','LineWidth',2.0)
plot(ei(2:end),ni/sum(ni),'k--','LineWidth',2.0)
ylabel('Fraction networks'), xlabel('Fraction Selective')

el = find(ne~=0);
il = find(ni~=0);
xl = min(el(1),il(1));
xh = max(el(end),il(end));
if xl<2
    xl=2;
end
if xh >= length(ee)-1
    xh = length(ee)-2;
end
xlim([ee(xl-1),ee(xh+2)])
set(gca,'FontName','Arial','FontSize',11)
print(fig1,['/Users/roach/work/frac_selective_fig',date_str],'-dpdf','-painters')
%% Change in selectivity
figure; hold on
for ii=1:length(f)
    if good_run(ii) == 1
        plot(1,sigmaees{ii}(end,1)-sigmaees{ii}(1,1),'k.','MarkerSize',16)
        plot(2,sigmaeis{ii}(end,1)-sigmaeis{ii}(1,1),'k.','MarkerSize',16)
        plot(3,sigmaies{ii}(end,1)-sigmaies{ii}(1,1),'k.','MarkerSize',16)
        plot(4,sigmaiis{ii}(end,1)-sigmaiis{ii}(1,1),'k.','MarkerSize',16)
    end
end
%% Absolute selectivity end
fig1=figure; hold on

sels_ee = zeros(length(f),1);
sels_ie = zeros(length(f),1);
sels_ei = zeros(length(f),1);
sels_ii = zeros(length(f),1);

for ii=1:length(f)
    if good_run(ii) == 1
        plot(1,sigmaees{ii}(end,1),'k.','MarkerSize',16)
        plot(2,sigmaeis{ii}(end,1),'k.','MarkerSize',16)
        plot(3,sigmaies{ii}(end,1),'k.','MarkerSize',16)
        plot(4,sigmaiis{ii}(end,1),'k.','MarkerSize',16)
        sels_ee(ii) = sigmaees{ii}(end,1); 
        sels_ei(ii) = sigmaeis{ii}(end,1);
        sels_ie(ii) = sigmaies{ii}(end,1);
        sels_ii(ii) = sigmaiis{ii}(end,1);
    end
end
errorbar(1.25,mean(sels_ee(good_run==1)),std(sels_ee(good_run==1))/sqrt(sum(good_run)),'.k','LineWidth',2.0,'MarkerSize',16)
errorbar(2.25,mean(sels_ei(good_run==1)),std(sels_ei(good_run==1))/sqrt(sum(good_run)),'.k','LineWidth',2.0,'MarkerSize',16)
errorbar(3.25,mean(sels_ie(good_run==1)),std(sels_ie(good_run==1))/sqrt(sum(good_run)),'.k','LineWidth',2.0,'MarkerSize',16)
errorbar(4.25,mean(sels_ii(good_run==1)),std(sels_ii(good_run==1))/sqrt(sum(good_run)),'.k','LineWidth',2.0,'MarkerSize',16)
plot([0.5 4.5],[0 0],'--','Color',[0.5 0.5 0.5],'LineWidth',2.0)
set(gca,'FontName','Arial','FontSize',12,'XTick',[1,2,3,4],'XTickLabel',{'\Sigma^{EE}','\Sigma^{EI}','\Sigma^{IE}','\Sigma^{II}'})

print(fig1,['/Users/roach/work/end_connsel_fig',date_str],'-dpdf','-painters')

%% Absolute selectivity end
fig1=figure('Units','Inches','Position',[1,1,4,3],'PaperPosition',[1,1,4,3]); hold on
plot([0.5 4.5],[0 0],'--','Color',[0.5 0.5 0.5],'LineWidth',2.0)
sels_ee = zeros(length(f),1);
sels_ie = zeros(length(f),1);
sels_ei = zeros(length(f),1);
sels_ii = zeros(length(f),1);
for ii=1:length(f)
    if good_run(ii) == 1
        sels_ee(ii) = sigmaees{ii}(end,1); 
        sels_ei(ii) = sigmaeis{ii}(end,1);
        sels_ie(ii) = sigmaies{ii}(end,1);
        sels_ii(ii) = sigmaiis{ii}(end,1);
    end
end

[~,ll] = sort(sels_ee,'descend');
cmap = parula(length(ll)+5);
for ii=1:length(f)
    if good_run(ll(ii)) == 1
        plot([1,2,3,4],[sigmaees{ll(ii)}(end,1),sigmaeis{ll(ii)}(end,1),sigmaies{ll(ii)}(end,1),sigmaiis{ll(ii)}(end,1)],'.-','Color',cmap(ii,:),'MarkerSize',16,'LineWidth',2)
    %     plot(2,sigmaeis{ii}(end,1),'k.','MarkerSize',16)
    %     plot(3,sigmaies{ii}(end,1),'k.','MarkerSize',16)
    %     plot(4,sigmaiis{ii}(end,1),'k.','MarkerSize',16)
    %     sels_ee(ii) = sigmaees{ii}(end,1); 
    %     sels_ei(ii) = sigmaeis{ii}(end,1);
    %     sels_ie(ii) = sigmaies{ii}(end,1);
    %     sels_ii(ii) = sigmaiis{ii}(end,1);
    end
end
errorbar([1,2,3,4],[mean(sels_ee(good_run==1)),mean(sels_ei(good_run==1)),mean(sels_ie(good_run==1)),mean(sels_ii(good_run==1))]...
    ,[std(sels_ee(good_run==1))/sqrt(sum(good_run)),std(sels_ei(good_run==1))/sqrt(sum(good_run)),...
    std(sels_ie(good_run==1))/sqrt(sum(good_run)),std(sels_ii(good_run==1))/sqrt(sum(good_run))],'k','LineWidth',2.0)
yl = get(gca,'YLim');
ylim([yl(1)-0.025 yl(2)])
set(gca,'FontName','Arial','FontSize',11,'XTick',[1,2,3,4],'XTickLabel',{'\Sigma^{EE}','\Sigma^{EI}','\Sigma^{IE}','\Sigma^{II}'})
colormap(cmap(1:end-5,:))
cb1 = colorbar();
set(cb1,'YTick',[0 ceil(sum(good_run)/2)/sum(good_run) 1],'YTickLabel',{'1',int2str(ceil(sum(good_run)/2)),int2str(sum(good_run))}), 
set(cb1,'FontName','Arial','FontSize',11)
ylabel(cb1,'\Sigma^{EE} Rank','Rotation',-90,'Position',[3.5 0.5])
print(fig1,['/Users/roach/work/end_connsel_netcon_fig',date_str],'-dpdf','-painters')
%% Absolute selectivity end
fig1=figure; hold on
plot([0.5 4.5],[0 0],'--','Color',[0.5 0.5 0.5],'LineWidth',2.0)
wsels_ee = zeros(length(f),1);
wsels_ie = zeros(length(f),1);
wsels_ei = zeros(length(f),1);
wsels_ii = zeros(length(f),1);
for ii=1:length(f)
    if good_run(ii) == 1
        wsels_ee(ii) = sigmaees{ii}(end,4); 
        wsels_ei(ii) = sigmaeis{ii}(end,4);
        wsels_ie(ii) = sigmaies{ii}(end,4);
        wsels_ii(ii) = sigmaiis{ii}(end,4);
    end
end

[~,ll] = sort(sels_ee);
cmap = parula(length(ll)+5);
for ii=1:length(f)
    if good_run(ii) == 1
        plot([1,2,3,4],[sigmaees{ll(ii)}(end,4),sigmaeis{ll(ii)}(end,4),sigmaies{ll(ii)}(end,4),sigmaiis{ll(ii)}(end,4)],'.-','Color',cmap(ii,:),'MarkerSize',16,'LineWidth',2)
    %     plot(2,sigmaeis{ii}(end,1),'k.','MarkerSize',16)
    %     plot(3,sigmaies{ii}(end,1),'k.','MarkerSize',16)
    %     plot(4,sigmaiis{ii}(end,1),'k.','MarkerSize',16)
    %     sels_ee(ii) = sigmaees{ii}(end,1); 
    %     sels_ei(ii) = sigmaeis{ii}(end,1);
    %     sels_ie(ii) = sigmaies{ii}(end,1);
    %     sels_ii(ii) = sigmaiis{ii}(end,1);
    end
end

yl = get(gca,'YLim');
ylim([yl(1)-0.025 yl(2)])
set(gca,'FontName','Arial','FontSize',12,'XTick',[1,2,3,4],'XTickLabel',{'\Sigma^{EE}','\Sigma^{EI}','\Sigma^{IE}','\Sigma^{II}'})
colormap(cmap(1:end-5,:))
cb1 = colorbar();
set(cb1,'YTick',[0 25/51 1],'YTickLabel',{'1','25','51'}), ylabel(cb1,'\Sigma^{EE} Rank','Rotation',-90)
print(fig1,['/Users/roach/work/end_connwsel_netcon_fig',date_str],'-dpdf','-painters')
%%
cat = [1.25,2.25,3.25,4.25];
sig = [mean(sels_ee(good_run==1)),mean(sels_ei(good_run==1)),mean(sels_ie(good_run==1)),mean(sels_ii(good_run==1))];
ssig = [std(sels_ee(good_run==1))/sqrt(sum(good_run)),std(sels_ei(good_run==1))/sqrt(sum(good_run)),std(sels_ie(good_run==1))/sqrt(sum(good_run)),std(sels_ii(good_run==1))/sqrt(sum(good_run))];
save(['/Users/roach/work/',date_str,'.mat'],'cat','sig','ssig')
%%
fig1=figure; hold on
for ii=1:length(f)
    if good_run(ii) ==1
        plot(sigmaees{ii}(end,1),sigmaeis{ii}(end,1)*sigmaies{ii}(end,1),'.','MarkerSize',16)
        see_vect(ii)    = sigmaees{ii}(end,1);
        seisie_vect(ii) = sigmaeis{ii}(end,1)*sigmaies{ii}(end,1);
    end
end
 [r,p] = corr(see_vect(good_run ==1)',seisie_vect(good_run ==1)')
set(gca,'FontName','Arial','FontSize',12)
xlabel('\Sigma^{EE}'),ylabel('\Sigma^{EI}*\Sigma^{IE}')
print(fig1,['/Users/roach/work/sigmaEE_contra_ipsi',date_str],'-dpdf','-painters')
%%
fig1=figure; hold on
for ii=1:length(f)
    if good_run(ii) ==1
        plot(sigmaees{ii}(end,4),sigmaeis{ii}(end,4)*sigmaies{ii}(end,4),'.','MarkerSize',16)
        wsee_vect(ii)    = sigmaees{ii}(end,4);
        wseisie_vect(ii) = sigmaeis{ii}(end,4)*sigmaies{ii}(end,4);
    end
end
[r,p] = corr(wsee_vect(good_run ==1)',wseisie_vect(good_run ==1)')
set(gca,'FontName','Arial','FontSize',12)
xlabel('w\Sigma^{EE}'),ylabel('w\Sigma^{EI}*\Sigma^{IE}')
print(fig1,['/Users/roach/work/wsigmaEE_contra_ipsi',date_str],'-dpdf','-painters')
%%
fig1=figure; hold on
for ii=1:length(f)
    if good_run(ii) == 1
        plot(sigmaees{ii}(end,1),sigmaiis{ii}(end,1),'.','MarkerSize',16)
    end
end
xlabel('\Sigma^{EE}'),ylabel('\Sigma^{II}')
print(fig1,['/Users/roach/work/sigmaEE_sigmaII',date_str],'-dpdf','-painters')
%% Absolute selectivity start
figure; hold on
for ii=1:length(f)
    if good_run(ii) == 1
        plot(1,sigmaees{ii}(1,1),'k.','MarkerSize',16)
        plot(2,sigmaeis{ii}(1,1),'k.','MarkerSize',16)
        plot(3,sigmaies{ii}(1,1),'k.','MarkerSize',16)
        plot(4,sigmaiis{ii}(1,1),'k.','MarkerSize',16)
    end
end
%%
fig1 = figure('Units','Inches','Position',[2,2,1.215,0.783],'PaperUnits','Inches','PaperPosition',[2,2,1.215,0.783]); hold on
x = get(gca,'XLim');
y = get(gca,'YLim');
plot([min(min(x),min(y)),max(max(x),max(y))],[min(min(x),min(y)),max(max(x),max(y))],'k--','LineWidth',0.5)
for ii=1:length(f)
    if good_run(ii) == 1
        plot(sigmaees{ii}(1,1),sigmaees{ii}(end,1),'g.','MarkerSize',5)
        plot(sigmaeis{ii}(1,1),sigmaeis{ii}(end,1),'m.','MarkerSize',5)
        plot(sigmaies{ii}(1,1),sigmaies{ii}(end,1),'c.','MarkerSize',5)
        plot(sigmaiis{ii}(1,1),sigmaiis{ii}(end,1),'.','Color',[1 0.5 0],'MarkerSize',5)
    end
end

xlabel('Selectivity before training'), ylabel('Selectivity after training')
set(gca,'FontName','Arial','FontSize',5)
print(fig1,['/Users/roach/work/startVend_connsel_fig',date_str],'-dpdf','-painters')

%%
fig1 = figure;

ax1 = axes('Position',[.1 .6 .25 .3]); hold on
ax2 = axes('Position',[.4 .6 .25 .3]); hold on
ax3 = axes('Position',[.1 .1 .25 .3]); hold on
ax4 = axes('Position',[.4 .1 .25 .3]); hold on
ax5 = axes('Position',[.7 .6 .25 .3]); hold on
ax6 = axes('Position',[.7 .1 .25 .3]); hold on

axes(ax1)
for ii=1:length(f)
    if good_run(ii) ==1
        plot(wee_corrs{ii},'g','LineWidth',2.0)
    end
end
ylim([0.65 1])
axes(ax2)
for ii=1:length(f)
    if good_run(ii) == 1
        plot(wei_corrs{ii},'m','LineWidth',2.0)
    end
end
ylim([0.65 1])
axes(ax3)
for ii=1:length(f)
    if good_run(ii) == 1
        plot(wie_corrs{ii},'c','LineWidth',2.0)
    end
end
ylim([0.65 1])
axes(ax4)
for ii=1:length(f)
    if good_run(ii) == 1
        plot(wii_corrs{ii},'Color',[1 0.5 0],'LineWidth',2.0)
    end
end
ylim([0.65 1])
axes(ax5)
for ii=1:length(f)
    if good_run(ii) == 1
        plot(wine_corrs{ii},'Color',[0.3 0.3 0.3],'LineWidth',2.0)
    end
end
axes(ax6)
for ii=1:length(f)
    if good_run(ii) == 1
        plot(wo_corrs{ii},'Color',[0.6 0.6 0.6],'LineWidth',2.0)
    end
end
print(fig1,['/Users/roach/work/weight_corr_fig',date_str],'-dpdf','-painters')
%%
fig1 = figure('Units','Inches','Position',[1,1,4,3],'PaperPosition',[1,1,4,3]); hold on
cmap = parula(length(f)+5);
[~,l] = sort(cellfun(@length,perfT_hists),'descend');
for ii=1:length(f)
    if good_run(l(ii)) == 1
        plot(perfT_hists{l(ii)},'Color',cmap(ii,:),'LineWidth',1.0)
    end
end
colormap(cmap(1:end-5,:))
cb1 = colorbar;
set(cb1,'YTick',[0 ceil(sum(good_run)/2)/sum(good_run) 1],'YTickLabel',{'1',int2str(ceil(sum(good_run)/2)),int2str(sum(good_run))})
set(cb1,'FontName','Helvetica','FontSize',10)
ylabel(cb1,'Rank','Rotation',-90,'Position',[3.5,0.5])
xl = get(gca,'XLim');
plot([xl(1) xl(2)],[0.85 0.85],'--','Color',[0.6 0.6 0.6],'LineWidth',1.0)
xlabel('Training Epoch'), ylabel('Performance')
set(gca,'FontName','Helvetica','FontSize',10)
xl = get(gca,'XLim');
yl = get(gca,'YLim');
xlim([-5 xl(2)]), ylim([-0.025 yl(2)])
print(fig1,['/Users/roach/work/perfXtrain_fig',date_str],'-dpdf','-painters')

%%
n_boot = 5000;
sel_corr = -1*ones(4,4,n_boot+1);

sels_ee
sels_ei
sels_ie
sels_ii

sel_corr(1,2,1) = corr(sels_ee(good_run==1),sels_ei(good_run==1));
sel_corr(1,3,1) = corr(sels_ee(good_run==1),sels_ie(good_run==1));
sel_corr(1,4,1) = corr(sels_ee(good_run==1),sels_ii(good_run==1));

sel_corr(2,3,1) = corr(sels_ei(good_run==1),sels_ie(good_run==1));
sel_corr(2,4,1) = corr(sels_ei(good_run==1),sels_ii(good_run==1));

sel_corr(3,4,1) = corr(sels_ie(good_run==1),sels_ii(good_run==1));

for ii= 2:(n_boot+1)
    xx = randperm(sum(good_run));
    y = find(good_run==1);
    x = y(xx);
    sel_corr(1,2,ii) = corr(sels_ee(good_run==1),sels_ei(x));
    sel_corr(1,3,ii) = corr(sels_ee(good_run==1),sels_ie(x));
    sel_corr(1,4,ii) = corr(sels_ee(good_run==1),sels_ii(x));

    sel_corr(2,3,ii) = corr(sels_ei(good_run==1),sels_ie(x));
    sel_corr(2,4,ii) = corr(sels_ei(good_run==1),sels_ii(x));

    sel_corr(3,4,ii) = corr(sels_ie(good_run==1),sels_ii(x));
end
sel_sig = -1*ones(4,4);
if (sum(sel_corr(1,2,1) > sel_corr(1,2,2:end))/n_boot) > 0.975 || (sum(sel_corr(1,2,1) > sel_corr(1,2,2:end))/n_boot) < 0.025
    sel_sig(1,2) = 1;
end
if (sum(sel_corr(1,3,1) > sel_corr(1,3,2:end))/n_boot) > 0.975 || (sum(sel_corr(1,3,1) > sel_corr(1,3,2:end))/n_boot) < 0.025
    sel_sig(1,3) = 1;
end
if (sum(sel_corr(1,4,1) > sel_corr(1,2,2:end))/n_boot) > 0.975 || (sum(sel_corr(1,4,1) > sel_corr(1,4,2:end))/n_boot) < 0.025
    sel_sig(1,4) = 1;
end

if (sum(sel_corr(2,3,1) > sel_corr(2,3,2:end))/n_boot) > 0.975 || (sum(sel_corr(2,3,1) > sel_corr(2,3,2:end))/n_boot) < 0.025
    sel_sig(2,3) = 1;
end
if (sum(sel_corr(2,4,1) > sel_corr(2,4,2:end))/n_boot) > 0.975 || (sum(sel_corr(2,4,1) > sel_corr(2,4,2:end))/n_boot) < 0.025
    sel_sig(2,4) = 1;
end

if (sum(sel_corr(3,4,1) > sel_corr(3,4,2:end))/n_boot) > 0.975 || (sum(sel_corr(3,4,1) > sel_corr(3,4,2:end))/n_boot) < 0.025
    sel_sig(3,4) = 1;
end

fig1 = figure('Units','Inches','Position',[1,1,4,3],'PaperPosition',[1,1,4,3]);
imagesc(sel_corr(:,:,1))
if sel_sig(1,2) == 1
    text(1.85,1,[num2str(round(sel_corr(1,2,1),2)),'*'])
else
    text(1.85,1,num2str(round(sel_corr(1,2,1),2)))
end
if sel_sig(1,3) == 1
    text(2.85,1,[num2str(round(sel_corr(1,3,1),2)),'*'])
else
    text(2.85,1,num2str(round(sel_corr(1,3,1),2)))
end
if sel_sig(1,4) == 1
    text(3.85,1,[num2str(round(sel_corr(1,4,1),2)),'*'])
else
    text(3.85,1,num2str(round(sel_corr(1,4,1),2)))
end

if sel_sig(2,3) == 1
    text(2.85,2,[num2str(round(sel_corr(2,3,1),2)),'*'])
else
    text(2.85,2,num2str(round(sel_corr(2,3,1),2)))
end
if sel_sig(2,4) == 1
    text(3.85,2,[num2str(round(sel_corr(2,4,1),2)),'*'])
else
    text(3.85,2,num2str(round(sel_corr(2,4,1),2)))
end

if sel_sig(3,4) == 1
    text(3.85,3,[num2str(round(sel_corr(3,4,1),2)),'*'])
else
    text(3.85,3,num2str(round(sel_corr(3,4,1),2)))
end
cmap = colormap;
cmap = [1,1,1;cmap];
colormap(cmap)
caxis([round(min(min(triu(sel_corr(:,:,1),1)))-0.0075,2) round(max(max(triu(sel_corr(:,:,1),1)))+0.005,2)])
cb1 = colorbar;
set(cb1,'YLim',[-0.035 0.58])
set(gca,'XTick',1:4,'XTickLabel',{'EE','EI','IE','II'})
set(gca,'YTick',1:4,'YTickLabel',{'EE','EI','IE','II'})
set(gca,'FontName','Helvetica','FontSize',12)
print(fig1,['/Users/roach/work/sigma_corr_figTT',date_str],'-dpdf','-painters')
% grid on
% subplot(1,2,2),imagesc(sel_sig)
%%

% ax1 = axes('Position',[0.1 0.6 0.3 0.3]);
% ax2 = axes('Position',[0.6 0.6 0.3 0.3]);
% ax3 = axes('Position',[0.1 0.1 0.3 0.3]);
% ax4 = axes('Position',[0.6 0.1 0.3 0.3]);


subplot(4,4,2), plot(sels_ee(good_run==1),sels_ei(good_run==1),'.')
subplot(4,4,3), plot(sels_ee(good_run==1),sels_ie(good_run==1),'.')
subplot(4,4,4), plot(sels_ee(good_run==1),sels_ii(good_run==1),'.')

subplot(4,4,7), plot(sels_ei(good_run==1),sels_ie(good_run==1),'.')
subplot(4,4,8), plot(sels_ei(good_run==1),sels_ii(good_run==1),'.')

subplot(4,4,12), plot(sels_ie(good_run==1),sels_ii(good_run==1),'.')
%%
fig1 = figure('Units','Inches','Position',[1,1,4,3],'PaperPosition',[1,1,4,3]);
ax1 = axes('Position',[0.1 0.6 0.3 0.3]); hold on
mx = round(max([max(squeeze(val_pred(1,:,1))),max(squeeze(val_pred(1,:,2)))])+0.005,2);
area([trial_starts_val(1) trial_starts_val(1) trial_end_val(1) trial_end_val(1)],[0 mx mx 0],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_pred(1,:,1)),'r','LineWidth',1)
plot(squeeze(val_pred(1,:,2)),'b','LineWidth',1)

ax2 = axes('Position',[0.6 0.6 0.3 0.3]); hold on
mx = round(max([max(squeeze(val_pred(end,:,1))),max(squeeze(val_pred(end,:,2)))])+0.005,2);
area([trial_starts_val(1) trial_starts_val(1) trial_end_val(1) trial_end_val(1)],[0 mx mx 0],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_pred(end,:,1)),'r','LineWidth',1)
plot(squeeze(val_pred(end,:,2)),'b','LineWidth',1)

ax3 = axes('Position',[0.1 0.35 0.3 0.15]); hold on
mx1 = round(max(max(squeeze(val_act_e(1,:,:))))+0.005,2);
mx2 = round(max(max(squeeze(val_act_i(1,:,:))))+0.005,2);
mx = max(mx1,mx2);
area([trial_starts_val(1) trial_starts_val(1) trial_end_val(1) trial_end_val(1)],[-0.2 mx1 mx1 -0.2],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_act_e(1,:,auc_e(:,2)==1)),'r','LineWidth',1)
plot(squeeze(val_act_e(1,:,auc_e(:,2)==-1)),'b','LineWidth',1)
% plot(squeeze(val_act_e(1,:,auc_e(:,2)==0)),'Color',[0 0 0],'LineWidth',1)
ax4 = axes('Position',[0.1 0.15 0.3 0.15]); hold on
area([trial_starts_val(1) trial_starts_val(1) trial_end_val(1) trial_end_val(1)],[-0.1 mx2 mx2 -0.1],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_act_i(1,:,auc_i(:,2)==1)),'r','LineWidth',1)
plot(squeeze(val_act_i(1,:,auc_i(:,2)==-1)),'b','LineWidth',1)
% plot(squeeze(val_act_i(1,:,auc_i(:,2)==0)),'Color',[0 0 0],'LineWidth',1)

ax5 = axes('Position',[0.6 0.35 0.3 0.15]); hold on
mx1 = round(max(max(squeeze(val_act_e(end,:,:))))+0.005,2);
mx2 = round(max(max(squeeze(val_act_i(end,:,:))))+0.005,2);
% mx = max(mx1,mx2);
area([trial_starts_val(end) trial_starts_val(end) trial_end_val(end) trial_end_val(end)],[-0.2 mx1 mx1 -0.2],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_act_e(end,:,auc_e(:,2)==1)),'r','LineWidth',1)
plot(squeeze(val_act_e(end,:,auc_e(:,2)==-1)),'b','LineWidth',1)
% plot(squeeze(val_act_e(1,:,auc_e(:,2)==0)),'Color',[0 0 0],'LineWidth',1)
ax6 = axes('Position',[0.6 0.15 0.3 0.15]); hold on
area([trial_starts_val(end) trial_starts_val(end) trial_end_val(end) trial_end_val(end)],[-0.1 mx2 mx2 -0.1],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_act_i(end,:,auc_i(:,2)==1)),'r','LineWidth',1)
plot(squeeze(val_act_i(end,:,auc_i(:,2)==-1)),'b','LineWidth',1)
% plot(squeeze(val_act_i(1,:,auc_i(:,2)==0)),'Color',[0 0 0],'LineWidth',1)
print(fig1,['/Users/roach/work/ex_trls',date_str],'-dpdf','-painters')

%%
[~,ll] = sort(sels_ee,'descend');
cmap = parula(length(sels_ee));
B = 0:0.05:0.5;
load([f{1},'/evo_sigmasel.mat'],'auc_e','auc_i')
e_choice_tuning = zeros(length(sels_ee),size(auc_e,1));
i_choice_tuning = zeros(length(sels_ee),size(auc_i,1));

subplot(2,1,1), hold on
subplot(2,1,2), hold on
for ii=1:length(sels_ee)
    load([f{ii},'/evo_sigmasel.mat'],'auc_e','auc_i')
    e_choice_tuning(ii,:) = abs(auc_e(:,1)-0.5);
    i_choice_tuning(ii,:) = abs(auc_i(:,1)-0.5);
    [xe] = histcounts(e_choice_tuning(ii,:),B);
    [xi] = histcounts(i_choice_tuning(ii,:),B);
    subplot(2,1,1), plot(B(2:end),xe/sum(xe),'Color',cmap(ii,:),'LineWidth',1)
    subplot(2,1,2), plot(B(2:end),xi/sum(xi),'Color',cmap(ii,:),'LineWidth',1)
end
%%
[xe] = histcounts(reshape(e_choice_tuning,[],1),B); hold on
[xi] = histcounts(reshape(i_choice_tuning,[],1),B);
plot([median(reshape(e_choice_tuning,[],1)),median(reshape(e_choice_tuning,[],1))],[0 round(max([max(xe/sum(xe)) max(xi/sum(xi))]),1)+0.05],'-','Color',[0.6 0.6 0.6],'LineWidth',2)
plot([median(reshape(i_choice_tuning,[],1)),median(reshape(i_choice_tuning,[],1))],[0 round(max([max(xe/sum(xe)) max(xi/sum(xi))]),1)+0.05],'--','Color',[0.6 0.6 0.6],'LineWidth',2)
plot(B(2:end),xe/sum(xe),'-k','LineWidth',2), plot(B(2:end),xi/sum(xi),'--k','LineWidth',2)
ranksum(reshape(e_choice_tuning,[],1),reshape(i_choice_tuning,[],1))
xlim([0.04 0.51]), ylim([0 0.55])
box off



%%
fig1 = figure('Units','centimeters','PaperUnits','centimeters','Position',[1,1,17.4,10.0],'PaperPosition',[1,1,17.4,10.0]);
[~,mxe] = max(auc_e(:,1));
[~,mxi] = max(auc_i(:,1));
[~,mne] = min(auc_e(:,1));
[~,mni] = min(auc_i(:,1));

h1 = axes('Position',[0,0,1,1],'Color','none','XColor','none','YColor','none');
ax(1) = axes('Position',[0.075 0.6 0.2 0.3],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]);

ax(2) = axes('Position',[0.055 0.35 0.1 0.1],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on
ax(3) = axes('Position',[0.165 0.35 0.1 0.1],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on
ax(4) = axes('Position',[0.055 0.225 0.1 0.1],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on
ax(5) = axes('Position',[0.055 0.1 0.1 0.1],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on
ax(6) = axes('Position',[0.165 0.225 0.1 0.1],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on
ax(7) = axes('Position',[0.165 0.1 0.1 0.1],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on

ax(8) = axes('Position',[0.325 0.85 0.15 0.125],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on
ax(9) = axes('Position',[0.325 0.6 0.15 0.125],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on

ax(10) = axes('Position',[0.325 0.1 0.15 0.35],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on
ax(11) = axes('Position',[0.535 0.1 0.15 0.35],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on

ax(12) = axes('Position',[0.535 0.85 0.15 0.125],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on
ax(13) = axes('Position',[0.535 0.6 0.15 0.125],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on

ax(14) = axes('Position',[0.885 0.6 0.1 0.35],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on
ax(15) = axes('Position',[0.885 0.1 0.1 0.35],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on


axes(ax(2))
mx1 = round(max([max(squeeze(val_pred(1,:,1))),max(squeeze(val_pred(1,:,2)))])+0.005,2);
mx2 = round(max([max(squeeze(val_pred(end,:,1))),max(squeeze(val_pred(end,:,2)))])+0.005,2);
mx = round(max([mx1,mx2])+0.05,1);

area([trial_starts_val(1) trial_starts_val(1) trial_end_val(1) trial_end_val(1)],[0 mx mx 0],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_pred(1,:,1)),'r','LineWidth',1)
plot(squeeze(val_pred(1,:,2)),'b','LineWidth',1)
set(ax(2),'YTick',[0 mx],'YLim',[0 mx])

axes(ax(3))

area([trial_starts_val(1) trial_starts_val(1) trial_end_val(1) trial_end_val(1)],[0 mx mx 0],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_pred(end,:,1)),'r','LineWidth',1)
plot(squeeze(val_pred(end,:,2)),'b','LineWidth',1)
set(ax(3),'YTick',[0 mx],'YLim',[0 mx])

axes(ax(4))
mx1 = round(max(max(squeeze(val_act_e(1,:,:))))+0.005,2);
mx2 = round(max(max(squeeze(val_act_i(1,:,:))))+0.005,2);
mx = max(mx1,mx2);
area([trial_starts_val(1) trial_starts_val(1) trial_end_val(1) trial_end_val(1)],[-0.2 mx1 mx1 -0.2],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_act_e(1,:,mxe)),'r','LineWidth',1)
plot(squeeze(val_act_e(1,:,mne)),'b','LineWidth',1)
% plot(squeeze(val_act_e(1,:,auc_e(:,2)==0)),'Color',[0 0 0],'LineWidth',1)
set(ax(4),'YTick',[0 ceil(mx1)])

axes(ax(5))
area([trial_starts_val(1) trial_starts_val(1) trial_end_val(1) trial_end_val(1)],[-0.1 mx2 mx2 -0.1],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_act_i(1,:,mxi)),'r','LineWidth',1)
plot(squeeze(val_act_i(1,:,mni)),'b','LineWidth',1)
% plot(squeeze(val_act_i(1,:,auc_i(:,2)==0)),'Color',[0 0 0],'LineWidth',1)
set(ax(5),'YTick',[0 ceil(mx2)])

axes(ax(6))
% mx1 = round(max(max(squeeze(val_act_e(end,:,:))))+0.005,2);
% mx2 = round(max(max(squeeze(val_act_i(end,:,:))))+0.005,2);
% mx = max(mx1,mx2);
area([trial_starts_val(end) trial_starts_val(end) trial_end_val(end) trial_end_val(end)],[-0.2 mx1 mx1 -0.2],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_act_e(end,:,mxe)),'r','LineWidth',1)
plot(squeeze(val_act_e(end,:,mne)),'b','LineWidth',1)
% plot(squeeze(val_act_e(1,:,auc_e(:,2)==0)),'Color',[0 0 0],'LineWidth',1)
set(ax(6),'YTick',[0 ceil(mx1)])

axes(ax(7))
area([trial_starts_val(end) trial_starts_val(end) trial_end_val(end) trial_end_val(end)],[-0.1 mx2 mx2 -0.1],'FaceColor',[0.6 0.6 0.6],'EdgeColor','none')
plot(squeeze(val_act_i(end,:,mxi)),'r','LineWidth',1)
plot(squeeze(val_act_i(end,:,mni)),'b','LineWidth',1)
% plot(squeeze(val_act_i(1,:,auc_i(:,2)==0)),'Color',[0 0 0],'LineWidth',1)
set(ax(7),'YTick',[0 ceil(mx2)])

axes(ax(2))
ylabel({'output';'[AU]'}), set(ax(2),'XTick',[0 60],'XTickLabel',[])
% set(ax(2),'Ytick',[0 1])

axes(ax(3))
set(ax(3),'XTick',[0 60],'XTickLabel',[])
set(ax(3),'YTickLabel',[])

set(ax(6),'XTick',[0 60],'XTickLabel',[],'YTick',[0,ceil(mx1)],...
    'YTickLabel',{},'YLim',[0,ceil(mx1)])

axes(ax(4))
ylabel({'exe.';'act [AU]'})
set(ax(4),'XTickLabel',{}),set(ax(4),'YTick',[0,ceil(mx1)],'YLim',[0,ceil(mx1)],'XTick',[0 60],'XTickLabel',[])

axes(ax(5))
xlabel('time [AU]'), ylabel({'inh.';'act [AU]'}),set(ax(5),'YTick',[0,ceil(mx2)],'XTick',[0 60],'YLim',[0,ceil(mx2)])

axes(ax(7))
xlabel('time [AU]'),set(ax(7),'YTick',[0,ceil(mx2)],'YTickLabel',{},'XTick',[0 60],'YLim',[0,ceil(mx2)])

%%%%%%%%%%

pch_arry = zeros(length(p_ch1s{1}(:,1)),length(p_ch1s));
uu = find(good_run==1);
axes(ax(8))
for ii=1:length(p_ch1s)
    if good_run(ii) == 1
        plot(p_ch1s{ii}(:,1),p_ch1s{ii}(:,2),'-','LineWidth',1,'Color',[0.75 0.75 0.75],'MarkerSize',16)
        pch_arry(:,ii) = p_ch1s{ii}(:,2);
    end
end
shadedErrorBar(p_ch1s{uu(1)}(:,1),nanmean(pch_arry(:,good_run==1),2),nanstd(pch_arry(:,good_run==1),1,2)/sqrt(sum(good_run==1)),'lineprops','r')
ylim([-0.05 1.05]), set(ax(8),'YTick',[0. 1.0]),set(ax(8),'XLim',[-22 22])
xlabel(''),ylabel('P(Ch. 1)')
% set(gca,'FontName','Arial','FontSize',11)

axes(ax(9))
ntrials_arry = zeros(length(ntrials{1}(:,1)),length(ntrials));

for ii=1:length(p_ch1s)
    if good_run(ii) == 1
        plot(ntrials{ii}(:,1),ntrials{ii}(:,2),'-','LineWidth',1,'Color',[0.75 0.75 0.75],'MarkerSize',16)
        ntrials_arry(:,ii) = ntrials{ii}(:,2);
    end
end
shadedErrorBar(ntrials{uu(1)}(:,1),nanmean(ntrials_arry(:,good_run==1),2),nanstd(ntrials_arry(:,good_run==1),1,2)/sqrt(sum(good_run==1)),'lineprops','r')
yl = get(ax(9),'YLim');
yl(1) = floor(yl(1)/10)*10;
yl(2) = ceil(yl(2)/10)*10;
xlabel('Stimulus'),ylabel('% Trial Comp.'), set(ax(9),'YTick',[yl(1) yl(2)]),set(ax(9),'XLim',[-22 22],'YLim',yl)
% set(gca,'FontName','Arial','FontSize',11)

%%%%%
axes(ax(10))
cmap = parula(length(f)+5);
[~,l] = sort(cellfun(@length,perfT_hists),'descend');
for ii=1:length(f)
    if good_run(l(ii)) == 1
        plot(perfT_hists{l(ii)},'Color',cmap(ii,:),'LineWidth',1.0)
    end
end
colormap(ax(10),cmap(1:end-5,:))
% cb1 = colorbar;
% set(cb1,'YTick',[0 ceil(sum(good_run)/2)/75 1],'YTickLabel',{'1',int2str(ceil(sum(good_run)/2)),int2str(sum(good_run))})
% set(cb1,'FontName','Helvetica','FontSize',10)
% ylabel(cb1,'Rank','Rotation',-90,'Position',[3.5,0.5])
xl = get(gca,'XLim');
plot([xl(1) xl(2)],[0.85 0.85],'--','Color',[0.6 0.6 0.6],'LineWidth',1.0)
xlabel('Training Epoch'), ylabel('Performance')
% set(gca,'FontName','Helvetica','FontSize',10)
xl = get(ax(10),'XLim');
yl = get(ax(10),'YLim');
xlim([-5 ceil(xl(2)/10)*10]), ylim([0 ceil(yl(2)*10)/10])
set(ax(10),'XTick',[0 ceil(xl(2)/10)*10],'YTick',[0 ceil(yl(2)*10)/10])
%%%%%%

axes(ax(11))
plot([0.5 4.5],[0 0],'--','Color',[0.5 0.5 0.5],'LineWidth',2.0)
sels_ee = zeros(length(f),1);
sels_ie = zeros(length(f),1);
sels_ei = zeros(length(f),1);
sels_ii = zeros(length(f),1);
for ii=1:length(f)
    if good_run(ii) == 1
        sels_ee(ii) = sigmaees{ii}(end,1); 
        sels_ei(ii) = sigmaeis{ii}(end,1);
        sels_ie(ii) = sigmaies{ii}(end,1);
        sels_ii(ii) = sigmaiis{ii}(end,1);
    end
end

[~,ll] = sort(sels_ee,'descend');
cmap = parula(length(ll)+5);
for ii=1:length(f)
    if good_run(ii) == 1
        plot([1,2,3,4],[sigmaees{ll(ii)}(end,1),sigmaeis{ll(ii)}(end,1),sigmaies{ll(ii)}(end,1),sigmaiis{ll(ii)}(end,1)],'.-','Color',cmap(ii,:),'MarkerSize',16,'LineWidth',2)
    %     plot(2,sigmaeis{ii}(end,1),'k.','MarkerSize',16)
    %     plot(3,sigmaies{ii}(end,1),'k.','MarkerSize',16)
    %     plot(4,sigmaiis{ii}(end,1),'k.','MarkerSize',16)
    %     sels_ee(ii) = sigmaees{ii}(end,1); 
    %     sels_ei(ii) = sigmaeis{ii}(end,1);
    %     sels_ie(ii) = sigmaies{ii}(end,1);
    %     sels_ii(ii) = sigmaiis{ii}(end,1);
    end
end
errorbar([1,2,3,4],[mean(sels_ee(good_run==1)),mean(sels_ei(good_run==1)),mean(sels_ie(good_run==1)),mean(sels_ii(good_run==1))]...
    ,[std(sels_ee(good_run==1))/sqrt(sum(good_run)),std(sels_ei(good_run==1))/sqrt(sum(good_run)),...
    std(sels_ie(good_run==1))/sqrt(sum(good_run)),std(sels_ii(good_run==1))/sqrt(sum(good_run))],'k','LineWidth',2.0)
yl = get(ax(11),'YLim');
ylim([yl(1)-0.025 yl(2)]), %set(ax(11),'YTick',[-0.2 0 0.8])
ylabel('\Sigma')
set(gca,'XTick',[1,2,3,4],'XTickLabel',{'\Sigma^{EE}','\Sigma^{EI}','\Sigma^{IE}','\Sigma^{II}'})
colormap(ax(11),cmap(1:end-5,:))
cb1 = colorbar('Position',[0.765 0.1 0.0075 0.15]);
set(cb1,'YTick',[0 ceil(sum(good_run)/2)/sum(good_run) 1],'YTickLabel',{'1',int2str(ceil(sum(good_run)/2)),int2str(sum(good_run))}), 
% set(cb1,'FontName','Arial','FontSize',11)
ylabel(cb1,'Rank','Rotation',-90,'Position',[6.5 0.5])

%%%%

axes(ax(12))
fsele = zeros(length(auces),1);
fseli = zeros(length(auces),1);
for ii = 1:length(f)
    if good_run(ii) == 1
        fsele(ii) = sum(auces{ii}(:,2) ~= 0)/length(auces{ii}(:,2));
        fseli(ii) = sum(aucis{ii}(:,2) ~= 0)/length(aucis{ii}(:,2));
    end
end
[ne,ee] = histcounts(fsele(good_run==1),0:0.05:1);
[ni,ei] = histcounts(fseli(good_run==1),0:0.05:1);

plot(ee(2:end),ne/sum(ne),'k','LineWidth',2.0)
plot(ei(2:end),ni/sum(ni),'k--','LineWidth',2.0)
ylabel('Fraction networks'), xlabel('Fraction Selective')
yl = ceil(max([max(ne/sum(ne)) max(ni/sum(ni))])*10)/10;
xlim([0.5 1]), ylim([0 yl])
set(ax(12),'XTick',[0.5 1],'YTick',[0 yl])
el = find(ne~=0);
il = find(ni~=0);
xl = min(el(1),il(1));
xh = max(el(end),il(end));
if xl<2
    xl=2;
end
if xh >= length(ee)-1
    xh = length(ee)-2;
end
% xlim([ee(xl-1),ee(xh+2)])
% set(gca,'FontName','Arial','FontSize',11)

%%%%%

axes(ax(13))
[xe] = histcounts(reshape(e_choice_tuning,[],1),B); hold on
[xi] = histcounts(reshape(i_choice_tuning,[],1),B);
plot([median(reshape(e_choice_tuning,[],1)),median(reshape(e_choice_tuning,[],1))],[0 round(max([max(xe/sum(xe)) max(xi/sum(xi))]),1)+0.05],'-','Color',[0.6 0.6 0.6],'LineWidth',2)
plot([median(reshape(i_choice_tuning,[],1)),median(reshape(i_choice_tuning,[],1))],[0 round(max([max(xe/sum(xe)) max(xi/sum(xi))]),1)+0.05],'--','Color',[0.6 0.6 0.6],'LineWidth',2)
plot(B(2:end),xe/sum(xe),'-k','LineWidth',2), plot(B(2:end),xi/sum(xi),'--k','LineWidth',2)
ranksum(reshape(e_choice_tuning,[],1),reshape(i_choice_tuning,[],1))
xlim([0.04 0.51]), ylim([0 0.55])
set(ax(13),'XTick',[0.05 0.5],'YTick',[0 0.5])
xlabel('Choice Tuning'), ylabel('fraction cells')
box off

%%%%

axes(ax(14))
imagesc(sel_corr(:,:,1)),set(ax(14),'YDir','reverse')
if sel_sig(1,2) == 1
    text(1.65,1,[num2str(round(sel_corr(1,2,1),2)),'*'],'FontName','Arial','FontSize',6)
else
    text(1.65,1,num2str(round(sel_corr(1,2,1),2)),'FontName','Arial','FontSize',6)
end
if sel_sig(1,3) == 1
    text(2.65,1,[num2str(round(sel_corr(1,3,1),2)),'*'],'FontName','Arial','FontSize',6)
else
    text(2.65,1,num2str(round(sel_corr(1,3,1),2)),'FontName','Arial','FontSize',6)
end
if sel_sig(1,4) == 1
    text(3.65,1,[num2str(round(sel_corr(1,4,1),2)),'*'],'FontName','Arial','FontSize',6)
else
    text(3.65,1,num2str(round(sel_corr(1,4,1),2)),'FontName','Arial','FontSize',6)
end

if sel_sig(2,3) == 1
    text(2.65,2,[num2str(round(sel_corr(2,3,1),2)),'*'],'FontName','Arial','FontSize',6)
else
    text(2.65,2,num2str(round(sel_corr(2,3,1),2)),'FontName','Arial','FontSize',6)
end
if sel_sig(2,4) == 1
    text(3.65,2,[num2str(round(sel_corr(2,4,1),2)),'*'],'FontName','Arial','FontSize',6)
else
    text(3.65,2,num2str(round(sel_corr(2,4,1),2)),'FontName','Arial','FontSize',6)
end

if sel_sig(3,4) == 1
    text(3.65,3,[num2str(round(sel_corr(3,4,1),2)),'*'],'FontName','Arial','FontSize',6)
else
    text(3.65,3,num2str(round(sel_corr(3,4,1),2)),'FontName','Arial','FontSize',6)
end
axis tight
cmap = colormap;
cmap = [1,1,1;cmap];
colormap(ax(14),cmap)
caxis([round(min(min(triu(sel_corr(:,:,1),1)))-0.015,2) round(max(max(triu(sel_corr(:,:,1),1)))+0.005,1)])
cb2 = colorbar('Position',[0.95 0.6 0.0075 0.35]);
set(cb2,'YLim',[-0.04 round(max(max(triu(sel_corr(:,:,1),1)))+0.005,1)]...
    ,'YTick',[-0.04 round(max(max(triu(sel_corr(:,:,1),1)))+0.005,1)])
set(gca,'XTick',1:4,'XTickLabel',{'EE','EI','IE','II'})
set(gca,'YTick',1:4,'YTickLabel',{'EE','EI','IE','II'})

%%%

axes(ax(15))
for ii=1:length(f)
    if good_run(ii) ==1
        plot(sigmaees{ii}(end,1),sigmaeis{ii}(end,1)*sigmaies{ii}(end,1),'.k','MarkerSize',16)
        see_vect(ii)    = sigmaees{ii}(end,1);
        seisie_vect(ii) = sigmaeis{ii}(end,1)*sigmaies{ii}(end,1);
    end
end
[r,p] = corr(see_vect(good_run ==1)',seisie_vect(good_run ==1)')
xlabel('\Sigma^{EE}'),ylabel('\Sigma^{EI}*\Sigma^{IE}')
xl = get(ax(15),'XLim');
yl = get(ax(15),'YLim');
xl(1) = floor(xl(1)*10)/10;
xl(2) = ceil(xl(2)*10)/10;
yl(1) = floor(yl(1)*1000)/1000;
yl(2) = ceil(yl(2)*1000)/1000;
set(ax(15),'XLim',xl,'YLim',yl,'XTick',xl,'YTick',yl)
% ax(14) = axes('Position',[0.885 0.6 0.1 0.35],'FontName','Arial','FontSize',8); hold on
% ax(15) = axes('Position',[0.885 0.1 0.1 0.35],'FontName','Arial','FontSize',8); hold on
% 

% ax(10) = axes('Position',[0.325 0.1 0.15 0.35],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on
% ax(11) = axes('Position',[0.535 0.1 0.15 0.35],'FontName','Arial','FontSize',8,'LineWidth',1,'Xcolor',[0 0 0],'YColor',[0 0 0]); hold on
set(ax(10),'Position',[0.325 0.1 0.175 0.35])
set(ax(11),'Position',[0.535 0.1 0.195 0.35])

set(ax(15),'Position',[0.78 0.6 0.2 0.35])
set(ax(14),'Position',[0.78 0.1 0.17 0.35])
set(cb1,'Position',[0.69 0.3 0.0075 0.15])
set(cb2,'Position',[0.95 0.1 0.0075 0.35])
for ii=1:length(ax)
    set(ax(ii),'TickDir','out')
end
set(cb1,'TickDir','out')
set(cb2,'TickDir','out')
%%
ne = size(e_choice_tuning,2);
save(['/Users/roach/work/rnn_circuit_',int2str(size(e_choice_tuning,2)),'.mat'],'see_vect','seisie_vect','ne')
exportgraphics(fig1,['/Users/roach/work/rnn_circuit_fig_',int2str(size(e_choice_tuning,2)),'.pdf'],'ContentType','vector')
%%
%%
fig1 = figure('Units','centimeters','PaperUnits',...
    'centimeters','Position',[1,1,8.5,5.5],'PaperPosition',[1,1,8.5,5.5]); hold on
chronos_arry = zeros(length(chronos{ii}(:,1)),length(chronos));

for ii=1:length(p_ch1s)
    if good_run(ii) == 1
        plot(chronos{ii}(:,1),chronos{ii}(:,2),'-','LineWidth',1,'Color',[0.75 0.75 0.75],'MarkerSize',16)
        chronos_arry(:,ii) = chronos{ii}(:,2);
    end
end
shadedErrorBar(chronos{uu(1)}(:,1),nanmean(chronos_arry(:,good_run==1),2),...
    nanstd(chronos_arry(:,good_run==1),1,2)/sqrt(sum(good_run==1)),'lineprops',{'r','LineWidth',1})
%ylim([-0.05 1.05])
xlabel('Stimulus'), ylabel('reaction time [AU]')
set(gca,'FontName','Arial','FontSize',8)
exportgraphics(fig1,'/Users/roach/work/rnn_chrono_sat.pdf','ContentType','vector')
%%
trn_trials = cellfun(@length,perfT_hists).*double(batch_size);
mean_ttrials = mean(trn_trials)
std_ttrials = std(trn_trials)

mean_choicetuningE = mean(reshape(e_choice_tuning,[],1))
std_choicetuningE = std(reshape(e_choice_tuning,[],1))
mean_choicetuningI = mean(reshape(i_choice_tuning,[],1))
std_choicetuningI = std(reshape(i_choice_tuning,[],1))

log10(ranksum(reshape(e_choice_tuning,[],1),reshape(i_choice_tuning,[],1)))
log10(ranksum(fsele,fseli))

mfsele = mean(fsele)
sfsele = std(fsele)
mfseli = mean(fseli)
sfseli = std(fseli)


msEE = mean(sels_ee)
ssEE = std(sels_ee)

msEI = mean(sels_ei)
ssEI = std(sels_ei)

msIE = mean(sels_ie)
ssIE = std(sels_ie)

msII = mean(sels_ii)
ssII = std(sels_ii)