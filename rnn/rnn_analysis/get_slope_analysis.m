
clear
%% slope 0.5
f = uipickfiles;

pairs_s05 = zeros(length(f),3);

for oo = 1:length(f)
    load([f{oo},'/evo_sigmasel.mat'])
    pairs_s05(oo,1) = sigma_ee(end,1);
    pairs_s05(oo,2) = sigma_ei(end,1);
    pairs_s05(oo,3) = sigma_ie(end,1);
end

%% slope 1.0
f = uipickfiles;

pairs_s10 = zeros(length(f),3);

for oo = 1:length(f)
    load([f{oo},'/evo_sigmasel.mat'])
    pairs_s10(oo,1) = sigma_ee(end,1);
    pairs_s10(oo,2) = sigma_ei(end,1);
    pairs_s10(oo,3) = sigma_ie(end,1);
end

%% slope 1.5
f = uipickfiles;

pairs_s15 = zeros(length(f),3);

for oo = 1:length(f)
    load([f{oo},'/evo_sigmasel.mat'])
    pairs_s15(oo,1) = sigma_ee(end,1);
    pairs_s15(oo,2) = sigma_ei(end,1);
    pairs_s15(oo,3) = sigma_ie(end,1);
end
%%
fig1 = figure('Units','inches','Position',[5,5,1.9,1.35],'PaperPosition',[5,5,1.9,1.35]);

ax = axes('Position',[.15 .15 0.7 0.7]); hold on

mx_x = round(max([max(pairs_s05(:,1)),max(pairs_s10(:,1)),max(pairs_s15(:,1))])+0.05,1);
mn_x = round(min([min(pairs_s05(:,1)),min(pairs_s10(:,1)),min(pairs_s15(:,1))])-0.05,1);

mx_y = round(max([max(pairs_s05(:,2).*pairs_s05(:,3)),max(pairs_s10(:,2).*pairs_s10(:,3)),max(pairs_s15(:,2).*pairs_s15(:,3))])+0.005,2);
mn_y = round(min([min(pairs_s05(:,2).*pairs_s05(:,3)),min(pairs_s10(:,2).*pairs_s10(:,3)),min(pairs_s15(:,2).*pairs_s15(:,3))])-0.005,2);

plot(pairs_s05(:,1),pairs_s05(:,2).*pairs_s05(:,3),'g.','MarkerSize',8)
plot(pairs_s10(:,1),pairs_s10(:,2).*pairs_s10(:,3),'.','Color',[0.6 0.6 0.6],'MarkerSize',8)
plot(pairs_s15(:,1),pairs_s15(:,2).*pairs_s15(:,3),'m.','MarkerSize',8)

set(ax,'XLim',[mn_x mx_x],'YLim',[mn_y mx_y])
set(ax,'XTick',[mn_x mx_x],'YTick',[mn_y 0 mx_y])
set(ax,'TickDir','out')

exportgraphics(fig1,'/Users/roach/work/sat_paper_panels/sigma_slope_scatter.pdf','ContentType','vector')

%% histograms 1
fig1 = figure('Units','inches','Position',[5,5,2.444,1.76],'PaperPosition',[5,5,2.444,1.76]);
ax = axes('Position',[.15 .15 0.7 0.7]); hold on

b_ee = linspace(mn_x,mx_x,15);
c_05 = histcounts(pairs_s05(:,1),b_ee);
c_10 = histcounts(pairs_s10(:,1),b_ee);
c_15 = histcounts(pairs_s15(:,1),b_ee);

plot(b_ee,[0,c_05/sum(c_05)],'g','LineWidth',1)
plot(b_ee,[0,c_10/sum(c_10)],'Color',[0.6 0.6 0.6],'LineWidth',1)
plot(b_ee,[0,c_15/sum(c_15)],'m','LineWidth',1)

mxx = round(max([max(c_05/sum(c_05)) max(c_10/sum(c_10)) max(c_15/sum(c_15))])+0.005,2);

set(ax,'YLim',[0 mxx],'XLim',[b_ee(1) b_ee(end)])
set(ax,'YTick',[0 mxx],'XTick',[b_ee(1) b_ee(end)])
set(ax,'TickDir','out')
exportgraphics(fig1,'/Users/roach/work/sat_paper_panels/sigmaEE_slope_hist.pdf','ContentType','vector')

%% histograms 2
fig1 = figure('Units','inches','Position',[5,5,2.5492,1.8604],'PaperPosition',[5,5,2.5492,1.8604]);
ax = axes('Position',[.15 .15 0.7 0.7]); hold on

b_ei = linspace(mn_y,mx_y,15);
c_05 = histcounts(pairs_s05(:,2).*pairs_s05(:,3),b_ei);
c_10 = histcounts(pairs_s10(:,2).*pairs_s10(:,3),b_ei);
c_15 = histcounts(pairs_s15(:,2).*pairs_s15(:,3),b_ei);

plot(b_ei,[0,c_05/sum(c_05)],'g','LineWidth',1)
plot(b_ei,[0,c_10/sum(c_10)],'Color',[0.6 0.6 0.6],'LineWidth',1)
plot(b_ei,[0,c_15/sum(c_15)],'m','LineWidth',1)

mxx = round(max([max(c_05/sum(c_05)) max(c_10/sum(c_10)) max(c_15/sum(c_15))])+0.005,2);

set(ax,'YLim',[0 mxx],'XLim',[b_ei(1) b_ei(end)])
set(ax,'YTick',[0 mxx],'XTick',[b_ei(1) 0 b_ei(end)])
set(ax,'TickDir','out')
exportgraphics(fig1,'/Users/roach/work/sat_paper_panels/sigmaEI_slope_hist.pdf','ContentType','vector')
