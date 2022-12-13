clear
f = uipickfiles('FilterSpec','/Users/roach/work/RNNs/');

% p_mag = [1,0.25,0.05,-0.05,-0.25,-1];

p_mag = -1:0.25:1;

drt = zeros(length(f),length(p_mag));
dnt = zeros(length(f),length(p_mag));
dperf = zeros(length(f),length(p_mag));
dacc  = zeros(length(f),length(p_mag));

b_chrono = zeros(length(f),length(p_mag),21);
b_psycho = zeros(length(f),length(p_mag),21);
b_trialc = zeros(length(f),length(p_mag),21);

p_chrono = zeros(length(f),length(p_mag),21);
p_psycho = zeros(length(f),length(p_mag),21);
p_trialc = zeros(length(f),length(p_mag),21);


for jj = 1:length(p_mag)
    
    pstr = num2str(p_mag(jj));
    if contains(pstr,'.')
    else
        pstr = [pstr,'.0'];
    end
    
    for oo = 1:length(f)
        load([f{oo},'/perturb_E_0.0_I_',pstr,'.mat'])
        drt(oo,jj,:) = mean((perturb_chrono-baseline_chrono)./baseline_chrono);
        dnt(oo,jj,:) = mean((perturb_ntrial-baseline_ntrial)./baseline_ntrial);
        dperf(oo,jj,:) =  baseline_perf_total - perturb_perf_total;
        dacc(oo,jj,:) =  perturb_perf - baseline_perf;
        
        b_chrono(oo,jj,:) = baseline_chrono;
        b_psycho(oo,jj,:) = baseline_psycho;
        b_trialc(oo,jj,:) = baseline_ntrial;

        p_chrono(oo,jj,:) = perturb_chrono;
        p_psycho(oo,jj,:) = perturb_psycho;
        p_trialc(oo,jj,:) = perturb_ntrial;
    end
end

% plot(sort(dnt))
%%
figure, hold on
plot(binns,squeeze(mean(b_chrono(:,1,:),1)),'--')
plot(binns,squeeze(mean(b_chrono(:,end,:),1)),'--')

plot(binns,squeeze(mean(p_chrono(:,1,:),1)),'m-')

plot(binns,squeeze(mean(p_chrono(:,end,:),1)),'g-')

%%
figure, hold on
shadedErrorBar(binns,squeeze(mean(b_trialc(:,1,:)./100,1)),...
    squeeze(std(b_trialc(:,1,:)./100,1,1))/sqrt(size(b_trialc,1)),...
    'LineProps',{'--','Color',[0.6 0.6 0.6],...
    'LineWidth',1})
shadedErrorBar(binns,squeeze(mean(b_trialc(:,end,:)./100,1)),...
    squeeze(std(b_trialc(:,end,:)./100,1,1))/sqrt(size(b_trialc,1)),...
    'lineprops',{'--','Color',[0.6 0.6 0.6],...
    'LineWidth',1})

shadedErrorBar(binns,squeeze(mean(p_trialc(:,1,:)./100,1)),...
    squeeze(std(p_trialc(:,1,:)./100,1,1))/sqrt(size(p_trialc,1)),...
    'lineprops',{'m-','LineWidth',1})

shadedErrorBar(binns,squeeze(mean(p_trialc(:,end,:)./100,1)),...
    squeeze(std(p_trialc(:,end,:)./100,1,1))/sqrt(size(p_trialc,1))...
    ,'lineprops',{'g-','LineWidth',1})




%%
figure, hold on
plot(binns,squeeze(mean(b_psycho(:,1,:),1)),'--')
plot(binns,squeeze(mean(b_psycho(:,end,:),1)),'--')

plot(binns,squeeze(mean(p_psycho(:,1,:),1)),'m-')

plot(binns,squeeze(mean(p_psycho(:,end,:),1)),'g-')
%%
subplot(3,2,1), hold on
plot(binns,squeeze(b_chrono(:,1,:)),'--','LineWidth',1,'Color',[0.6 0.6 0.6])
plot(binns,squeeze(p_chrono(:,1,:)),'-','LineWidth',1,'Color',[0 0 0])

subplot(3,2,2), hold on
plot(binns,squeeze(b_trialc(:,1,:)),'--','LineWidth',1,'Color',[0.6 0.6 0.6])
plot(binns,squeeze(p_trialc(:,1,:)),'-','LineWidth',1,'Color',[0 0 0])

subplot(3,2,3), hold on
plot(binns,squeeze(b_chrono(:,2,:)),'--','LineWidth',1,'Color',[0.6 0.6 0.6])
plot(binns,squeeze(p_chrono(:,2,:)),'-','LineWidth',1,'Color',[0 0 0])

subplot(3,2,4), hold on
plot(binns,squeeze(b_trialc(:,2,:)),'--','LineWidth',1,'Color',[0.6 0.6 0.6])
plot(binns,squeeze(p_trialc(:,2,:)),'-','LineWidth',1,'Color',[0 0 0])

subplot(3,2,5), hold on
plot(binns,squeeze(b_chrono(:,3,:)),'--','LineWidth',1,'Color',[0.6 0.6 0.6])
plot(binns,squeeze(p_chrono(:,3,:)),'-','LineWidth',1,'Color',[0 0 0])

subplot(3,2,6), hold on
plot(binns,squeeze(b_trialc(:,3,:)),'--','LineWidth',1,'Color',[0.6 0.6 0.6])
plot(binns,squeeze(p_trialc(:,3,:)),'-','LineWidth',1,'Color',[0 0 0])


%%

figure
violin(drt); legend off
set(gca,'XTick',1:length(p_mag),'XTickLabel',num2str_cell(p_mag,2))
box off
xlabel('Inh inact. mag.')
ylabel('\Delta RT')
%%
figure
violin(dnt); legend off
set(gca,'XTick',1:length(p_mag),'XTickLabel',num2str_cell(p_mag,2))
box off
xlabel('Inh inact. mag.')
ylabel('\Delta trial comp')
%%
figure
violin(dperf); legend off
set(gca,'XTick',1:length(p_mag),'XTickLabel',num2str_cell(p_mag,2))
box off
xlabel('Inh inact. mag.')
ylabel('\Delta Perf.')
%%
figure
violin(dacc); legend off
set(gca,'XTick',1:length(p_mag),'XTickLabel',num2str_cell(p_mag,2))
box off
xlabel('Inh inact. mag.')
ylabel('\Delta Acc.')

%%

for ii=1:length(p_mag)
    plot(drt(:,ii),dnt(:,ii),'.','MarkerSize',8), hold on
end

%%
subplot(4,1,1), hold on
plot(drt,'.k','MarkerSize',8)
plot([0 length(f)],[mean(drt) mean(drt)],'k-','LineWidth',2)
plot([0 length(f)],[mean(drt)-std(drt) mean(drt)-std(drt)],'k--','LineWidth',2)
plot([0 length(f)],[mean(drt)+std(drt) mean(drt)+std(drt)],'k--','LineWidth',2)
subplot(4,1,2), hold on
plot(dnt,'.k','MarkerSize',8)
plot([0 length(f)],[mean(dnt) mean(dnt)],'k-','LineWidth',2)
plot([0 length(f)],[mean(dnt)-std(dnt) mean(dnt)-std(dnt)],'k--','LineWidth',2)
plot([0 length(f)],[mean(dnt)+std(dnt) mean(dnt)+std(dnt)],'k--','LineWidth',2)
subplot(4,1,3), hold on
plot(dperf,'.k','MarkerSize',8)
plot([0 length(f)],[mean(dperf) mean(dperf)],'k-','LineWidth',2)
plot([0 length(f)],[mean(dperf)-std(dperf) mean(dperf)-std(dperf)],'k--','LineWidth',2)
plot([0 length(f)],[mean(dperf)+std(dperf) mean(dperf)+std(dperf)],'k--','LineWidth',2)
subplot(4,1,4), hold on
plot(dacc,'.k','MarkerSize',8)
plot([0 length(f)],[mean(dacc) mean(dacc)],'k-','LineWidth',2)
plot([0 length(f)],[mean(dacc)-std(dacc) mean(dacc)-std(dacc)],'k--','LineWidth',2)
plot([0 length(f)],[mean(dacc)+std(dacc) mean(dacc)+std(dacc)],'k--','LineWidth',2)

%%
ff = load('net_selXfilename.mat','f');
load('net_selXfilename.mat', 'sels_ee', 'sels_ei', 'sels_ie', 'sels_ii')
map = zeros(length(f),1);% which ff is f
for ii = 1:length(ff.f)
    for jj = 1:length(f)
        if strcmp(f{jj},ff.f{ii})
            map(ii) = jj;
            break
        end
    end
end

plot(sels_ei(map).*sels_ie(map),dnt(:,1),'.','MarkerSize',12)
%%

p_sel = -0.25;

pstr = num2str(p_sel);
if contains(pstr,'.')
else
    pstr = [pstr,'.0'];
end

load([f{oo},'/perturb_E_0.0_I_',pstr,'.mat'])

uu = find(trails_coh==20);

% trl = 2100;
trl = uu(1);

subplot(3,1,1),hold on

dp = squeeze(perturb_out(trl,:,:));
db = squeeze(baseline_out(trl,:,:));

mx = max([max(dp(:)),max(db(:))]);
mn = min([min(dp(:)),min(db(:))]);

plot([trails_starts(trl) trails_starts(trl)],[mn mx],'--','Color',[0.7 0.7 0.7],'LineWidth',2)
plot([trails_ends(trl) trails_ends(trl)],[mn mx],'--','Color',[0.7 0.7 0.7],'LineWidth',2)


plot(squeeze(perturb_out(trl,:,1)),'b','LineWidth',2)
plot(squeeze(perturb_out(trl,:,2)),'r','LineWidth',2)
plot(squeeze(baseline_out(trl,:,1)),'b--','LineWidth',2), hold on
plot(squeeze(baseline_out(trl,:,2)),'r--','LineWidth',2)



subplot(3,1,2),hold on

dp = mean(squeeze(perturb_acte(trl,:,:)),2);
db = mean(squeeze(baseline_acte(trl,:,:)),2);

mx = max([max(dp(:)),max(db(:))]);
mn = min([min(dp(:)),min(db(:))]);

plot([trails_starts(trl) trails_starts(trl)],[mn mx],'--','Color',[0.7 0.7 0.7],'LineWidth',2)
plot([trails_ends(trl) trails_ends(trl)],[mn mx],'--','Color',[0.7 0.7 0.7],'LineWidth',2)


plot(mean(squeeze(perturb_acte(trl,:,:)),2),'k','LineWidth',2),hold on
plot(mean(squeeze(baseline_acti(trl,:,:)),2),'k--','LineWidth',2)


subplot(3,1,3),hold on

dp = mean(squeeze(perturb_acti(trl,:,:)),2);
db = mean(squeeze(baseline_acti(trl,:,:)),2);

mx = max([max(dp(:)),max(db(:))]);
mn = min([min(dp(:)),min(db(:))]);

plot([trails_starts(trl) trails_starts(trl)],[mn mx],'--','Color',[0.7 0.7 0.7],'LineWidth',2)
plot([trails_ends(trl) trails_ends(trl)],[mn mx],'--','Color',[0.7 0.7 0.7],'LineWidth',2)


plot(mean(squeeze(perturb_acti(trl,:,:)),2),'k','LineWidth',2),hold on
plot(mean(squeeze(baseline_acte(trl,:,:)),2),'k--','LineWidth',2)

%%
ff = load('net_selXfilename.mat');

map = zeros(length(ff.f),1);
for ii = 1:length(f)
    for jj = 1:length(ff.f)
        if strcmp(f{ii},ff.f{jj})
            map(jj) = ii;
            break
        end
    end
end

plot(ff.sels_ei(map).*ff.sels_ie(map),drt,'.','MarkerSize',14)
xlabel('\Sigma^{EI}\Sigma^{IE}'), ylabel('\Delta RT')