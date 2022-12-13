
fig1=figure; hold on
time = 0:0.002:5;

mx=ceil(max([max(trial_data.runC.data(1,:)),max(trial_data.runC.data(2,:))])+1);

patch([2 2 4 4 1],[0 mx mx 0 0],[0.8 0.8 0.8],'EdgeColor','none')
plot(time(20:end-20),trial_data.runC.data(1,20:end-20),'b')
plot(time(20:end-20),trial_data.runC.data(2,20:end-20),'r')
axis([0 5 0 mx]), set(gca,'XTick',[0 5]), set(gca,'YTick',[0 mx])
exportgraphics(fig1,'/Users/roach/four_eqn_exactE.pdf','ContentType','vector')

%%
fig1=figure; hold on
mx=ceil(max([max(trial_data.runC.data(3,:)),max(trial_data.runC.data(4,:))])+3);
mn=ceil(min([min(trial_data.runC.data(3,20:end-20)),min(trial_data.runC.data(4,20:end-20))]));

patch([2 2 4 4 1],[mn mx mx mn mn],[0.8 0.8 0.8],'EdgeColor','none')
plot(time(20:end-20),trial_data.runC.data(3,20:end-20),'b')
plot(time(20:end-20),trial_data.runC.data(4,20:end-20),'r')
axis([0 5 mn mx]), set(gca,'XTick',[0 5]), set(gca,'YTick',[mn mx])
exportgraphics(fig1,'/Users/roach/four_eqn_exactI.pdf','ContentType','vector')

%%
fig1=figure; hold on
x = tmpu.stable == 1;
plot3(tmpu.points(x,1),tmpu.points(x,2),tmpu.points(x,3)-tmpu.points(x,4),'s','MarkerEdgeColor','none','MarkerFaceColor',[0,0,0])

x = tmpu.stable == 0;
plot3(tmpu.points(x,1),tmpu.points(x,2),tmpu.points(x,3)-tmpu.points(x,4),'s','MarkerEdgeColor','none','MarkerFaceColor',[0.7,0.7,0.7])
axis([0 0.7 0 0.7 -0.6 0.6])
set(gca,'xtick',[0 0.7]), set(gca,'ytick',[0 0.7]), set(gca,'ztick',[-0.6 0.6])
set(gca,'xgrid','on'),set(gca,'ygrid','on'),set(gca,'zgrid','on')
view([-61 46])
exportgraphics(fig1,'/Users/roach/four_eqn_exactFPU.pdf','ContentType','vector')
%%
fig1=figure; hold on

plot3(run.S(1,:),run.S(2,:),run.S(3,:)-run.S(4,:),'Color',[0.6 0.6 0.6])


x = tmps.stable == 1;
plot3(tmps.points(x,1),tmps.points(x,2),tmps.points(x,3)-tmps.points(x,4),...
    's','MarkerEdgeColor','none','MarkerFaceColor',[0,0,0])

x = tmps.stable == 0;
plot3(tmps.points(x,1),tmps.points(x,2),tmps.points(x,3)-tmps.points(x,4),...
    's','MarkerEdgeColor','none','MarkerFaceColor',[0.7,0.7,0.7])


set(gca,'xtick',[0 0.7]), set(gca,'ytick',[0 0.7]), set(gca,'ztick',[-0.6 0.6])
set(gca,'xgrid','on'),set(gca,'ygrid','on'),set(gca,'zgrid','on')
view([-61 46])
exportgraphics(fig1,'/Users/roach/four_eqn_exactFPS.pdf','ContentType','vector')