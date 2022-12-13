 
nu0I = nu0Ib*ones(2,3);
[alpha1,alpha2,I0E1,I0E2,Tnmda,Tampa,alpha,a1_ih,a2_ih,I0I1,I0I2] = ...
    gen_alphas(w_e,w_i,see,sei,sie,nuext,model,f,nu0I(1,:));

[nl1,nl2,c] = get_nulls_fps2(model,mu,ch,see,sei,sie,nuext,f,nu0I(1,:),eps,do_nulls,0);
cc = c(c(:,1)==c(:,2),1:2);


t_int = 10000:-1:0;
bk1_traj = zeros(length(t_int),2);
fd1_traj = zeros(length(t_int),2);
bk2_traj = zeros(length(t_int),2);
fd2_traj = zeros(length(t_int),2);
dev = 1e-3;
st_pt1 = [cc(1)-dev*cc(1) cc(2)-dev*cc(2)];
st_pt2 = [cc(1)-dev*cc(1) cc(2)+dev*cc(2)];
st_pt3 = [cc(1)+dev*cc(1) cc(2)+dev*cc(2)];
st_pt4 = [cc(1)+dev*cc(1) cc(2)-dev*cc(2)];
bk1_traj(1,:) = st_pt1;
fd1_traj(1,:) = st_pt2;
bk2_traj(1,:) = st_pt3;
fd2_traj(1,:) = st_pt4;

st_flag = zeros(4,1);

for uu=2:length(t_int)
    if st_flag(1) == 0
        F = dsdt_noAMPA(bk1_traj(uu-1,1),bk1_traj(uu-1,2),alpha1,alpha2,I0E1,I0E1,40,0,100,0.641);
        bk1_traj(uu,1) = bk1_traj(uu-1,1) - F(1);
        bk1_traj(uu,2) = bk1_traj(uu-1,2) - F(2);
        if bk1_traj(uu,1) <= 0 || bk1_traj(uu,2) <= 0 || bk1_traj(uu,1) >= 1 || bk1_traj(uu,2) >= 1
            st_flag(1) = uu;
        end
    end
    if st_flag(2) == 0
        F = dsdt_noAMPA(fd1_traj(uu-1,1),fd1_traj(uu-1,2),alpha1,alpha2,I0E1,I0E1,40,0,100,0.641);
        fd1_traj(uu,1) = fd1_traj(uu-1,1) + F(1);
        fd1_traj(uu,2) = fd1_traj(uu-1,2) + F(2);
        if fd1_traj(uu,1) <= 0 || fd1_traj(uu,2) <= 0 || fd1_traj(uu,1) >= 1 || fd1_traj(uu,2) >= 1
            st_flag(2) = uu;
        end
    end
    
    if st_flag(3) == 0
        F = dsdt_noAMPA(bk2_traj(uu-1,1),bk2_traj(uu-1,2),alpha1,alpha2,I0E1,I0E1,40,0,100,0.641);
        bk2_traj(uu,1) = bk2_traj(uu-1,1) - F(1);
        bk2_traj(uu,2) = bk2_traj(uu-1,2) - F(2);
        if bk2_traj(uu,1) <= 0 || bk2_traj(uu,2) <= 0 || bk2_traj(uu,1) >= 1 || bk2_traj(uu,2) >= 1
            st_flag(3) = uu;
        end
    end
    
    if st_flag(4) == 0
        F = dsdt_noAMPA(fd2_traj(uu-1,1),fd2_traj(uu-1,2),alpha1,alpha2,I0E1,I0E1,40,0,100,0.641);
        fd2_traj(uu,1) = fd2_traj(uu-1,1) + F(1);
        fd2_traj(uu,2) = fd2_traj(uu-1,2) + F(2);
        if fd2_traj(uu,1) <= 0 || fd2_traj(uu,2) <= 0 || fd2_traj(uu,1) >= 1 || fd2_traj(uu,2) >= 1
            st_flag(4) = uu;
        end
    end
end

plot(bk1_traj(:,1),bk1_traj(:,2),'k','LineWidth',2.0), hold on
plot(fd1_traj(:,1),fd1_traj(:,2),'r','LineWidth',2.0)
plot(bk2_traj(:,1),bk2_traj(:,2),'k','LineWidth',2.0), hold on
plot(fd2_traj(:,1),fd2_traj(:,2),'r','LineWidth',2.0)
axis([0 0.75 0. 0.75])