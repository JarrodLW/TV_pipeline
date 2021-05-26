mi=-10
Ma=10
pas=0.5;
[X1,X2]=meshgrid(mi:pas:Ma,mi:pas:Ma);
[d h]=size(X1);

G=zeros(d,d,2);
G(:,:,1)=X1;
G(:,:,2)=X2;

CoulField='g';
CoulGrid='k';
%%
G=zeros(d,d,2);
G(:,:,1)=X1;
G(:,:,2)=X2;

G_vect=zeros(d,d,2);
Scales=3
dx=d;
dy=d;
alpha=1*[1,1, 2,2, 2 ,-0.5, -0.5, -0.5, 0.3,0.3,0.3]
beta=-1
CP=[0,0]
theta=0

figure()
hold on
for t=1:10
    for u1=1:dx
        for u2=1:dy
            a= ([G(u1,u2,1) ;G(u1,u2,2) ] - CP')'*([G(u1,u2,1) ;G(u1,u2,2) ] - CP')/(Scales^2);
            G_vect(u1,u2,1)=exp(-a) * (-alpha(t)*G(u1,u2,2) + beta*G(u1,u2,1));
            G_vect(u1,u2,2)=exp(-a) * (alpha(t)*G(u1,u2,1)  + beta*G(u1,u2,2));
            
        end
    end
    
    G=G+0.1*G_vect;
    theta=theta+0.1*alpha(t)
    %pause
    
    clf;
    hold on
    axis equal
    %set(gca,'YTick',[-10 0 10])
    %set(gca,'XTick',[-10 0 10])
    axis off
    plot(G(:,:,1),G(:,:,2),CoulGrid);
    plot(G(:,:,1)',G(:,:,2)',CoulGrid);
  %  quiver(G(:,:,1),G(:,:,2),G_vect(:,:,1), G_vect(:,:,2),CoulField,'AutoScale','off','MaxHeadSize',10);
    plot([0.5*cos(theta),0],[0.5*sin(theta),0],'-b','linewidth',2)
%plot(0,0,'xr','linewidth',5)
pause
end