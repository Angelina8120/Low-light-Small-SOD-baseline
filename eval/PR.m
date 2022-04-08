% DSSC;R3Net;LPS;BASNet;SCRN;CPD;AADF-Net;ITSD;F3Net;GCPANet;MINet;LDF;SAMNet;PIE;Ours
m = load('.\eval\result\DSSC\LISP\prec.mat');
n = load('.\eval\result\DSSC\LISP\rec.mat');
y = m.prec(1,:);
x = n.rec(1,:);
figure(6); 
hold on; 
plot(x ,y,  '--' , 'Color',[94/255 87/255 77/255], 'linewidth', 1);% DSS 黑色虚线 √
%plot(x ,y,  '--' , 'Color',[0.5 0.5 0.5], 'linewidth', 1); % R3Net 灰色虚线 √
%plot(x ,y,  '--', 'Color',[137/255 170/255 230/255], 'linewidth', 1 ); % LPS 小男孩蓝虚线 
%plot(x ,y, '--', 'Color',[244/255 213/255 141/255], 'linewidth', 1 ); % BASNet茉莉花色 中香槟虚线 214/255 210/255 147/255
%plot(x ,y,  '--', 'Color',[158/255 173/255 111/255], 'linewidth', 1 ); % SCRN 黄绿色虚线  青瓷色虚线 [161/255 234/255 171/255]
%plot(x ,y,  '--', 'Color', [187/255 133/255 136/255], 'linewidth', 1 ); % CPD-R 老玫瑰虚线
%plot(x ,y, '--','Color',[0.3 0.8 0.9] , 'linewidth', 1); % AADF-Net 浅蓝色虚线 √
%plot(x ,y,  '--','Color',[177/255 128/255 160/255], 'linewidth', 1 ); % ITSD 紫红色虚线

%plot(x ,y,  'Color',[94/255 87/255 77/255], 'linewidth', 1 ); % F3Net 黑色
%plot(x ,y,  'Color',[0.5 0.5 0.5], 'linewidth', 1); %  GCPANet 灰色
%plot(x ,y,  'Color',[244/255 213/255 141/255], 'linewidth', 1 ); % MINet 茉莉花色
%plot(x ,y,  'Color', [158/255 173/255 111/255], 'linewidth', 1 ); % LDF 黄绿色
%plot(x ,y,  'Color',[0.3 0.8 0.9], 'linewidth', 1 ); %  SAMNet浅蓝色
%plot(x ,y,  'Color', [187/255 133/255 136/255], 'linewidth', 1 ); % PIE老玫瑰
%plot(x ,y,  'r' , 'linewidth', 1.5); % EIGNet(Ours) 红色

%legend({'EIGNet (Ours)','PIENet','SAMNet','LDF','MINet','GCPANet','F3Net','ITSD','AADF-Net','CPD-R','SCRN','BASNet','LPS','R3Net','DSS'},'Location','northwest','NumColumns',5)
%columnlegend(3, cellstr(num2str([1:10]')), 'location','northwest');

xlabel('Recall'); 
ylabel('Precision');
