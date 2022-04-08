% DSSC;R3Net;LPS;BASNet;SCRN;CPD;AADF-Net;ITSD;F3Net;GCPANet;MINet;LDF;SAMNet;PIE;Ours
m = load('.\eval\result\DSSC\LISP\prec.mat');
n = load('.\eval\result\DSSC\LISP\rec.mat');
y = m.prec(1,:);
x = n.rec(1,:);
figure(6); 
hold on; 
plot(x ,y,  '--' , 'Color',[94/255 87/255 77/255], 'linewidth', 1);% DSS ��ɫ���� ��
%plot(x ,y,  '--' , 'Color',[0.5 0.5 0.5], 'linewidth', 1); % R3Net ��ɫ���� ��
%plot(x ,y,  '--', 'Color',[137/255 170/255 230/255], 'linewidth', 1 ); % LPS С�к������� 
%plot(x ,y, '--', 'Color',[244/255 213/255 141/255], 'linewidth', 1 ); % BASNet����ɫ ���������� 214/255 210/255 147/255
%plot(x ,y,  '--', 'Color',[158/255 173/255 111/255], 'linewidth', 1 ); % SCRN ����ɫ����  ���ɫ���� [161/255 234/255 171/255]
%plot(x ,y,  '--', 'Color', [187/255 133/255 136/255], 'linewidth', 1 ); % CPD-R ��õ������
%plot(x ,y, '--','Color',[0.3 0.8 0.9] , 'linewidth', 1); % AADF-Net ǳ��ɫ���� ��
%plot(x ,y,  '--','Color',[177/255 128/255 160/255], 'linewidth', 1 ); % ITSD �Ϻ�ɫ����

%plot(x ,y,  'Color',[94/255 87/255 77/255], 'linewidth', 1 ); % F3Net ��ɫ
%plot(x ,y,  'Color',[0.5 0.5 0.5], 'linewidth', 1); %  GCPANet ��ɫ
%plot(x ,y,  'Color',[244/255 213/255 141/255], 'linewidth', 1 ); % MINet ����ɫ
%plot(x ,y,  'Color', [158/255 173/255 111/255], 'linewidth', 1 ); % LDF ����ɫ
%plot(x ,y,  'Color',[0.3 0.8 0.9], 'linewidth', 1 ); %  SAMNetǳ��ɫ
%plot(x ,y,  'Color', [187/255 133/255 136/255], 'linewidth', 1 ); % PIE��õ��
%plot(x ,y,  'r' , 'linewidth', 1.5); % EIGNet(Ours) ��ɫ

%legend({'EIGNet (Ours)','PIENet','SAMNet','LDF','MINet','GCPANet','F3Net','ITSD','AADF-Net','CPD-R','SCRN','BASNet','LPS','R3Net','DSS'},'Location','northwest','NumColumns',5)
%columnlegend(3, cellstr(num2str([1:10]')), 'location','northwest');

xlabel('Recall'); 
ylabel('Precision');
