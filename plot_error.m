% Copyright 2015 Kuan Liu & Aurelien Bellet
% 
% This file is part of HDSL.
% 
% HDSL is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% HDSL is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with HDSL.  If not, see <http://www.gnu.org/licenses/>.

function plot_error(para,base,Stat,M)

if ~exist('result','dir')
    mkdir result
end
fontsize = 18;

b1 = base.err1;
b3 = base.err3;
err1 = Stat.err1;
err3 = Stat.err3;
obj = Stat.obj;
act_f = Stat.af;
l = size(err1,1);
q= (0:l-1) * 50;
%%
type = [{'Overall'} {'balanced'}];
for i = 1:length(type)
    t = type{i};
    % 1NN overall
    fig = figure;
    inc = i-1;
    plot(q,ones(l,1) * b1(1+inc),':',q,ones(l,1) * b1(3+inc),'-',q,err1(:,1+inc),'--+',q,err3(:,1+inc),'--+',q,err1(:,3+inc),'--*',q,err3(:,3+inc),'--*','linewidth',2);
    ti = [t ' error'];
    title(ti,'fontsize',fontsize);
    xlabel('iteration #','fontsize',fontsize);
    ylabel('error rate','fontsize',fontsize);
    legend('standard bilinear 1nn train','standard bilinear 1nn valid','HDSL 1nn train','HDSL 3nn train','HDSL 1nn valid','HDSL 3nn valid');
    
    output_name = ['result/' para.filename '_' t '_g' num2str(para.gamma) '_s' num2str(para.seeds)];
    if para.ls == 1
        output_name = [output_name '_ls'];
    end
    saveas(fig,[output_name,'.png']);
%     print([output_name,'.eps'],'-deps');
end

% obj
fig = figure;
plot(q, obj);
ti = 'objective function';
title(ti,'fontsize',fontsize);
xlabel('iteration #','fontsize',fontsize);
ylabel('objective','fontsize',fontsize);
output_name = ['result/' para.filename  '_g' num2str(para.gamma)  '_s' num2str(para.seeds)];
if para.ls == 1
    output_name = [output_name '_ls'];
end
saveas(fig,[output_name,' _obj.png']);
% print([output_name,'_obj.eps'],'-deps');

% # active feature
fig = figure;
plot(q, act_f);
ti = '# of active features';
title(ti,'fontsize',fontsize);
xlabel('iteration #','fontsize',fontsize);
ylabel('# of active features','fontsize',fontsize);
output_name = ['result/' para.filename '_g' num2str(para.gamma)  '_s' num2str(para.seeds)];
if para.ls == 1
    output_name = [output_name '_ls'];
end
saveas(fig,[output_name,' _af.png']);
% print([output_name,'_af.eps'],'-deps');


save(['result/' 'detail_' para.filename '_g' num2str(para.gamma) 'ls' num2str(para.ls) '_s' num2str(para.seeds) '.mat']);