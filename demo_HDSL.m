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

function Ex_HDSL()
addpath('helpers');
para.filename = 'dexter_n2';

Gamma = 100;  %[10,100,1000, 10000];
nbIter = 200;
dataset = ['data/' para.filename];
load(dataset)

para.ls = 10;
scaling = 0;

nG = length(Gamma);
nS = 1; % # of random seeds
for j = 1:nG;
    gamma = Gamma(j);
    para.gamma = gamma;
    ErrTr  = zeros(nS,4);
    ErrVal = zeros(nS,4);
    
    for s = 2:nS+1;
        para.seeds = s;
        rand('seed',s);

        % compute NN accuracy using standard bilinear similarity
        fprintf('\n-----------------------------------------\nResults with standard bilinear similarity\n-----------------------------------------\n');
        [predTr1, predVa1] = knn_classify_bilin(dataTr,labelTr,dataVa,speye(size(dataTr,2)),1);
        base.err1 = print_error(predTr1, labelTr, predVa1, labelVa, '1-NN');
        [predTr3, predVa3] = knn_classify_bilin(dataTr,labelTr,dataVa,speye(size(dataTr,2)),3);
        base.err3 = print_error(predTr3, labelTr, predVa3, labelVa, '3-NN');
        fprintf('\n')

        fprintf('------------\nRunning HDSL\n------------\n');
        % creating constraints: a/ random b/ lmnn-style
        % Cons = generate_rand_triplets(labelTr, size(dataTr,1)*15);
        Cons = generate_knn_triplets(dataTr, labelTr, 3, 5);
        
        [M, Stat] = hdsl_triplet_away(dataTr, labelTr, Cons, gamma, nbIter, dataVa, labelVa);
        plot_error(para,base,Stat,M);
        
        ErrTr(s,:)  = [Stat.err1(end,1),Stat.err1(end,2),Stat.err3(end,1),Stat.err3(end,2)];
        ErrVal(s,:) = [Stat.err1(end,3),Stat.err1(end,4),Stat.err3(end,3),Stat.err3(end,4)];
    end   
    mEtr  = mean(ErrTr,1)*100;
    mEval = mean(ErrVal,1)*100;
    
    F_str = ['result/' para.filename '-g' num2str(gamma) 'scale-' num2str(scaling) '.mat'];
    save(F_str,'mEtr','mEval','ErrTr','ErrVal','gamma');
end
