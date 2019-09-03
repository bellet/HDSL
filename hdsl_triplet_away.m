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

function [M, Stats] = hdsl_triplet_away(data, label, Cons, scale, nbIter, dataVa, labelVa, dataTe, labelTe)

% hyper-parameters
debug = 0;
MINIMUM_ITER = 20;

% 
n_Cons = zeros(nbIter,1);
n_act_feat = zeros(nbIter,1);
n_relate =zeros(nbIter,1);
[~,d] = size(data);
nbCons = size(Cons,2);
Cons = uint64(Cons);

% for speed up: do not involve all 0 features
u = max(data,[],1);  
nzf = find(u~=0);
length(nzf);

% allocate enough space in sparse matrix, maintain the coeficient of selected bases 
M = spalloc(d,d,nbIter*4);
cons_related = spalloc(size(Cons,2),nbIter,nbIter * d);
n_feat = 0;
feat = zeros(2, nbIter);
feat_sign = zeros(nbIter,1);
coef_feat = zeros(nbIter,1);

% initialize with random basis
f = randsample(d,2);
if (f(1) > f(2))
    tt = f(1);
    f(1) = f(2);
    f(2) = tt;
end
direction = 1;
alpha = 1;
signF = randsample([-1 1],1);
M(f,f) = scale;
M(f(1),f(2)) = M(f(1),f(2)) * signF;
M(f(2),f(1)) = M(f(2),f(1)) * signF;

% value of Frobenius inner product between each constraint and the current metric matrix
AtM = zeros(nbCons,1);
[AtM, violation, loss] = update_AtM(AtM,Cons,data(:,f),signF,alpha,scale);
[~, ~, ~, ~, relate] = line_search(AtM,Cons,data(:,f),signF,scale);
n_relate(n_feat+1) = sum(relate);
cons_related(:,n_feat+1) = sparse(relate);

% for evaluations
err3nn = []; % 1: train 2: train balanced 3: valid 4; valid balanced
err1nn = [];
act_fea = [];
obj = [];
direction_record = [];
picked_feat = [];
picked_step = [];
best_err1 = inf;
best_err3 = inf;
update_type = zeros(nbIter,1); % 0: add new 1: add old 2: away
update_type(1) = 0;

features = containers.Map; % maintain whether one basis has been selected.
idxF = []; % selected basis in away step
hit = 0; % successfully picking one pair of feature
delete = 0; % delete one useless basis


% main loop
for i=2:nbIter+1
    
    % update coefficients
    [feat, coef_feat, feat_sign, n_feat, features] = update_feat...
        (direction, alpha, f, signF, n_feat, coef_feat, feat, ...
        feat_sign, features, idxF);
    
    % output
    if mod(i-1,50) == 0 || i == 2
        fprintf('iter %i obj:%f, #0-loss: %d, #correct: %d, #active features: %i\n',i-1,mean(loss),sum(loss==0),sum(violation<1),full(sum(sum(M~=0,2)~=0)));
        [predTr1, predVa1] = knn_classify_bilin(data,label,dataVa,M,1);
        [e1] = print_error(predTr1, label, predVa1, labelVa, '1-NN');
        [predTr3, predVa3] = knn_classify_bilin(data,label,dataVa,M,3);
        [e3] = print_error(predTr3, label, predVa3, labelVa, '3-NN');
  
        err1nn = [err1nn;e1];
        err3nn = [err3nn;e3];            
        if e1(3) < best_err1
            best_err1 = e1(3);
            best_M1 = M;
        end
        if e3(3) < best_err3
            best_err3 = e3(3);
            best_M3 = M;
        end
        
        af = full(sum(sum(M~=0,2)~=0));
        act_fea = [act_fea; af];
        obj = [obj; mean(loss)];
    end
    
    % converge test: if objective does not change or bases does not change.
    if (length(obj) >  MINIMUM_ITER && obj(end) == obj(end-MINIMUM_ITER) && act_fea(end) == act_fea(end-MINIMUM_ITER))
        fprintf('Converge \n');
        break;
    end
       
    % search for constraints with non zero loss
    idxL = find(loss ~= 0);
    constraint = cumsum(loss ~= 0);
            % constraint(loss == 0) = 0;
    constraint = int32(constraint); 
    if length(idxL) < 1
        display('could not find one instance');
        break;
    end
    n_Cons(i) = length(idxL);  
    A = data(Cons(1,idxL),:)';
    B = (data(Cons(2,idxL),:)-data(Cons(3,idxL),:))';
    AtMtmp = AtM(idxL);
    
    while (hit~=1)   
        % search for basis via fast heuristic: run find_feat_all twice
        ii = randsample(nzf,1);
        [f, ~, ~, ~, ~] = find_feat_all(A,B,d,AtMtmp,numel(idxL),ii);
        if f(1) == ii
            jj = f(2);
        else
            jj = f(1);
        end
        [f, signF, value, ~, ~] = find_feat_all(A,B,d,AtMtmp,numel(idxL),jj);
        
        % search away step basis: among all selected basis, take one away
        % or forward step
        [idx_af,~] = find(coef_feat(1:n_feat)~=0); % active features
        n_act_feat(i) = length(idx_af);
        [f_a, signF_a, value_a, Mgrad, idxF, ~] = away(A,B,cons_related(:,idx_af),feat(:,idx_af),feat_sign(idx_af),AtM,coef_feat(idx_af),constraint,debug);
        if idxF ~= 0
            idxF = idx_af(idxF);
        end
        
        % compare the three and decide
        if value_a(2) > value
            value_f = value_a(2);
            f = f_a(3:4);
            signF = signF_a(2);
            update_type(i) = 1;
        else
            value_f = value;
        end
        direction = 1;

        if (value_f + value_a(1) - 2 * Mgrad < 0)
            direction = -1; % away
            f = f_a(1:2);
            signF = signF_a(1);
            update_type(i) = 2;
        end
        direction_record = [direction_record; direction];
        if (value ~= -1)
            hit = 1;
        end
    end
    hit = 0;
  
    % do line-search and update AtM, violation and loss
    % update matrix M
    if direction > 0
        [AtM, alpha, violation, loss, relate] = line_search(AtM,Cons,data(:,f),signF,scale);        
        if alpha == 0
        else
            n_relate(n_feat+1) = sum(relate);
            cons_related(:,n_feat+1) = sparse(relate);
        end     
    elseif direction == -1;     
        [AtM, alpha, violation, loss] = line_search_away(AtM,Cons,data(:,f),signF,scale, coef_feat(idxF),debug);
        if alpha == coef_feat(idxF) / (1 - coef_feat(idxF))
            delete = delete + 1;
        end
    end
    
    picked_feat = [picked_feat; reshape(f,1,2)];
    picked_step = [picked_step; alpha];
    
    M = (1 - alpha * direction) * M;
    M(f(1),f(1)) = M(f(1),f(1)) + scale * alpha * direction ;
    M(f(2),f(2)) = M(f(2),f(2)) + scale * alpha * direction;
    M(f(1),f(2)) = M(f(1),f(2)) + scale * alpha * direction * signF;
    M(f(2),f(1)) = M(f(2),f(1)) + scale * alpha * direction * signF;
    
end


fprintf('\n----------------------------\nResults with HDSL similarity\n----------------------------\n');
fprintf('%i obj:%f, #0-loss: %d, #correct: %d, #active features: %i\n',i-1,mean(loss),sum(loss==0),sum(violation<1),full(sum(sum(M~=0,2)~=0)));
[predTr1, predVa1] = knn_classify_bilin(data,label,dataVa,M,1);
print_error(predTr1, label, predVa1, labelVa, '1-NN');
[predTr3, predVa3] = knn_classify_bilin(data,label,dataVa,M,3);
print_error(predTr3, label, predVa3, labelVa, '3-NN');
	

% Save results
Stats.err1 = err1nn;
Stats.err3 = err3nn;
Stats.af = act_fea;
Stats.obj = obj;
Stats.best_M1 = best_M1;
Stats.best_M3 = best_M3;
Stats.best_e3 = best_err3;
Stats.best_e1 = best_err1;
Stats.feat = feat;
Stats.n_feat = n_feat;
Stats.feat_sign = feat_sign;
Stats.coef_feat = coef_feat;
Stats.dir = direction_record;
Stats.delete = delete;
Stats.update_type = update_type;
Stats.no_cons_related = n_relate;
Stats.picked_feat = picked_feat;
Stats.picked_step = picked_step;

end






% update selected bases and maintain their coefficients 
function [feat, coef_feat, feat_sign, n_feat, features] = update_feat...
    (direction, alpha, f, signF, n_feat, coef_feat, feat, ...
    feat_sign, features, idxF)

if alpha == 0
else
    %write a function
    if f(1) > f(2)
        k1 = num2str(f(2));
        k2 = num2str(f(1));
    else
        k2 = num2str(f(2));
        k1 = num2str(f(1));
    end
    key_f = [k1 ',' k2 ',' num2str(signF)];
    
    if direction == 1
        if features.isKey(key_f)
            coef_feat(1:n_feat) =  coef_feat(1:n_feat) * (1-alpha);
            idx = features(key_f);
            coef_feat(idx) = coef_feat(idx) + alpha;
        else
            coef_feat(1:n_feat) =  coef_feat(1:n_feat) * (1-alpha);
            n_feat = n_feat+1;
            features(key_f) = n_feat;
            coef_feat(n_feat) = alpha;
            feat(:,n_feat) = f';
            feat_sign(n_feat) = signF;
        end
    else
        coef_feat(1:n_feat) = coef_feat(1:n_feat) * (1+alpha);
        coef_feat(idxF) = coef_feat(idxF) - alpha;
    end
end

end
