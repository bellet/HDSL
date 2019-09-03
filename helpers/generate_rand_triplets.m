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

 
function Cons = generate_rand_triplets(label, nbCons)

fprintf('Creating %d random triplets\n',nbCons);

n = length(label);
Cons = zeros(3,nbCons);

for i=1:nbCons
	idx1 = randi(n);
	idxPos = find(label == label(idx1));
	idx2 = randsample(idxPos,1);
	idxNeg = find(label ~= label(idx1));
	idx3 = randsample(idxNeg,1);
	Cons(:,i) = [idx1;idx2;idx3];
end
