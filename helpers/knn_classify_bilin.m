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


% to speed things up, could get rid of mode function

function [predTr predTe] = knn_classify_bilin(dataTr,labelTr,dataTe,M,k)

	nTr = size(dataTr,1);
	nTe = size(dataTe,1);
	
	predTr = zeros(1,nTr);
	predTe = zeros(1,nTe);
	
	B = 750;
	for i=1:B:max(nTr,nTe)
		if i <= nTr
			BTr = min(B-1,nTr-i);
			DTr = dataTr*M*dataTr(i:i+BTr,:)';
			[~,NN] = mink2(-full(DTr),k+1);
			NN=NN';
			predTr(i:i+BTr) = mode(labelTr(NN(:,2:k+1)),2);
		end
		if i <= nTe
			BTe = min(B-1,nTe-i);
			DTe = dataTr*M*dataTe(i:i+BTe,:)';
			[~,NN] = mink2(-full(DTe),k);
			NN = NN';
			predTe(i:i+BTe) = mode(labelTr(NN),2);
		end
	end

	predTr = predTr';
	predTe = predTe';	

end

