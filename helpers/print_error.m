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

function [error ] = print_error(predTr, labelTr, predTe, labelTe, string)
% error 1: 
	CTr = confusionmat(predTr,labelTr);
	CTe = confusionmat(predTe,labelTe);
	error(1) = 1-sum(diag(CTr))/sum(sum(CTr));
    error(2) = mean(1-(diag(CTr)'./sum(CTr)));
    error(3) = 1-sum(diag(CTe))/sum(sum(CTe));
    error(4) = mean(1-(diag(CTe)'./sum(CTe)));
	fprintf('%s training error: %g (overall), %g (balanced), validation error: %g (overall), %g (balanced)\n',string,error(1),error(2),error(3),error(4));
