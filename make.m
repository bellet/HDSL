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

mex -largeArrayDims helpers/mink2.c

mex -largeArrayDims find_feat_all.c

mex -largeArrayDims away.c

mex -largeArrayDims line_search_away.c

mex -largeArrayDims line_search.c

mex -largeArrayDims update_AtM.c