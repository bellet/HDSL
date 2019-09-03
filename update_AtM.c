/*
Copyright 2015 Kuan Liu & Aurelien Bellet

This file is part of HDSL.

HDSL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

HDSL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with HDSL.  If not, see <http://www.gnu.org/licenses/>.

#include "mex.h"
#include <stdint.h>
#include <math.h>
*/

#include "mex.h"
#include <stdint.h>

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define SHL( a ) ( ((a) >= 1) ? 0 : ( ((a) <= 0) ? 0.5 - (a) : 0.5*(1-(a))*(1-(a)) ) ) /* computes smoothed hinge loss*/
/*
get value corresponding to element (i,j) in sparse matrix
i,j start at 1
based on binary search*/
double getValue (mwIndex *Ir, mwIndex *Jc, double *Pr, int i, int j) {

	int k, left, right, mid, cas;
	left = Jc[j-1];
	right = Jc[j]-1;
	while (left <= right) {
		mid = left + (right-left)/2;
		if (Ir[mid]+1 == i)
			return Pr[mid];
		else if (Ir[mid]+1 > i)
			right = mid - 1;
		else
			left = mid + 1;
	}

	return 0.0;

}
/*
input 1: AtM
input 2: Cons
input 3: data (only 2 features, in sparse format)
input 4: signF
input 5: alpha
input 6: scale
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	int numCons, i, xIdx, yIdx, zIdx;
	uint64_t *Cons;
	double *AtM, *AtMNew, *dataPr, *violation, *losses;
	mwIndex *dataJc, *dataIr;
	double signF, alpha, scale, x1, x2, yz1, yz2, tmp;
	
	if (nrhs != 6)
		mexErrMsgTxt("Wrong number of input arguments.");
	if (nlhs > 3)
		mexErrMsgTxt("Too many output arguments.");
	if (!mxIsSparse(prhs[2]))
		mexErrMsgTxt("Data must be in sparse format.");

	AtM = mxGetPr(prhs[0]);
	Cons = (uint64_t *)mxGetPr(prhs[1]);
	numCons = mxGetN(prhs[1]);
	dataPr = mxGetPr(prhs[2]);
	dataJc = mxGetJc(prhs[2]);
	dataIr = mxGetIr(prhs[2]);
	signF = mxGetScalar(prhs[3]);
	alpha = mxGetScalar(prhs[4]);
	scale = mxGetScalar(prhs[5]);
	
	plhs[0] = mxCreateDoubleMatrix(numCons, 1, mxREAL);
	AtMNew = (double *) mxGetData(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(numCons, 1, mxREAL);
	violation = (double *) mxGetData(plhs[1]);
	plhs[2] = mxCreateDoubleMatrix(numCons, 1, mxREAL);
	losses = (double *) mxGetData(plhs[2]);
	
	for (i=0; i<numCons; i++) {
		xIdx = Cons[i*3];
		yIdx = Cons[i*3+1];
		zIdx = Cons[i*3+2];
		x1 = getValue(dataIr,dataJc,dataPr,xIdx,1);
		x2 = getValue(dataIr,dataJc,dataPr,xIdx,2);
		yz1 = getValue(dataIr,dataJc,dataPr,yIdx,1) - getValue(dataIr,dataJc,dataPr,zIdx,1);
		yz2 = getValue(dataIr,dataJc,dataPr,yIdx,2) - getValue(dataIr,dataJc,dataPr,zIdx,2);
		tmp = x1*yz1 + x2*yz2;
		if (signF > 0)
			tmp = tmp + x1*yz2 + x2*yz1;
		else
			tmp = tmp - x1*yz2 - x2*yz1;
		AtMNew[i] = (1-alpha)*AtM[i] + scale*alpha*tmp;
		violation[i] = max(0,1-AtMNew[i]);
		losses[i] = SHL(AtMNew[i]);
	}
		
	return;
	
}
