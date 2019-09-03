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
#include <math.h>

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define SHL( a ) ( ((a) >= 1) ? 0 : ( ((a) <= 0) ? 0.5 - (a) : 0.5*(1-(a))*(1-(a)) ) ) /* computes smoothed hinge loss*/
#define SHL_gradls( AtM, AtB, alpha ) ( ( ((1-(alpha))*(AtM)+(alpha)*(AtB)) >= 1 ) ? 0 : ( ( ((1-(alpha))*(AtM)+(alpha)*(AtB)) <= 0 ) ? (AtM)-(AtB) : (1-((1-(alpha))*(AtM)+(alpha)*(AtB)))*((AtM)-(AtB)) ) ) /* computes gradient for line search*/

/* get value corresponding to element (i,j) in sparse matrix
 i,j start at 1
 based on binary search
        */
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

/* input 1: AtM
 input 2: Cons
 input 3: data (only 2 features, in sparse format)
 input 4: signF
 input 5: scale
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	int numCons, i, xIdx, yIdx, zIdx;
	uint64_t *Cons;
	double *AtM, *AtMNew, *dataPr, *best_alpha, *val, *violation, *losses, *relate;
	mwIndex *dataJc, *dataIr;
	double alpha, scale, signF, x1, x2, yz1, yz2, obj, obja, objb, grada, gradb, gradm, a, b, m;
    
	if (nrhs != 5)
		mexErrMsgTxt("Wrong number of input arguments.");
	if (nlhs > 5)
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
	scale = mxGetScalar(prhs[4]);
	
	plhs[0] = mxCreateDoubleMatrix(numCons, 1, mxREAL);
	AtMNew = (double *) mxGetData(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	best_alpha = (double *) mxGetData(plhs[1]);
	plhs[2] = mxCreateDoubleMatrix(numCons, 1, mxREAL);
	violation = (double *) mxGetData(plhs[2]);
	plhs[3] = mxCreateDoubleMatrix(numCons, 1, mxREAL);
	losses = (double *) mxGetData(plhs[3]);    
    plhs[4] = mxCreateDoubleMatrix(numCons, 1, mxREAL);
    relate = (double*) mxGetData(plhs[4]);
   
	val = (double *) malloc(numCons*sizeof(double));

	obja = 0;
	objb = 0;
	grada = 0;
	gradb = 0;
    
	for (i=0; i<numCons; i++) {
		xIdx = Cons[i*3];
		yIdx = Cons[i*3+1];
		zIdx = Cons[i*3+2];
		x1 = getValue(dataIr,dataJc,dataPr,xIdx,1);
		x2 = getValue(dataIr,dataJc,dataPr,xIdx,2);
		yz1 = getValue(dataIr,dataJc,dataPr,yIdx,1) - getValue(dataIr,dataJc,dataPr,zIdx,1);
		yz2 = getValue(dataIr,dataJc,dataPr,yIdx,2) - getValue(dataIr,dataJc,dataPr,zIdx,2);
		val[i] = x1*yz1 + x2*yz2;
		if (signF > 0)
			val[i] += x1*yz2 + x2*yz1;
		else
			val[i] -= x1*yz2 + x2*yz1;
        if (val[i] != 0)
            relate[i] = 1;
		obja += SHL(AtM[i]);
		objb += SHL(scale*val[i]);
		grada += SHL_gradls(AtM[i],scale*val[i],0);
		gradb += SHL_gradls(AtM[i],scale*val[i],1);
	}
	obja /= numCons;
	objb /= numCons;
	
	/*mexPrintf("%g %g %g %g\n",obja,objb,grada,gradb);*/
	
	if (grada*gradb >= 0) { /* if gradients have same sign, then best step in [0,1] is either 0 or 1*/
		if (obja <= objb)
			best_alpha[0] = 0;
		else
			best_alpha[0] = 1;
	} else { /* otherwise we do bisection search*/
		a = 0; b = 1;
		while (fabs(a-b) > 1e-3/scale) {
			m = (a+b)/2;
			/*mexPrintf("inter %g %g %g\n",a,b,m);*/
			gradm = 0;
			for (i=0; i<numCons; i++) {
				gradm += SHL_gradls(AtM[i],scale*val[i],m);
			}
			/*mexPrintf("grad %g %g %g%\n",grada,gradb,gradm);*/
			if (gradm == 0) {
				best_alpha[0] = m;
				break;
			}
			if (gradm*grada < 0) {
				b = m;
				gradb = gradm;
			}
			if (gradm*gradb < 0) {
				a = m;
				grada = gradm;
			}
		}
		/*mexPrintf("alpha %g grad %g%\n",m,gradm);*/
		best_alpha[0] = m;
	}
	
	obj = 0;
	for (i=0; i<numCons; i++) {
		AtMNew[i] = (1-best_alpha[0])*AtM[i] + scale*best_alpha[0]*val[i];
		violation[i] = max(0,1-AtMNew[i]);
		losses[i] = SHL(AtMNew[i]);
		obj += losses[i];
	}
	obj /= numCons;
	/*mexPrintf("alpha %g obj %g%\n",best_alpha[0],obj);*/
		
	return;
	
}
