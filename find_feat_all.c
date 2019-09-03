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

/* get value corresponding to element (i,j) in sparse matrix
 * i,j start at 1
 * based on binary search
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

/* input 1: x
 * input 2: y-z
 * all in sparse form
 */

/* pick i at random, optimize over j
 * look at all constraints
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	int f1, f2 , d, xIdx, yzIdx,s1,s2,e1,e2,j1,j2;
	uint64_t *f_best,*f_over, tmpInt;
    
	mwSize xNnz, yzNnz;
	mwIndex *xJc, *xIr, *yzJc, *yzIr;
	double *xPr,*yzPr,*AtM;
	double value, A11, A12, A21, A22,weight;
	double x1,x2, yz1,yz2, offdiag,ondiag, *signF, *best_value, *diag, *off_diag, *score;
    int nCons,i,j,k,count=0,n_pairs = 0,*pf1,*pf2;
   
	if (nrhs != 6)
		mexErrMsgTxt("Wrong number of input arguments.");
	if (nlhs > 5)
		mexErrMsgTxt("Too many output arguments.");
	if (!(mxIsSparse(prhs[0]) && mxIsSparse(prhs[1])))
		mexErrMsgTxt("Arguments must be in sparse format.");
	if (!(mxIsDouble(prhs[0]) && mxIsDouble(prhs[1])))
		mexErrMsgTxt("Arguments must be double numbers.");

	xPr = mxGetPr(prhs[0]);
	yzPr = mxGetPr(prhs[1]);
	xJc = mxGetJc(prhs[0]);
	xIr = mxGetIr(prhs[0]);
	yzJc = mxGetJc(prhs[1]);
	yzIr = mxGetIr(prhs[1]);
	yzNnz = yzJc[1];  /* these two are not used */
	xNnz = xJc[1];
    d = (int) mxGetScalar(prhs[2]);	
    AtM = mxGetPr(prhs[3]);
	nCons = (int) mxGetScalar(prhs[4]);
    f1 = (int) mxGetScalar(prhs[5]);

	plhs[0] = mxCreateNumericMatrix(1, 2, mxUINT64_CLASS, mxREAL);
	f_best = (uint64_t*) mxGetData(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	signF = (double*) mxGetData(plhs[1]);
	plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
	best_value = (double*) mxGetData(plhs[2]);
	plhs[3] = mxCreateDoubleMatrix(1, d+1, mxREAL);
    score = (double*) mxGetData(plhs[3]);
    plhs[4] = mxCreateNumericMatrix(1,d+1, mxUINT64_CLASS, mxREAL);
	f_over = (uint64_t*) mxGetData(plhs[4]);
    
    /*pf1  = (int *)calloc(n_pairs, sizeof(int));
	pf2  = (int *)calloc(n_pairs, sizeof(int));*/
	diag = (double *) calloc(d+1,sizeof(double));
    off_diag = (double *) calloc(d+1,sizeof(double));
    
    for (k = 0; k < nCons; k++){
        if (AtM[k] >= 1)
            mexErrMsgTxt("Satisfied constraints should not be chosen.");
        else if (AtM[k] < 0)
            weight = 1;
        else
            weight = 1 - AtM[k];
        
        x1 = getValue(xIr,xJc,xPr,f1,k+1);
        yz1 = getValue(yzIr,yzJc,yzPr,f1,k+1);
        
        if (x1 == 0){ /* then loop only over x */
            for (xIdx = xJc[k]; xIdx < xJc[k+1]; xIdx ++){
                f2  = xIr[xIdx] + 1;
                x2  = xPr[xIdx];
                yz2 = getValue(yzIr,yzJc,yzPr,f2,k+1);
                count ++;
                
                A21 = x2 * yz1;
                A22 = x2 * yz2;
                diag[f2] += A22 * weight; /* A11 = 0 */
                off_diag[f2] += A21 * weight; /* A12 = 0 */
                /*if ( A22 + fabs(A21) > 0)*/
                f_over[f2] ++;
            }
        }
        else  /* loop over the union of x/yz */
        {
            s1 = xJc[k];
            s2 = yzJc[k];
            e1 = xJc[k+1];
            e2 = yzJc[k+1];
            A11 = x1 * yz1;
            while (s1 < e1 || s2 < e2){
                /* compute f2 */
                j1 = xIr[s1] + 1;
                j2 = yzIr[s2] + 1;
                if (j1 == f1){
                    s1++;
                    j1 = xIr[s1] + 1;
                }
                if (j2 == f1){
                    s2++;
                    j2 = yzIr[s2] + 1;
                }
                
                /* compute x2, yz2 */
                if (s1 >= e1 && s2 >= e2)
                    break;
                else if (s1 >= e1 && s2 < e2){
                    f2 = j2; 
                    x2 = 0; yz2 = yzPr[s2];
                    s2++;
                }
                else if (s1 < e1 && s2 >= e2){
                    f2 = j1; 
                    x2 = xPr[s1];   yz2 = 0;
                    s1++;
                }
                else{
                    if (j1 == j2){  /* and != f1*/
                        f2 = j1;
                        x2  = xPr[s1];
                        yz2 = yzPr[s2];
                        s1++;
                        s2++;
                    }
                    else if (j1 < j2){
                        f2 = j1;
                        x2 = xPr[s1]; yz2 = 0;
                        s1++;
                    }
                    else{
                        f2 = j2;
                        x2 = 0; yz2 = yzPr[s2];
                        s2++;
                    }
                }
                
                
               
                /* to be optimized */
                
                A12 = x1 * yz2;
                A21 = x2 * yz1;
                A22 = x2 * yz2;
                diag[f2] += (A11+A22) * weight;
                off_diag[f2] += (A12+A21) * weight;
                /*if (A11+A22 + fabs(A12+A21) > 0)*/
                f_over[f2] ++;
            }
        }
    }
    
 /*   mexPrintf("# of entries visited: %d (# of constraints: %d)\n",count,nCons);*/
    
    /* loop the diag, off_diag arrays */
    best_value[0] = -1;
    for (i = 0; i<= d; i++){
        value = diag[i] + fabs(off_diag[i]);
        score[i] = value;
        if (value > best_value[0]){
            best_value[0] = value;
            f_best[1] = i;
            if (off_diag[i]<0)
                signF[0] = -1.0;
            else
                signF[0] = +1.0;
        }
    }
    f_best[0] = f1;   
    if (f_best[0] > f_best[1])
    {
        tmpInt = f_best[0];
        f_best[0] = f_best[1];
        f_best[1] = tmpInt;
    }
    
    
    free(diag);
    free(off_diag);
	return;

}
