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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    int f1, f2 , nFeat, xIdx, yzIdx, cIdx, s1,s2,e1,e2,j1,j2,*constraint;
    uint64_t *f_best,*f_over, *idxF,tmpInt;
    int debug = 0;
    
    mwSize xNnz, yzNnz;
    mwIndex *xJc, *xIr, *yzJc, *yzIr, *cJc, *cIr;
    double *xPr,*yzPr,*cPr,*AtM, * basis, * basisSign, *coef;
    double value, sign, A11, A12, A21, A22,weight,grad;
    double x1,x2, yz1,yz2, *signF, *best_value, *M_grad, *score;
    int nCons,i,opt_i, opt_i2,j,k,count=0,n_pairs = 0,nR,nC, c=0;
    
    if (nrhs != 9)
        mexErrMsgTxt("Wrong number of input arguments.");
    if (nlhs > 9)
        mexErrMsgTxt("Too many output arguments.");
    if (!(mxIsSparse(prhs[0]) && mxIsSparse(prhs[1])))
        mexErrMsgTxt("Arguments must be in sparse format.");
    if (!(mxIsDouble(prhs[0]) && mxIsDouble(prhs[1])))
        mexErrMsgTxt("Arguments must be double numbers.");
    
    xPr = mxGetPr(prhs[0]);
    xJc = mxGetJc(prhs[0]);
    xIr = mxGetIr(prhs[0]);
    
    yzPr = mxGetPr(prhs[1]);
    yzJc = mxGetJc(prhs[1]);
    yzIr = mxGetIr(prhs[1]);
    
    cPr = mxGetPr(prhs[2]);
    cJc = mxGetJc(prhs[2]);
    cIr = mxGetIr(prhs[2]);
    
    basis = mxGetPr(prhs[3]);
    basisSign = mxGetPr(prhs[4]);
    
    AtM = mxGetPr(prhs[5]);
    coef = mxGetPr(prhs[6]);
    constraint = (int*) mxGetData(prhs[7]);
    
    debug = (int) mxGetScalar(prhs[8]);
    
    nFeat = mxGetM(prhs[4]);
    nR = mxGetM(prhs[3]);
    nC = mxGetN(prhs[3]);
    if (nR != 2)
        mexErrMsgTxt("number of rows of basis vector should be 2");
    if (nC != nFeat)
        mexErrMsgTxt("number of columns of basis vector should be nFeat");
    
    
    plhs[0] = mxCreateNumericMatrix(1, 4, mxUINT64_CLASS, mxREAL);
    f_best = (uint64_t*) mxGetData(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(1, 2, mxREAL);
    signF = (double*) mxGetData(plhs[1]);
    plhs[2] = mxCreateDoubleMatrix(1, 2, mxREAL);
    best_value = (double*) mxGetData(plhs[2]);
    plhs[3] = mxCreateDoubleMatrix(1, 1, mxREAL);
    M_grad = (double*) mxGetData(plhs[3]);
    plhs[4] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    idxF = (uint64_t*) mxGetData(plhs[4]);    
    plhs[5] = mxCreateDoubleMatrix(1, nFeat, mxREAL);
    score = (double*) mxGetData(plhs[5]);
    
    /* necessary? */
    for(i=0; i<nFeat; i++){
        score[i] = 0;
    }
    M_grad[0] = 0;
    
    for (i = 0; i < nFeat; i++){
        f1 = basis[i*2];
        f2 = basis[i*2+1];
        sign = basisSign[i];
        for (cIdx = cJc[i]; cIdx < cJc[i+1]; cIdx++){
             
            c = cIr[cIdx]; /* cons # */
            if (AtM[c] >= 1)
                continue;
            else if (AtM[c] < 0)
                weight = 1;
            else
                weight = 1 - AtM[c];

            k = constraint[c];  /* should be c or c+1. double check*/

            if (k==0)
                mexErrMsgTxt("constraints index wrong..");
            
            x1  = getValue(xIr,xJc,xPr,f1,k);
            x2  = getValue(xIr,xJc,xPr,f2,k);
            yz1 = getValue(yzIr,yzJc,yzPr,f1,k);
            yz2 = getValue(yzIr,yzJc,yzPr,f2,k);
            A11 = x1 * yz1;
            A12 = x1 * yz2;
            A21 = x2 * yz1;
            A22 = x2 * yz2;
            
            grad= weight * ((A11+A22) + sign * (A12+A21));
            if (grad == 0)
                mexErrMsgTxt("constraints index wrong..");
            
            score[i] += grad;
            M_grad[0]  += grad * coef[i];
        }
    }
    
    
    /* loop the score arrays
     * locate the minimum/maximum one
     *
     * could be absorbed into for loop above TODO
     */
    best_value[0] = 0;
    best_value[1] = 0;
    for (i = 0; i< nFeat; i++){
        value = score[i];
        score[i] = value;
        if (i == 0 || value < best_value[0]){
            best_value[0] = value;
            opt_i = i;
        }
        if (i == 0 || value > best_value[1]){
            best_value[1] = value;
            opt_i2 = i;
        }
    }
    f_best[0] = basis[2 * opt_i];
    f_best[1] = basis[2 * opt_i +1];
    signF[0] = basisSign[opt_i];
    idxF[0] = opt_i + 1;
    if (f_best[0] > f_best[1])
    {
        tmpInt = f_best[0];
        f_best[0] = f_best[1];
        f_best[1] = tmpInt;
    }
    
    
    f_best[2] = basis[2 * opt_i2];
    f_best[3] = basis[2 * opt_i2 + 1];
    signF[1] = basisSign[opt_i2];
    
    if (f_best[2] > f_best[3])
    {
        tmpInt = f_best[2];
        f_best[2] = f_best[3];
        f_best[3] = tmpInt;
    }
    
    
    return;
}
