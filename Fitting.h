// [9/18/2015 Yinlin.Hu]

#ifndef _FITTING_H
#define _FITTING_H

#include "Matrix.h"

#ifdef USE_LAPACK
#define HAVE_LAPACK_CONFIG_H
#define LAPACK_COMPLEX_STRUCTURE
#include <lapacke.h>
#endif

template <class T>
Vector<float> ParaboloidFitting1(const Vector<T> x, const Vector<T> y, const Vector<T> z, int ptCnt = 0);
template <class T>
Vector<float> ParaboloidFitting(const Vector<T> x, const Vector<T> y, const Vector<T> z, int ptCnt = 0);
template <class T>
Vector<float> AffineFitting(const Vector<T> pt1, const Vector<T> pt2, const Vector<T> weight, int ptCnt = 0);
template <class T>
Vector<float> AffineFitting_RANSAC(const Vector<T> pt1, const Vector<T> pt2, const Vector<T> weight, int ptCnt = 0);

//
template <class T>
Vector<float> ParaboloidFitting1(const Vector<T> x, const Vector<T> y, const Vector<T> z, int ptCnt/* = 0*/)
{
	// (x^2, y^2, xy, x, y, 1)
	Vector<float> result(6);

	int cnt = ptCnt;
	if(cnt == 0)
		cnt = x.dim();

	if(y.dim() < cnt || z.dim() < cnt){
		printf("x,y,z have not enough data!\n");
		return result;
	}

	// construct A
	Matrix<float> A(cnt, 6);
	Vector<float> b(cnt);
	float* aData = A.data();
	for(int i=0; i<cnt; i++){
		aData[i*6 + 0] = x[i]*x[i];
		aData[i*6 + 1] = y[i]*y[i];
		aData[i*6 + 2] = x[i]*y[i];
		aData[i*6 + 3] = x[i];
		aData[i*6 + 4] = y[i];
		aData[i*6 + 5] = 1;
		b[i] = z[i];
	}

	// solve linear system
	A.SolveLinearSystem(result, b);

	return result;
}

void TestParaboloidFitting1()
{
	// the result should be "0.564 0.130 -0.094 -17.450 0.456 181.617"
	Vector<float> x(20),y(20),z(20);
	Vector<float> resultCoe(6);

	x[ 0] = 11.92;  y[ 0] =  9.17; z[ 0] = 56.32;
	x[ 1] = 12.46;  y[ 1] =  7.13; z[ 1] = 53.28;
	x[ 2] = 14.45;  y[ 2] =  5.91; z[ 2] = 49.18;
	x[ 3] = 15.29;  y[ 3] =  4.84; z[ 3] = 45.34;
	x[ 4] = 17.00;  y[ 4] =  4.87; z[ 4] = 46.15;
	x[ 5] = 16.98;  y[ 5] =  7.52; z[ 5] = 44.59;
	x[ 6] = 16.32;  y[ 6] =  9.64; z[ 6] = 41.70;
	x[ 7] = 14.13;  y[ 7] =  9.31; z[ 7] = 48.42;
	x[ 8] = 17.74;  y[ 8] = 10.02; z[ 8] = 50.29;
	x[ 9] = 18.10;  y[ 9] = 11.72; z[ 9] = 60.76;
	x[10] = 17.25;  y[10] = 16.54; z[10] = 65.30;
	x[11] = 16.76;  y[11] = 13.77; z[11] = 53.03;
	x[12] = 15.77;  y[12] = 14.29; z[12] = 55.27;
	x[13] = 13.89;  y[13] = 13.62; z[13] = 60.18;
	x[14] = 14.09;  y[14] = 11.57; z[14] = 63.02;
	x[15] = 11.84;  y[15] = 12.14; z[15] = 70.31;
	x[16] =  9.94;  y[16] = 12.80; z[16] = 75.87;
	x[17] =  9.55;  y[17] = 10.90; z[17] = 78.00;
	x[18] = 10.90;  y[18] = 10.08; z[18] = 68.19;
	x[19] = 10.57;  y[19] =  9.15; z[19] = 63.14;

// 	for (int i = 0; i < 20; i++){
// 		printf("%.2f %.2f %.2f\n", x[i], y[i], z[i]);
// 	}

	resultCoe = ParaboloidFitting1(x,y,z);
// 	for(int i=0; i<6; i++){
// 		printf("%.3f ", resultCoe[i]);
// 	}
}

//
template <class T>
Vector<float> ParaboloidFitting(const Vector<T> x, const Vector<T> y, const Vector<T> z, int ptCnt/* = 0*/)
{
	// (x^2, y^2, x, y, 1)
	Vector<float> result(5);

	int cnt = ptCnt;
	if (cnt == 0)
		cnt = x.dim();

	if (y.dim() < cnt || z.dim() < cnt){
		printf("x,y,z have not enough data!\n");
		return result;
	}

	// construct A
	Matrix<float> A(cnt, 5);
	Vector<float> b(cnt);
	float* aData = A.data();

#ifdef USE_LAPACK
	for (int i = 0; i < cnt; i++){
		aData[i + 0 * cnt] = x[i] * x[i];
		aData[i + 1 * cnt] = y[i] * y[i];
		aData[i + 2 * cnt] = x[i];
		aData[i + 3 * cnt] = y[i];
		aData[i + 4 * cnt] = 1;
		b[i]=z[i];
	}
	LAPACKE_sgels(LAPACK_COL_MAJOR, 'N', cnt, 5, 1, aData, cnt, b.data(), cnt);
	for (int i = 0; i < 5; i++){
		result[i] = b[i];
		//printf("%.3f ", result[i]);
	}
	//printf("\n");
#else
	for (int i = 0; i < cnt; i++){
		aData[i * 5 + 0] = x[i] * x[i];
		aData[i * 5 + 1] = y[i] * y[i];
		aData[i * 5 + 2] = x[i];
		aData[i * 5 + 3] = y[i];
		aData[i * 5 + 4] = 1;
		b[i] = z[i];
	}
	// solve linear system
	A.SolveLinearSystem(result, b);
// 	for (int i = 0; i < 5; i++){
// 		printf("%.3f ", result[i]);
// 	}
// 	printf("\n\n");
#endif

	return result;
}


template <class T>
Vector<float> AffineFitting(const Vector<T> pt1, const Vector<T> pt2, const Vector<T> weight, int ptCnt /*= 0*/)
{
	// m11, m12, b1 
	// m21, m22, b2
	Vector<float> param(6);

	int cnt = ptCnt;
	if (cnt == 0)
		cnt = weight.dim();

	if (pt1.dim() < 2 * cnt || pt2.dim() < 2 * cnt || weight.dim() < cnt){
		printf("not enough data for affine fitting!\n");
		return param;
	}

	// construct A
	Matrix<float> A(2 * cnt, 6);
	Vector<float> b(2 * cnt);
	float* aData = A.data();
#ifdef USE_LAPACK
	for (int i = 0; i < cnt; i++){
		aData[i + 0 * cnt] = x[i] * x[i];
		aData[i + 1 * cnt] = y[i] * y[i];
		aData[i + 2 * cnt] = x[i];
		aData[i + 3 * cnt] = y[i];
		aData[i + 4 * cnt] = 1;
		b[i] = z[i];
	}
	LAPACKE_sgels(LAPACK_COL_MAJOR, 'N', cnt, 5, 1, aData, cnt, b.data(), cnt);
	for (int i = 0; i < 5; i++){
		result[i] = b[i];
		//printf("%.3f ", result[i]);
	}
	//printf("\n");
#else
	for (int i = 0; i < cnt; i++){
		aData[i * 12 + 0] = weight[i] * pt1[2 * i];
		aData[i * 12 + 1] = weight[i] * pt1[2 * i + 1];
		aData[i * 12 + 2] = weight[i] * 1;

		aData[i * 12 + 9] = weight[i] * pt1[2 * i];
		aData[i * 12 + 10] = weight[i] * pt1[2 * i + 1];
		aData[i * 12 + 11] = weight[i] * 1;

		b[2 * i] = weight[i] * pt2[2 * i];
		b[2 * i + 1] = weight[i] * pt2[2 * i + 1];
	}
	// solve linear system
	A.SolveLinearSystem(param, b);
// 	for (int i = 0; i < 6; i++){
// 		printf("%.3f ", param[i]);
// 	}
// 	printf("\n\n");
#endif

	return param;
}


template <class T>
Vector<float> AffineFitting_RANSAC(const Vector<T> pt1, const Vector<T> pt2, const Vector<T> weight, int ptCnt /*= 0*/)
{
	float conf = 0.995; // confidence
	int maxIters = 100;
	int modelPts = 3;

	int minPtCnt = 5 * modelPts;
	if (ptCnt < minPtCnt){
		printf("WARNING: too small points for RANSAC!\n");
	}

	float errTh = 5.;

	// 3 pairs
	T p1[6], p2[6];

	// m11, m12, b1 
	// m21, m22, b2
	Vector<float> bestPara(6);
	int* goodPtIdx = new int[ptCnt];
	int* tmpPtIdx = new int[ptCnt];

	srand(0);
	int nIters = maxIters;
	int maxGoodCnt = -1;
	double minCost = FLT_MAX;

	for (int i = 0; i < nIters; i++)
	{
		int pickTimes = 0;

	PICK_DATA:
		// pick 3 group of points randomly
		for (int k = 0; k < 3; k++){
			int ptIdx = rand() % ptCnt;
			memcpy(p1 + 2 * k, pt1.data() + 2 * ptIdx, 2 * sizeof(T));
			memcpy(p2 + 2 * k, pt2.data() + 2 * ptIdx, 2 * sizeof(T));
		}
		// are the 3 points on the same line ?
		float deter = 0; // determinant
		deter = p1[0] * p1[3] + p1[2] * p1[5] + p1[4] * p1[1]
			- p1[4] * p1[3] - p1[0] * p1[5] - p1[2] * p1[1];
		if (abs(deter) <= FLT_EPSILON){
			pickTimes++;
			if (pickTimes >= ptCnt){ // too many failures
				delete[] tmpPtIdx;
				delete[] goodPtIdx;
				printf("RePick failed.\n");
				return AffineFitting(pt1, pt2, weight, ptCnt);
			}
			//printf("RePick data.\n");
			goto PICK_DATA;
		}

		// estimate the model
		float inv[9];
		inv[0] = (p1[3] - p1[5]) / deter;
		inv[1] = (p1[5] - p1[1]) / deter;
		inv[2] = (p1[1] - p1[3]) / deter;
		inv[3] = (p1[4] - p1[2]) / deter;
		inv[4] = (p1[0] - p1[4]) / deter;
		inv[5] = (p1[2] - p1[0]) / deter;
		inv[6] = (p1[2] * p1[5] - p1[3] * p1[4]) / deter;
		inv[7] = (p1[1] * p1[4] - p1[0] * p1[5]) / deter;
		inv[8] = (p1[0] * p1[3] - p1[1] * p1[2]) / deter;
		float pa[6]; // parameters
		pa[0] = inv[0] * p2[0] + inv[1] * p2[2] + inv[2] * p2[4];
		pa[1] = inv[3] * p2[0] + inv[4] * p2[2] + inv[5] * p2[4];
		pa[2] = inv[6] * p2[0] + inv[7] * p2[2] + inv[8] * p2[4];
		pa[3] = inv[0] * p2[1] + inv[1] * p2[3] + inv[2] * p2[5];
		pa[4] = inv[3] * p2[1] + inv[4] * p2[3] + inv[5] * p2[5];
		pa[5] = inv[6] * p2[1] + inv[7] * p2[3] + inv[8] * p2[5];

		// find inliers
		int goodCnt = 0;
		double cost = 0;
		for (int k = 0; k < ptCnt; k++){
			float x1 = pt1[2 * k];
			float y1 = pt1[2 * k + 1];
			float xp = pa[0] * x1 + pa[1] * y1 + pa[2];
			float yp = pa[3] * x1 + pa[4] * y1 + pa[5];
			if (xp != xp || yp != yp) {// isnan()
				printf("estimate model failed.\n");
				goto PICK_DATA;
			}
			float x2 = pt2[2 * k];
			float y2 = pt2[2 * k + 1];

			float dis = sqrt((xp - x2)*(xp - x2) + (yp - y2)*(yp - y2));
			if (dis < errTh){
				tmpPtIdx[goodCnt++] = k;
				cost += weight[k] * dis;
			}else{
				cost += weight[k] * errTh;
			}
		}

		//if (goodCnt > maxGoodCnt){
		if (cost < minCost){
			minCost = cost;

			maxGoodCnt = goodCnt;
			memcpy(bestPara.data(), pa, 6 * sizeof(float)); // update best models

			// update inliers
			memcpy(goodPtIdx, tmpPtIdx, goodCnt*sizeof(int));

			// update iteration numbers
			float goodRatio = (float)goodCnt / ptCnt;
			float denom = log(1 - pow(goodRatio, modelPts));
			if (denom < 0){
				//nIters = __min(log(1 - conf) / denom + 0.5, maxIters);
			}
		}
	}

	// re-compute the model from all inliers
	if (maxGoodCnt >= minPtCnt){
		Vector<T> goodPt1(maxGoodCnt * 2), goodPt2(maxGoodCnt * 2), goodWt(maxGoodCnt);
		for (int i = 0; i < maxGoodCnt; i++){
			int srcIdx = goodPtIdx[i];
			memcpy(goodPt1.data() + 2 * i, pt1.data() + 2 * srcIdx, 2 * sizeof(T));
			memcpy(goodPt2.data() + 2 * i, pt2.data() + 2 * srcIdx, 2 * sizeof(T));
			goodWt[i] = weight[srcIdx];
		}
		bestPara = AffineFitting(goodPt1, goodPt2, goodWt, maxGoodCnt);
	}else{ // de-generation case
		bestPara = AffineFitting(pt1, pt2, weight, ptCnt);
	}

	//printf("fitting: %d/%d\n", maxGoodCnt, ptCnt);

	delete[] tmpPtIdx;
	delete[] goodPtIdx;
	return bestPara;
}

#endif // _FITTING_H
