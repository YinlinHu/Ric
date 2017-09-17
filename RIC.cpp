#include "RIC.h"

#include "SLIC.h" // for over-segmentation
#include "opencv2/optflow.hpp" // for variational refinement
#include "OpticFlowIO.h"

#include "Heap.h"
#include "Fitting.h"
#include "Util.h"

RIC::RIC()
{
	_alpha = 0.7;
	_sp_size = 100;
	_sp_nncnt = 150;
	_model_iter = 4;
	_refine_models = 1;
	_cost_suffix = 0.001;
}

RIC::~RIC()
{

}

void RIC::Interpolate(FImage& img1, FImage& img2, FImage& costMap, FImage& inputMatches, FImage& outU, FImage& outV)
{
	CTimer t, total;

	int w = img1.width();
	int h = img1.height();
	int matchingCnt = inputMatches.height();

	costMap.Add(_cost_suffix);

	// Distance Transform
	FImage matDistanceMap(w, h);
	IntImage matLabels(w, h);
	memset(matLabels.pData, 0xFF, w*h*sizeof(int)); // init to -1;
	matDistanceMap.setValue(1e10);
	for (int i = 0; i < matchingCnt; i++){
		float* p = inputMatches.rowPtr(i);
		float x = p[0];
		float y = p[1];
		int idx = y*w + x;
		//
		matLabels[idx] = i;
		matDistanceMap[idx] = costMap[idx];
	}
	// some components will have the same label, 
	GeodesicDistanceTransform(costMap.pData, matDistanceMap.pData, matLabels.pData, w, h);
	t.toc("Distance Transform: ");

#if 0
	costMap.imshow("cost Map");
	matDistanceMap.imagesc("dis");
	matLabels.imagesc("labels", 0);
#endif

	IntImage matNN; // nearest neighbors of each matching
	FImage matNNDis; // the corresponding distances
	MatchingNeighborConstruction(matDistanceMap, matLabels, matchingCnt, matNN, matNNDis);

	// debug
#if 0
	// show matching cells
	FImage tu(w, h), tv(w, h); // tmp u and v
	for (int i = 0; i < w*h; i++){
		int id = matLabels[i];
		float* p = inputMatches.rowPtr(id);
		float u = p[2] - p[0];
		float v = p[3] - p[1];
		tu[i] = u;
		tv[i] = v;
	}
	OpticFlowIO::ShowFlow("matching cells", tu.pData, tv.pData, w, h);
#endif
	/************************************************************************/
	/*       Construct Superpixel Graph                                     */
	/************************************************************************/

	IntImage spLabels;
	IntImage spNN; // nearest neighbors of each superpixel
	FImage spPos;
	IntImage spItems;

	int spCnt = OverSegmentaion(img1, spLabels, _sp_size);
	SuperpixelNeighborConstruction(spLabels, spCnt, spNN);
	SuperpixelLayoutAnalysis(spLabels, spCnt, spPos, spItems);

	//printf("superpixel count: %d\n", spCnt);
	//spLabels.imshow("sp", 0);

	/************************************************************************/
	/*     Find Support Neighbors                                           */
	/************************************************************************/

	int* srcMatchIds = new int[spCnt];
	for (int i = 0; i < spCnt; i++){
		float* pos = spPos.rowPtr(i);
		int x = pos[0] + 0.5;
		int y = pos[1] + 0.5;
		srcMatchIds[i] = matLabels[y*w + x];
	}

	int supportCnt = _sp_nncnt;
	int* supportMatchIds = new int[spCnt*supportCnt];  // support matches for each superpixel
	float* supportMatchDis = new float[spCnt*supportCnt];

	FindSupportMatches(srcMatchIds, spCnt, supportCnt, matNN, matNNDis, supportMatchIds, supportMatchDis);

	t.toc("Graph Construction: ");

	/************************************************************************/
	/*    Affine fitting                                                    */
	/************************************************************************/
	// affine fit for every seed

	FImage FitModels(6, spCnt); // models for each superpixel

	// propagation
	PropagateModels(spCnt, spNN, supportMatchIds, supportMatchDis, supportCnt, inputMatches, FitModels);

	t.toc("Fitting: ");

	/************************************************************************/
	/* Apply fitting models                                                 */
	/************************************************************************/
	outU.allocate(w, h);
	outV.allocate(w, h);
	for (int i = 0; i < spCnt; i++){
		float* pModel = FitModels.rowPtr(i);
		int* pixelItems = spItems.rowPtr(i);
		int maxPixelCnt = spItems.width();
		for (int k = 0; k < maxPixelCnt; k++){
			int x = pixelItems[2 * k];
			int y = pixelItems[2 * k + 1];
			if (x < 0 || y < 0){
				break;
			}
			float fx = pModel[0] * x + pModel[1] * y + pModel[2];
			float fy = pModel[3] * x + pModel[4] * y + pModel[5];
			int idx = y*w + x;
			outU[idx] = fx - x;
			outV[idx] = fy - y;
			if (OpticFlowIO::unknown_flow(fx - x, fy - y)){
				//printf("%d: <%f, %f>\n", i, fx - x, fy - y);
			}
		}
	}

	t.toc("Apply fitting: ");

	/************************************************************************/
	/* Variational Refinement                                               */
	/************************************************************************/
	VariationalRefine(img1, img2, outU, outV, outU, outV);
	t.toc("Variational: ");
	
	delete[] srcMatchIds;
	delete[] supportMatchIds;
	delete[] supportMatchDis;

	total.toc("RIC total: ");
}

void RIC::SetSuperpixelSize(int spSize)
{
	_sp_size = spSize;
}

void RIC::SetSupportNeighborCount(int nbCnt)
{
	_sp_nncnt = nbCnt;
}

int RIC::OverSegmentaion(FImage& img, IntImage& outLabels, const int spSize)
{
	int w = img.width();
	int h = img.height();
	int numLabels = 0;

	unsigned int* buff = new unsigned int[w*h];
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			float* p = img.pixPtr(i, j);
			unsigned char* pBuf = (unsigned char*)(buff + i*w + j);
			pBuf[3] = p[0] * 255;
			pBuf[2] = p[1] * 255;
			pBuf[1] = p[2] * 255;
			pBuf[0] = 0;
		}
	}

	int* kLabels;
	SLIC slic;
	slic.DoSuperpixelSegmentation_ForGivenSuperpixelSize(buff, w, h, kLabels, numLabels, spSize, 15);

	outLabels.allocate(w, h);
	memcpy(outLabels.pData, kLabels, w*h*sizeof(int));

	delete[] kLabels;
	delete[] buff;

	return numLabels;
}

// use the post-processing from DeepFlow (implemented in OpenCV)
int RIC::VariationalRefine(FImage& img1, FImage& img2, FImage& u, FImage& v, FImage& outu, FImage& outv)
{
	int w = u.width();
	int h = u.height();

	u.MedianFiltering(2);
	v.MedianFiltering(2);

	cv::Ptr<cv::optflow::VariationalRefinement> vf = cv::optflow::createVariationalFlowRefinement();

	// initialization
	cv::Mat cvU(h, w, CV_32FC1);
	cv::Mat cvV(h, w, CV_32FC1);
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			cvU.at<float>(i, j) = u[i*w + j];
			cvV.at<float>(i, j) = v[i*w + j];
		}
	}

	// prepare gray image
	FImage img1g, img2g;
	img1.desaturate(img1g);
	img2.desaturate(img2g);
	cv::Mat cvImg1(h, w, CV_8UC1);
	cv::Mat cvImg2(h, w, CV_8UC1);
	// convert source image to cvImage
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			cvImg1.at<uchar>(i, j) = img1g.pData[i*w + j];
			cvImg2.at<uchar>(i, j) = img2g.pData[i*w + j];
		}
	}

#if 0
	// check the default parameters
	std::cout << vf->getFixedPointIterations() << std::endl;
	std::cout << vf->getSorIterations() << std::endl;
	std::cout << vf->getOmega() << std::endl;
	std::cout << vf->getAlpha() << std::endl;
	std::cout << vf->getDelta() << std::endl;
	std::cout << vf->getGamma() << std::endl;
#endif

// 	vf->setFixedPointIterations(5);
// 	vf->setSorIterations(5);
	vf->setOmega(1.9);	// Relaxation factor in SOR
// 	vf->setAlpha(20);	// smooth term
// 	vf->setDelta(5);	// color term
// 	vf->setGamma(10);	// gradient term

	vf->calcUV(cvImg1, cvImg2, cvU, cvV);
	// 	cv::imshow("cvU_a", cvU);
	// 	cv::imshow("cvV_a", cvV);

	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			u[i*w + j] = cvU.at<float>(i, j);
			v[i*w + j] = cvV.at<float>(i, j);
		}
	}

	return 0;
}

void RIC::Saliency(FImage& image, FImage& saliency)
{
	int w = image.width();
	int h = image.height();
	int ch = image.nchannels();

	// smooth image
	FImage tmpImg;
	image.smoothing(tmpImg, 4);

	// compute derivatives
	FImage dx, dy;
	tmpImg.dx(dx, true);
	tmpImg.dy(dy, true);

	// compute autocorrelation matrix
	FImage dxx(w, h), dxy(w, h), dyy(w, h);
	dxx.setValue(0);
	dxy.setValue(0);
	dyy.setValue(0);
	for (int i = 0; i < w*h; i++){
		for (int k = 0; k < ch; k++){
			dxx[i] += dx[i*ch + k] * dx[i*ch + k];
			dxy[i] += dx[i*ch + k] * dy[i*ch + k];
			dyy[i] += dy[i*ch + k] * dy[i*ch + k];
		}
	}
	dxx.smoothing(4);
	dxy.smoothing(4);
	dyy.smoothing(4);

	// compute smallest eigenvalue
	saliency.allocate(w, h);
	for (int i = 0; i < w*h; i++){
		float tmp = 0.5*(dxx[i] + dyy[i]);
		float f = __max(0, tmp*tmp + dxy[i] * dxy[i] - dxx[i] * dyy[i]);
		saliency[i] = sqrt(__max(0, tmp - sqrt(f)));
	}
}

// Adapted from OpenCV
void RIC::GeodesicDistanceTransform(float* cost, float* dmap, int* labels, int w, int h)
{
	float c1 = 1.0f / 2.0f;
	float c2 = sqrt(2.0f) / 2.0f;
	float d = 0.0f;
	int i, j;
	float *dist_row, *cost_row;
	float *dist_row_prev, *cost_row_prev;
	int *label_row;
	int *label_row_prev;

#define CHECK(cur_dist,cur_label,cur_cost,prev_dist,prev_label,prev_cost,coef)\
		{\
    d = prev_dist + coef*(cur_cost+prev_cost);\
    if(cur_dist>d){\
        cur_dist=d;\
        cur_label = prev_label;}\
		}

	//first pass (left-to-right, top-to-bottom):
	dist_row = dmap;
	label_row = labels;
	cost_row = cost;
	for (j = 1; j < w; j++)
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);

	for (i = 1; i < h; i++)
	{
		dist_row = dmap + i*w;
		dist_row_prev = dmap + (i - 1) * w;

		label_row = labels + i * w;
		label_row_prev = labels + (i - 1)*w;

		cost_row = cost + i * w;
		cost_row_prev = cost + (i - 1) * w;

		j = 0;
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
		j++;
		for (; j < w - 1; j++)
		{
			CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);
			CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
			CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
			CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
		}
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j - 1], label_row[j - 1], cost_row[j - 1], c1);
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
	}

	//second pass (right-to-left, bottom-to-top):
	dist_row = dmap + (h - 1) * w;
	label_row = labels + (h - 1)*w;
	cost_row = cost + (h - 1)*w;
	for (j = w - 2; j >= 0; j--)
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);

	for (i = h - 2; i >= 0; i--)
	{
		dist_row = dmap + i*w;
		dist_row_prev = dmap + (i + 1) * w;

		label_row = labels + i * w;
		label_row_prev = labels + (i + 1)*w;

		cost_row = cost + i * w;
		cost_row_prev = cost + (i + 1) * w;

		j = w - 1;
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
		j--;
		for (; j > 0; j--)
		{
			CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);
			CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
			CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
			CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j - 1], label_row_prev[j - 1], cost_row_prev[j - 1], c2);
		}
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row[j + 1], label_row[j + 1], cost_row[j + 1], c1);
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j + 1], label_row_prev[j + 1], cost_row_prev[j + 1], c2);
		CHECK(dist_row[j], label_row[j], cost_row[j], dist_row_prev[j], label_row_prev[j], cost_row_prev[j], c1);
	}
#undef CHECK
}

float RIC::GetWeightFromDistance(float dis)
{
	return exp(-dis / _alpha);
}

int RIC::HypothesisGeneration(int* matNodes, int matCnt, FImage& inputMatches, float* outModel)
{
	if (matCnt < 3){
		printf("Too small points for Hypothesis Generation.\n");
		return -1;
	}

	int pickTimes = 0;
	int maxPickTimes = 10;
	float p1[6], p2[6]; // 3 pairs
PICK_DATA:
	// pick 3 group of points randomly
	for (int k = 0; k < 3; k++){
		int matId = matNodes[rand() % matCnt];
		float* p = inputMatches.rowPtr(matId);
		p1[2 * k] = p[0];
		p1[2 * k + 1] = p[1];
		p2[2 * k] = p[2];
		p2[2 * k + 1] = p[3];
	}
	// are the 3 points on the same line ?
	float deter = 0; // determinant
	deter = p1[0] * p1[3] + p1[2] * p1[5] + p1[4] * p1[1]
		- p1[4] * p1[3] - p1[0] * p1[5] - p1[2] * p1[1];
	if (abs(deter) <= FLT_EPSILON){
		pickTimes++;
		if (pickTimes > maxPickTimes){ // too many failures
			//printf("RePick failed.\n");
			return -1;
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

	memcpy(outModel, pa, 6 * sizeof(float));

	return 0;
}

float RIC::HypothesisEvaluation(float* inModel, int* matNodes, float* matDis, int matCnt, FImage&inputMatches, int* outInLier)
{
	float errTh = 5.;

	// find inliers
	int inLierCnt = 0;
	double cost = 0;
	for (int k = 0; k < matCnt; k++){
		int matId = matNodes[k];
		float* p = inputMatches.rowPtr(matId);
		float x1 = p[0];
		float y1 = p[1];
		float xp = inModel[0] * x1 + inModel[1] * y1 + inModel[2];
		float yp = inModel[3] * x1 + inModel[4] * y1 + inModel[5];
		float pu = xp - x1, pv = yp - y1;

		float tu = p[2] - p[0];
		float tv = p[3] - p[1];
		float wt = GetWeightFromDistance(matDis[k]);

		if (pu != pu || pv != pv || OpticFlowIO::unknown_flow(tu, tv)) { // isnan()
			outInLier[k] = 0;
			cost += wt*errTh;
			continue;
		}

		float dis = sqrt((tu - pu)*(tu - pu) + (tv - pv)*(tv - pv));
		if (dis < errTh){
			outInLier[k] = 1;
			inLierCnt++;
			cost += wt * dis;
		}
		else{
			outInLier[k] = 0;
			cost += wt * errTh;
		}
	}
	return cost;
}

int RIC::PropagateModels(int spCnt, IntImage& spNN, int* supportMatchIds, float* supportMatchDis, int supportCnt, FImage&inputMatches, FImage& outModels)
{
	int iterCnt = _model_iter;

	srand(0);

	IntImage inLierFlag(supportCnt, spCnt);
	int* tmpInlierFlag = new int[supportCnt];
	float tmpModel[6];

	//init models (translation model)
	float rawModel[6] = { 1, 0, 0, 0, 1, 0 };
	for (int i = 0; i < spCnt; i++){
		int matId = supportMatchIds[i*supportCnt + 0]; // the nearest support matching
		float* p = inputMatches.rowPtr(matId);
		float u = p[2] - p[0];
		float v = p[3] - p[1];

		float* pModel = outModels.rowPtr(i);
		memcpy(pModel, rawModel, 6 * sizeof(float));
#if 0
		pModel[2] = rand() % (2 * MAX_DISPLACEMENT) - MAX_DISPLACEMENT;
		pModel[5] = rand() % (2 * MAX_DISPLACEMENT) - MAX_DISPLACEMENT;
#else
		pModel[2] = u;
		pModel[5] = v;
#endif
	}
	// return 0;

	// prepare data
	float* bestCost = new float[spCnt];
#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(8)
#endif
	for (int i = 0; i < spCnt; i++){
		bestCost[i] = HypothesisEvaluation(
			outModels.rowPtr(i),
			supportMatchIds + i*supportCnt,
			supportMatchDis + i*supportCnt,
			supportCnt,
			inputMatches,
			inLierFlag.rowPtr(i));
	}

	// propation
	int* vFlags = new int[spCnt];
	for (int iter = 0; iter < iterCnt; iter++)
	{
		memset(vFlags, 0, sizeof(int)*spCnt);

		int startPos = 0, endPos = spCnt, step = 1;
		if (iter % 2 == 1){
			startPos = spCnt - 1; endPos = -1; step = -1;
		}

		for (int idx = startPos; idx != endPos; idx += step){
			float* pModel = outModels.rowPtr(idx);
			int* pInLier = inLierFlag.rowPtr(idx);
			int* pSuperpixelNb = spNN.rowPtr(idx);

			if (idx >= 3 * (endPos + startPos) / 4){
				//break;
			}

			// propagate
			int maxNb = spNN.width();
			for (int i = 0; i < maxNb; i++){
				int nb = pSuperpixelNb[i];
				if (nb < 0) break;
				if (!vFlags[nb]) continue;

				float* pNNModel = outModels.rowPtr(nb);
				float cost = HypothesisEvaluation(
					pNNModel,
					supportMatchIds + idx*supportCnt,
					supportMatchDis + idx*supportCnt,
					supportCnt,
					inputMatches,
					tmpInlierFlag);
				if (cost < bestCost[idx]){
					memcpy(pModel, pNNModel, 6 * sizeof(float));
					memcpy(pInLier, tmpInlierFlag, supportCnt * sizeof(int));
					bestCost[idx] = cost;
				}
			}

			// random test
			int testCnt = 1;
			for (int i = 0; i < testCnt; i++){
				if (HypothesisGeneration(supportMatchIds + idx*supportCnt, supportCnt, inputMatches, tmpModel) == 0){
					float cost = HypothesisEvaluation(
						tmpModel,
						supportMatchIds + idx*supportCnt,
						supportMatchDis + idx*supportCnt,
						supportCnt,
						inputMatches,
						tmpInlierFlag);
					if (cost < bestCost[idx]){
						memcpy(pModel, tmpModel, 6 * sizeof(float));
						memcpy(pInLier, tmpInlierFlag, supportCnt * sizeof(int));
						bestCost[idx] = cost;
					}
				}
			}

			vFlags[idx] = 1;
		}
	}

	// refinement
	if (_refine_models){
		int averInlier = 0;
#ifdef WITH_OPENMP
#pragma omp parallel for num_threads(8)
#endif
		int minPtCnt = 15;
		for (int i = 0; i < spCnt; i++){
			Vector<float> pt1(supportCnt * 2), pt2(supportCnt * 2), wt(supportCnt);
			Vector<float> fitModel;

			int* matNodes = supportMatchIds + i*supportCnt;
			float* matDis = supportMatchDis + i*supportCnt;
			float* pModel = outModels.rowPtr(i);
			int* pInLier = inLierFlag.rowPtr(i);

			int inlierCnt = 0;
			for (int k = 0; k < supportCnt; k++){
				if (pInLier[k]){
					int matId = matNodes[k];
					float* p = inputMatches.rowPtr(matId);
					pt1[2 * inlierCnt] = p[0];
					pt1[2 * inlierCnt + 1] = p[1];
					pt2[2 * inlierCnt] = p[2];
					pt2[2 * inlierCnt + 1] = p[3];
					wt[inlierCnt] = GetWeightFromDistance(matDis[k]);
					//wt[inlierCnt] = 1;
					inlierCnt++;
				}
			}
			if (inlierCnt >= minPtCnt){
				fitModel = AffineFitting(pt1, pt2, wt, inlierCnt);
				memcpy(pModel, fitModel.data(), 6 * sizeof(float));
			}
			averInlier += inlierCnt;
		}
		printf("Average Inlier: %d\n", averInlier / spCnt);
	}

	delete[] tmpInlierFlag;
	delete[] bestCost;
	delete[] vFlags;
	return 0;
}

void RIC::SuperpixelNeighborConstruction(IntImage& labels, int labelCnt, IntImage& outNeighbor)
{
	int w = labels.width();
	int h = labels.height();

	// init
	int maxNeighbors = 32;
	outNeighbor.allocate(maxNeighbors, labelCnt); // only support 32 neighbors
	outNeighbor.setValue(-1);

	//
	int* diffPairs = new int[w*h * 4];
	int diffPairCnt = 0;
	int* lab = labels.data();
	for (int i = 1; i < h; i++){
		for (int j = 1; j < w; j++){
			int idx = i*w + j;

			int l0 = lab[idx];
			int l1 = lab[idx - 1];
			int l2 = lab[idx - w];

			if (l0 != l1){
				diffPairs[2 * diffPairCnt] = l0;
				diffPairs[2 * diffPairCnt + 1] = l1;
				diffPairCnt++;
			}

			if (l0 != l2){
				diffPairs[2 * diffPairCnt] = l0;
				diffPairs[2 * diffPairCnt + 1] = l2;
				diffPairCnt++;
			}
		}
	}

	// get neighbors
	for (int i = 0; i < diffPairCnt; i++){
		int a = diffPairs[2 * i];
		int b = diffPairs[2 * i + 1];
		int* nba = outNeighbor.rowPtr(a); // neighbor of node a
		int* nbb = outNeighbor.rowPtr(b); // neighbor of node b
		int k = 0;

		// add to neighbor list of a
		for (k = 0; k < maxNeighbors; k++){
			if (nba[k] < 0){
				break;
			}
			if (nba[k] == b){
				k = -1;
				break;
			}
		}
		if (k >= 0 && k < maxNeighbors){
			nba[k] = b;
		}

		// add to neighbor list of b
		for (k = 0; k < maxNeighbors; k++){
			if (nbb[k] < 0){
				break;
			}
			if (nbb[k] == a){
				k = -1;
				break;
			}
		}
		if (k >= 0 && k < maxNeighbors){
			nbb[k] = a;
		}
	}

	delete[] diffPairs;
}

void RIC::SuperpixelLayoutAnalysis(IntImage& labels, int labelCnt, FImage& outCenterPositions, IntImage& outNodeItemLists)
{
	int w = labels.width();
	int h = labels.height();

	outCenterPositions.allocate(2, labelCnt); // x and y
	outCenterPositions.setValue(0);

	// get center positions of each node
	int* itemCnt = new int[labelCnt];
	int* lab = labels.data();
	memset(itemCnt, 0, sizeof(int)*labelCnt);
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			int id = lab[i*w + j];
			outCenterPositions[2 * id] += j;
			outCenterPositions[2 * id + 1] += i;
			itemCnt[id]++;
		}
	}
	int maxItemCnt = 0;
	for (int i = 0; i < labelCnt; i++){
		if (itemCnt[i] > maxItemCnt){
			maxItemCnt = itemCnt[i];
		}
		if (itemCnt[i] > 0){
			outCenterPositions[2 * i] /= itemCnt[i];
			outCenterPositions[2 * i + 1] /= itemCnt[i];
		}
		else{
			outCenterPositions[2 * i] = -1;
			outCenterPositions[2 * i + 1] = -1;
		}
	}

	// get node item lists
	outNodeItemLists.allocate(maxItemCnt * 2, labelCnt);
	outNodeItemLists.setValue(-1);
	memset(itemCnt, 0, sizeof(int)*labelCnt);
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			int id = lab[i*w + j];
			int* it = outNodeItemLists.rowPtr(id);
			int cnt = itemCnt[id];
			it[2 * cnt] = j;
			it[2 * cnt + 1] = i;
			itemCnt[id]++;
		}
	}

	delete[] itemCnt;
}

// a border between two int (neighbor) is represented as a long
static inline LONG64 Key(int i, int j)
{
	// swap: always i<j
	if (i > j) { int t = i; i = j; j = t; };
	return (LONG64)i + ((LONG64)j << 32);
}
static inline int KeyFirst(LONG64 i) { return int(i); }
static inline int KeySecond(LONG64 i) { return int(i >> 32); }

void RIC::MatchingNeighborConstruction(FImage& disMap, IntImage& labels, int labelCnt, IntImage& outNeighbor, FImage& outNeighborDis)
{
	int w = disMap.width();
	int h = disMap.height();
	assert(labels.width() == w && labels.height() == h);

	// init
	int* neighborCount = new int[labelCnt];
	memset(neighborCount, 0, labelCnt*sizeof(int));

	int maxNeighbors = 32;
	outNeighbor.allocate(maxNeighbors, labelCnt); // only support 32 neighbors
	outNeighbor.setValue(-1);

	outNeighborDis.allocate(maxNeighbors, labelCnt);
	outNeighborDis.setValue(1e6);

	//
	float* dis = disMap.data();
	int* lab = labels.data();
	CHeap<double> H(w*h * 2, true);
	for (int i = 1; i < h; i++){
		for (int j = 1; j < w; j++){
			int idx = i*w + j;

			int l0 = lab[idx];
			int l1 = lab[idx - 1];
			int l2 = lab[idx - w];

			if (l0 != l1){
				LONG64 k = Key(l0, l1);
				double v = dis[idx] + dis[idx - 1];
				H.Push(&v, (double*)&k, 1);
			}

			if (l0 != l2){
				LONG64 k = Key(l0, l2);
				double v = dis[idx] + dis[idx - w];
				H.Push(&v, (double*)&k, 1);
			}
		}
	}

	while (H.Size()){
		LONG64 newK, currK;

		float minDis = H.Top((double*)&currK);
		int l0 = KeyFirst(currK);
		int l1 = KeySecond(currK);

		while (H.Size()){
			float dis = H.Top((double*)&newK);
			if (newK != currK){
				break;
			}
			H.Pop();

			if (dis < minDis){
				minDis = dis;
			}
		}

		//printf("%d, %d, %f\n", l0, l1, minDis);

		// add neighbors, no direction
		if (neighborCount[l0] < maxNeighbors){
			outNeighbor.rowPtr(l0)[neighborCount[l0]] = l1;
			outNeighborDis.rowPtr(l0)[neighborCount[l0]] = minDis;
			neighborCount[l0]++;
		}
		if (neighborCount[l1] < maxNeighbors){
			outNeighbor.rowPtr(l1)[neighborCount[l1]] = l0;
			outNeighborDis.rowPtr(l1)[neighborCount[l1]] = minDis;
			neighborCount[l1]++;
		}

		//
		currK = newK;
	}

	delete neighborCount;
}

void RIC::FindSupportMatches(int* srcIds, int srcCnt, int supportCnt, IntImage& matNN, FImage& matNNDis, int* outSupportIds, float* outSupportDis)
{
	memset(outSupportIds, 0xFF, supportCnt*srcCnt*sizeof(int)); // -1
	memset(outSupportDis, 0xFF, supportCnt*srcCnt*sizeof(float)); // -1

	int allNodeCnt = matNN.height();
	CHeap<double> H(allNodeCnt, true); // min-heap
	float* currDis = new float[allNodeCnt];

	for (int i = 0; i < srcCnt; i++)
	{
		int id = srcIds[i];
		int* pSupportIds = outSupportIds + i*supportCnt;
		float* pSupportDis = outSupportDis + i*supportCnt;

		H.Clear();
		memset(currDis, 0x7F, sizeof(float)*allNodeCnt); // max float

		int validSupportCnt = 0;

		H.Push(id, 0); // min distance
		currDis[id] = 0;

		while (H.Size()){
			double dis;
			int idx = H.Pop(&dis);

			if (dis > currDis[idx]){
				continue;
			}

			pSupportIds[validSupportCnt] = idx;
			pSupportDis[validSupportCnt] = dis;
			validSupportCnt++;
			if (validSupportCnt >= supportCnt){
				break;
			}

			int* nbIds = matNN.rowPtr(idx);
			float* nbDis = matNNDis.rowPtr(idx);
			int maxNb = matNN.width();
			for (int k = 0; k < maxNb; k++){
				int nb = nbIds[k];
				if (nb < 0){
					break;
				}
				float newDis = dis + nbDis[k];
				if (newDis < currDis[nb]){
					H.Push(nb, newDis);
					currDis[nb] = newDis;
				}
			}
		}
	}

	delete[] currDis;
}