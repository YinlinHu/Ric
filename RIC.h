#ifndef _RIC_H_
#define _RIC_H_

#include "Image.h"

class RIC
{
public:
	RIC();
	~RIC();

	void Interpolate(FImage& img1, FImage& img2, FImage& costMap, FImage& inputMatches, FImage& outU, FImage& outV);
	void SetSuperpixelSize(int spSize = 100);
	void SetSupportNeighborCount(int nbCnt = 150);

private:
	int OverSegmentaion(FImage& img, IntImage& outLabels, const int spSize);
	int VariationalRefine(FImage& img1, FImage& img2, FImage& u, FImage& v, FImage& outu, FImage& outv);
	//
	void Saliency(FImage& image, FImage& saliency);
	void GeodesicDistanceTransform(float* cost, float* dmap, int* labels, int w, int h);
	//
	float GetWeightFromDistance(float dis);
	int HypothesisGeneration(int* matNodes, int matCnt, 
		FImage& inputMatches, float* outModel);
	float HypothesisEvaluation(float* inModel, int* matNodes, 
		float* matDis, int matCnt, 
		FImage&inputMatches, int* outInLier);
	int PropagateModels(int spCnt, IntImage& spNN, 
		int* supportMatchIds, float* supportMatchDis, int supportCnt, 
		FImage&inputMatches, FImage& outModels);
	//
	// construct the neighbors of each superpixel
	void SuperpixelNeighborConstruction(IntImage& labels, int labelCnt, IntImage& outNeighbor);
	// get the center and all pixel items of each superpixel
	void SuperpixelLayoutAnalysis(IntImage& labels, int labelCnt,
		FImage& outCenterPositions, IntImage& outNodeItemLists);
	void MatchingNeighborConstruction(
		FImage& disMap, IntImage& labels, int labelCnt,
		IntImage& outNeighbor, FImage& outNeighborDis);
	// find the support neighbors of each source node (Dijkstra algorithm)
	void FindSupportMatches(
		int* srcIds, int srcCnt, int supportCnt,
		IntImage& matNN, FImage& matNNDis,
		int* outSupportIds, float* outSupportDis);

	float _alpha;
	int _sp_size;
	int _sp_nncnt;
	int _model_iter;
	int _refine_models;
	float _cost_suffix;
	float _maxFlow;
};

#endif

