#ifndef LFM_H
#define LFM_H

#include "../shared/constants.h"

void  lfmInit(unsigned int seed);              // khoi tao toan bo P, Q
void  lfmInitUserVector(int userId);            // khi getOrCreateUser tra ve user moi
void  lfmInitItemVector(int itemIdx);           // khi menu.txt co dong moi
void  lfmTrainFromHistory(int maxIter);         // batch train tu orderHistory
void  lfmOnlineUpdate(int userId, const int* itemIds,
                      const int* quantities, int count);
float lfmPredictScore(int userId, int itemIdx);
int   lfmGetTopK(int userId, const int* excluded, int exCount,
                 int* resultIdx, float* resultScore, int topK);
float lfmComputeLoss();
bool  lfmSaveModels(const char* pFile, const char* qFile);
bool  lfmLoadModels(const char* pFile, const char* qFile);

#endif
