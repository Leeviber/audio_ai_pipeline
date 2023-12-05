#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#ifdef RK3588

    #include "rknn_api.h"

    using namespace std;

    int init_rknn(char* model_name, rknn_context* ctx);

    void pre_process(vector<vector<float>> input, vector<float> *t1, vector<float>*t2);

    float post_process(vector<float> bn_out);

    int rknn_process(rknn_context ctx, vector<float>* input_a,vector<float>* input_b,vector<float> * bn_output);

#endif
