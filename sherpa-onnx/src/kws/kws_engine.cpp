#include "kws_engine.h"


static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}


int init_rknn(char* model_name, rknn_context* ctx)
{

    // rknn_context   ctx;
    int ret =0;
    int            model_data_size = 0;

    unsigned char* model_data      = load_model(model_name, &model_data_size);

    ret= rknn_init(ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {

        printf("rknn init failed ret=%d\n", ret);
        return -1;
    } 
    free(model_data);
    return 0;
}

void pre_process(vector<vector<float>> input, vector<float> *t1, vector<float>*t2) {

  int batch_dim = input.size();

  vector<float> x_flat;
  for (const auto &batch : input) {
    x_flat.insert(x_flat.end(), batch.begin(), batch.end()); 
  }

  vector<float> x_min(batch_dim);
  vector<float> x_max(batch_dim);
  
  // 获取最小值
  float minValue = *min_element(x_flat.begin(), x_flat.end());
  
  // 获取最大值
  float maxValue = *max_element(x_flat.begin(), x_flat.end());
 
  for(int i =0; i <x_flat.size();i++)
  {
     t1->push_back(x_flat[i]-minValue);
  }
  t2->push_back(maxValue-minValue);
 
}

float post_process(vector<float> bn_out) {

  int size=bn_out.size();
  float* x = bn_out.data();
  for(int i=0; i<size; i++) {
    x[i] = abs(x[i]);
  }
  for(int i=0; i<size; i++) {
    x[i] = pow(x[i], 2);
  }
  float sum = 0;
  for(int i=0; i<size; i++) {
    sum += x[i];
  }
  sum = sqrt(sum);
  sum = max(sum, 9.999f);

  return sum;

}
 

int rknn_process(rknn_context ctx, vector<float>* input_a,vector<float>* input_b,vector<float> * bn_output)
{

 
    rknn_input inputs[2];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = 1 * 9536*sizeof(float);
    inputs[0].fmt = RKNN_TENSOR_NCHW;
    inputs[0].pass_through = 0;
    inputs[0].buf =input_a->data();

    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].size = 1 * 1 *sizeof(float);
    inputs[1].fmt = RKNN_TENSOR_NCHW;
    inputs[1].pass_through = 0;
    inputs[1].buf = input_b->data();

   int rknn_ret=rknn_inputs_set(ctx, 2, inputs);

    if (rknn_ret < 0) {
        printf("rknn input set failed, ret=%d\n", rknn_ret);
        return -1;
    }

    rknn_ret = rknn_run(ctx, NULL);
    if (rknn_ret < 0) 
    {
        printf(" rknn run failed, ret=%d\n", rknn_ret);
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 1; i++)
    {
        outputs[i].want_float = 1;
    }

    rknn_ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (rknn_ret < 0) 
    {
        printf(" rknn output failed, ret=%d\n", rknn_ret);
        return -1;
    }

    memcpy(bn_output->data(),outputs[0].buf,2048*sizeof(float));

    rknn_ret = rknn_outputs_release(ctx, 1, outputs);
    if (rknn_ret < 0) 
    {
        printf(" rknn output release failed, ret=%d\n", rknn_ret);
        return -1;
    }

    return 0;
}

// int main() {
    
  

//     vector<vector<float>> randomVec(149, vector<float>(64));
//     std::ifstream in_file("input.bin", std::ios::binary);
//     float value;
//     for (int i = 0; i < randomVec.size(); ++i) {
//       for (int j = 0; j < randomVec[0].size(); ++j) {
//         in_file.read((char*)&value, sizeof(float));
 
//         randomVec[i][j] = value;
//       } 
//     }
  
 
//     rknn_context ctx=0;
//     char * model_path="eff_word.rknn";
//     int ret = init_rknn(model_path,&ctx);
 
//     for(int i =0 ; i<1;i++)
//     {
//       auto start = chrono::high_resolution_clock::now(); 
//       vector<float> t1;
//       vector<float> t2;
//       vector<float> bn_out(2048);
//       vector<float> final_out(2048);



//       pre_process(randomVec,&t1,&t2);


//       ret = rknn_process(ctx, &t1, &t2,&bn_out);

//       float rk_output=post_process(bn_out);
      
//       for (int i = 0; i < bn_out.size(); i++) 
//       {
//         final_out[i] = bn_out[i]/rk_output; 
//         printf("final_out%f\n",final_out[i]);

//       }
//       auto end = chrono::high_resolution_clock::now();
//       auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//       std::cout << duration.count() << "ms" << std::endl;

//     }

 
//     return 0;
// }