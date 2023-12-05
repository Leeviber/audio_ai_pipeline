#ifdef USE_NPU

#include <vector>
#include <iostream>
#include <fstream>

#include "speaker/rknn_speaker_model.h"

namespace wespeaker {

unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
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

unsigned char* load_model(const char* filename, int* model_size)
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

  RknnSpeakerModel::RknnSpeakerModel(const std::string& model_path) {
  
      int model_data_size = 0;

      unsigned char* model_data      = load_model(model_path.c_str(), &model_data_size);
      rknn_ret= rknn_init(&rk_ctx, model_data, model_data_size, 0, NULL);
      if (rknn_ret < 0) {

          printf("rknn init failed ret=%d\n", rknn_ret);
      } 
 
      free(model_data);
  }

  void RknnSpeakerModel::ExtractResnet(const std::vector<std::vector<float>>& feats,std::vector<float>& resnet_out){
      std::vector<float> flattenedFeats;
      flattenedFeats.reserve(198 * 80);

      for (const auto& row : feats) {
          flattenedFeats.insert(flattenedFeats.end(), row.begin(), row.end());
      }

      rknn_input inputs[1];
      memset(inputs, 0, sizeof(inputs));
      inputs[0].index = 0;
      inputs[0].type = RKNN_TENSOR_FLOAT32;
      inputs[0].size = 198 * 80 * sizeof(float);
      inputs[0].fmt = RKNN_TENSOR_NCHW;
      inputs[0].pass_through = 0;
      inputs[0].buf = flattenedFeats.data();

      rknn_ret=rknn_inputs_set(rk_ctx, 1, inputs);

      if (rknn_ret < 0) {
          printf("rknn input set failed, ret=%d\n", rknn_ret);
      } 

      rknn_ret = rknn_run(rk_ctx, NULL);
      if (rknn_ret < 0) 
      {
          printf(" rknn run failed, ret=%d\n", rknn_ret);
      }

      rknn_output outputs[1];
      memset(outputs, 0, sizeof(outputs));
      for (int i = 0; i < 1; i++)
      {
          outputs[i].want_float = 1;
      }

      rknn_ret = rknn_outputs_get(rk_ctx, 1, outputs, NULL);
      if (rknn_ret < 0) 
      {
          printf(" rknn output failed, ret=%d\n", rknn_ret);
      }

      memcpy(resnet_out.data(),outputs[0].buf,256*10*25*sizeof(float));

  }

} // namespace wespeaker
#endif
