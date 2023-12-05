// kws demo
#ifndef PIPER_KWS
#include "kws.h"

int32_t init_kws(kws_params *kws_params)
{
    float value;

    kws_params->mfcc_fb=std::vector<std::vector<float>>(257, std::vector<float>(64));
    std::ifstream fb_file(kws_params->fb_path, std::ios::binary);
    if (!fb_file) {
        fprintf(stderr, "Error opening fb_t.bin file!\n");
        return -1;
    }
    for (int i = 0; i < kws_params->mfcc_fb.size(); ++i) {
        for (int j = 0; j < kws_params->mfcc_fb[0].size(); ++j) {
        fb_file.read((char*)&value, sizeof(float));
        kws_params->mfcc_fb[i][j] = value;
        } 
    }
    kws_params->embedding=std::vector<std::vector<float>>(864, std::vector<float>(2048));
    std::ifstream eb_file(kws_params->emb_path, std::ios::binary);
    if (!eb_file) {
        fprintf(stderr, "Error opening eb_file.bin file!\n");
        return -1;
    }
    for (int i = 0; i < kws_params->embedding.size(); ++i) {
        for (int j = 0; j < kws_params->embedding[0].size(); ++j) {
        eb_file.read((char*)&value, sizeof(float));
        kws_params->embedding[i][j] = value;
        } 
    }

    kws_params->rk_ctx=0;
    char * model_path=kws_params->eff_model_path;
    int ret = init_rknn(model_path,&kws_params->rk_ctx);
    if (ret < 0) {
        printf(" rknn init failed ret\n");
        return ret;
    }
 
    return 0;
}


std::vector<float> preemphasis(std::vector<float>& signal, float coeff) {
  std::vector<float> output(signal.size());
  output[0] = signal[0];
  for(size_t i = 1; i < signal.size(); ++i) {
    output[i] = signal[i] - coeff * signal[i-1]; 
  }
  return output;
}

std::vector<std::vector<float>> rolling_window(const std::vector<float>& originalVec) {

  vector<vector<float>> outVec;
  for(int i=0; i<23681; i+=160) {
    std::vector<float> newVec;

    std::vector<float> temp(originalVec.begin()+i, originalVec.begin()+i+400); 
    newVec.insert(newVec.end(), temp.begin(), temp.end());
    outVec.push_back(newVec);
  }

  return outVec;
}

vector<vector<float>> framesig_simple(vector<float> sig) {
    
    sig.insert(sig.end(), 80, 0);
    std::vector<std::vector<float>> frames=rolling_window(sig);
    return frames;
} 


vector<vector<float>> dot(vector<vector<float>> &A, vector<vector<float>> &B) {

  int m = A.size();
  int n = B[0].size();
  int k = A[0].size();

  vector<vector<float>> C(m, vector<float>(n));

  for(int i=0; i<m; i++) {
    for(int j=0; j<n; j++) {
      for(int l=0; l<k; l++) {
        C[i][j] += A[i][l] * B[l][j]; 

        if(C[i][j]==0)
        {
          C[i][j]=eps;
        }
       }
    }
  }

  return C;
}

vector<vector<float>> mfcc(vector<float> input_audio,vector<vector<float>> mfcc_fb)
{
    vector<vector<float>>  frames = framesig_simple(input_audio);
  
    vector<vector<float>> frame_out;
    auto start_mfcc = std::chrono::high_resolution_clock::now(); // 获取当前时间

    
    kiss_fft_cfg kiss_fft_cfg=kiss_fft_alloc(512, 0, NULL, NULL);
    // printf("frames.size()=%d\n",frames.size());
  
    for(int i=0;i<frames.size();i++)
    {
        vector<kiss_fft_cpx> cx_in = vector<kiss_fft_cpx>(512);
        vector<kiss_fft_cpx> cx_out = vector<kiss_fft_cpx>(512);
 

        for(int j=0; j<frames[i].size();j++)
        {
            cx_in[j]=kiss_fft_cpx();
            cx_in[j].r = frames[i][j]*500;
            cx_in[j].i = 0.0f;
    
        }
 
        kiss_fft(kiss_fft_cfg, cx_in.data(), cx_out.data());
     
        vector<float> fft_out;
        for(int k=0;k<257;k++)
        {
             
          double img = cx_out[k].i;
          double real = cx_out[k].r;
          float spec = pow(real, 2) + pow(img, 2);
          fft_out.push_back(spec/512);
        }
      
        frame_out.push_back(fft_out);

    }
    kiss_fft_free(kiss_fft_cfg);

 
    vector<vector<float>> feat=dot(frame_out,mfcc_fb);

    for(int i =0;i<feat.size();i++)
    {
      for(int j=0; j<feat[0].size();j++)
      {

        feat[i][j] = log(feat[i][j]);

      }
    }
    auto end_mfcc = std::chrono::high_resolution_clock::now(); // 获取当前时间
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_mfcc - start_mfcc); // 计算时间差
    // std::cout << "mfcc function took " << duration.count() / 1000.0 << " milliseconds." << std::endl; // 输出时间差
    
    return feat;
}

std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {

  // Check matrix dimensions
  if (A[0].size() != B.size()) {
    printf("Matrix dimensions must match");
  }
  
  // Initialize result matrix
  std::vector<std::vector<float>> C(A.size(), std::vector< float>(B[0].size()));

  // Multiply matrices
  for (int i = 0; i < A.size(); i++) {
    for (int j = 0; j < B[0].size(); j++) {
      for (int k = 0; k < A[0].size(); k++) {
        C[i][j] += A[i][k] * B[k][j]; 
      }
    }
  }

  return C;
}
bool kws_process(online_params *params,kws_params *kws_params)
{    
  
    // auto start = std::chrono::high_resolution_clock::now(); // 获取当前时间

    std::vector<std::vector<float>> feat_mfcc=mfcc(params->audio.pcmf32_new,kws_params->mfcc_fb);
    std::vector<float> t1;
    std::vector<float> t2;
    std::vector<float> bn_out(2048);
    std::vector<std::vector<float>> final_out(2048,std::vector<float>(1));
    pre_process(feat_mfcc,&t1,&t2);
    // auto start_1 = std::chrono::high_resolution_clock::now(); // 获取当前时间

    int ret = rknn_process(kws_params->rk_ctx, &t1, &t2,&bn_out);
    // auto end_1 = std::chrono::high_resolution_clock::now(); // 获取当前时间
    // auto duration_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1); // 计算时间差
    // std::cout << "rknn  took " << duration_1.count() / 1000.0 << " milliseconds." << std::endl; // 输出时间差
    
    float rk_output=post_process(bn_out);
    for (int i = 0; i < bn_out.size(); i++) 
    {
      final_out[i][0] = bn_out[i]/rk_output; 

    }

    // auto start = std::chrono::high_resolution_clock::now(); // 获取当前时间

    std::vector<std::vector<float>> cos_sim=matmul(kws_params->embedding,final_out);

    std::vector<float> proba(864);
    for(int i=0;i<cos_sim.size();i++)
    {
      cos_sim[i][0]+=1;
      cos_sim[i][0]/=2; 
      proba[i]=cos_sim[i][0];
      if(proba[i]>0.7)
      {
        printf("proba[%d]=%f\n",i,proba[i]);
        return true;
      }

    }
    // float maxNum = *std::max_element(proba.begin(), proba.end());
    // printf("maxNum=%f\n",maxNum);
    // auto end = std::chrono::high_resolution_clock::now(); // 获取当前时间
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); // 计算时间差
    // std::cout << "full function took " << duration.count() / 1000.0 << " milliseconds." << std::endl; // 输出时间差
    
    return false;
    
}



int32_t deinit_kws(kws_params *kws_params)
{

 
 
    kws_params->mfcc_fb.clear();
    kws_params->embedding.clear();
    rknn_destroy(kws_params->rk_ctx);

 
    return 0;
}




#endif
