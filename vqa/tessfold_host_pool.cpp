// Deterministic BF16->FP32 convex leaf aggregation without an MxN matrix.
// Separate numerical candidate: NOT the original GEMM reduction order.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>
extern "C" int tessfold_host_pool(const uint16_t* raw,const float* score,const int64_t* group,int n,int d,int m,float tau,uint16_t* output,int64_t* reps) {
  std::vector<float> max(m,-INFINITY),sum(m,0.0f),weights(n),values(m*d,0.0f);
  std::vector<int64_t> rep(m,n),order(m);
  for(int i=0;i<n;i++)if(score[i]>max[group[i]]){max[group[i]]=score[i];rep[group[i]]=i;}
  for(int i=0;i<n;i++){float a=score[i]/tau,b=max[group[i]]/tau;weights[i]=std::exp(a-b);sum[group[i]]=sum[group[i]]+weights[i];}
  for(int i=0;i<n;i++) {
    float weight=weights[i]/sum[group[i]];float* dest=values.data()+group[i]*d;const uint16_t* source=raw+i*d;
    #pragma GCC ivdep
    for(int j=0;j<d;j++) {
      uint32_t bits=uint32_t(source[j])<<16;float value;std::memcpy(&value,&bits,4);
      float product=weight*value;dest[j]=dest[j]+product;
    }
  }
  std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](int a,int b){return rep[a]<rep[b];});
  for(int k=0;k<m;k++) {
    reps[k]=rep[order[k]];const float* row=values.data()+order[k]*d;
    #pragma GCC ivdep
    for(int j=0;j<d;j++) {
      uint32_t bits;std::memcpy(&bits,row+j,4);bits+=0x7fff+((bits>>16)&1);output[k*d+j]=bits>>16;
    }
  }
  return m;
}
