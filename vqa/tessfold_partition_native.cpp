// Independent compiled FP32 reference partition. No fast-math/reassociation.
// NumPy2.2 float32 sum uses8 accumulators and128-element pairwise blocks on
// the reduction axis; non-fast-axis sum is sequential across rows.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <tuple>
#include <vector>

static float pairwise(const float* x,int n,int stride=1) {
  if(n<8) {float r=-0.0f;for(int i=0;i<n;i++)r=r+x[i*stride];return r;}
  if(n<=128) {
    float r[8];for(int j=0;j<8;j++)r[j]=x[j*stride];
    int i=8;for(;i<=n-8;i+=8)for(int j=0;j<8;j++)r[j]=r[j]+x[(i+j)*stride];
    float out=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
    for(;i<n;i++)out=out+x[i*stride];return out;
  }
  int half=n/2;half-=half%8;
  return pairwise(x,half,stride)+pairwise(x+half*stride,n-half,stride);
}
struct Rect {int a,b,c,d,id;};
struct Item {float mass;int id;};
struct Less {bool operator()(const Item& a,const Item& b)const {
  return a.mass!=b.mass?a.mass<b.mass:a.id>b.id;
}};

extern "C" int tessfold_partition(const float* image,int h,int w,int m,int mode,int64_t* out) {
  if(h<1||w<1||m<1||m>h*w)return -1;
  int stride=w+1;
  std::vector<float> integral((h+1)*(w+1),0.0f);
  // Exactly np.pad(image).cumsum(0).cumsum(1), in float32.
  for(int c=1;c<=w;c++)for(int r=1;r<=h;r++)
    integral[r*stride+c]=integral[(r-1)*stride+c]+image[(r-1)*w+c-1];
  for(int r=0;r<=h;r++)for(int c=1;c<=w;c++)
    integral[r*stride+c]=integral[r*stride+c-1]+integral[r*stride+c];
  auto mass=[&](const Rect& r){
    float a=integral[r.b*stride+r.d]-integral[r.a*stride+r.d];
    float b=a-integral[r.b*stride+r.c];return b+integral[r.a*stride+r.c];
  };
  std::vector<Rect> nodes;nodes.reserve(2*m+1);
  std::vector<uint8_t> alive;alive.reserve(2*m+1);
  std::priority_queue<Item,std::vector<Item>,Less> heap;
  auto add=[&](int a,int b,int c,int d){
    int id=nodes.size();nodes.push_back({a,b,c,d,id});alive.push_back(1);
    if((b-a)*(d-c)>1)heap.push({mass(nodes.back()),id});
  };
  add(0,h,0,w);std::vector<float> weights(std::max(h,w));
  int leaves=1;
  while(leaves<m) {
    if(heap.empty())return -2;
    Rect r=nodes[heap.top().id];heap.pop();alive[r.id]=0;
    int hh=r.b-r.a,ww=r.d-r.c;
    bool vertical=mode==1?true:ww>hh;
    if(ww==1)vertical=false;if(hh==1)vertical=true;
    int length=vertical?ww:hh;
    if(vertical) {
      // Vectorize across independent columns, retaining the identical row-wise
      // addition order inside EACH float32 column reduction.
      std::fill(weights.begin(),weights.begin()+ww,0.0f);
      for(int y=r.a;y<r.b;y++)for(int x=0;x<ww;x++)
        weights[x]=weights[x]+image[y*w+r.c+x];
    } else {
      for(int y=0;y<hh;y++)weights[y]=pairwise(image+(r.a+y)*w+r.c,ww);
    }
    float half=pairwise(weights.data(),length)/2.0f;
    float cumulative=0.0f;int cut=length;
    for(int i=0;i<length;i++) {cumulative=cumulative+weights[i];if(cumulative>=half){cut=i+1;break;}}
    cut=std::max(1,std::min(length-1,cut));
    if(vertical){add(r.a,r.b,r.c,r.c+cut);add(r.a,r.b,r.c+cut,r.d);}
    else {add(r.a,r.a+cut,r.c,r.d);add(r.a+cut,r.b,r.c,r.d);}
    leaves++;
  }
  std::vector<Rect> final;
  for(int i=0;i<(int)nodes.size();i++)if(alive[i])final.push_back(nodes[i]);
  std::sort(final.begin(),final.end(),[](const Rect&a,const Rect&b){return std::tie(a.a,a.b,a.c,a.d)<std::tie(b.a,b.b,b.c,b.d);});
  for(int k=0;k<m;k++)for(int y=final[k].a;y<final[k].b;y++)for(int x=final[k].c;x<final[k].d;x++)out[y*w+x]=k;
  return m;
}

extern "C" int tessfold_support(const int64_t* groups,int n,int m,int64_t* permutation,int64_t* offsets) {
  std::fill(offsets,offsets+m+1,0);
  for(int i=0;i<n;i++){if(groups[i]<0||groups[i]>=m)return -1;offsets[groups[i]+1]++;}
  for(int k=0;k<m;k++)offsets[k+1]+=offsets[k];
  std::vector<int64_t> cursor(offsets,offsets+m);
  for(int i=0;i<n;i++)permutation[cursor[groups[i]]++]=i;
  return n;
}
