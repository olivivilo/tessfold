"""Independent non-autoregressive OCR probes. No target decoder/projector imports."""
import hashlib,re
import torch
from torch import nn
import torch.nn.functional as F
CHARS='abcdefghijklmnopqrstuvwxyz0123456789.,%$€£-/():+'
NC=len(CHARS);NB=64;NT=8;DIM=NC+NB+NT
STOP=set('a an the is are was were of in on at to for from by with and or what which who how many much does do did this that document shown image according'.split())

def text_target(text):
    text=text.lower();v=torch.zeros(DIM)
    for c in set(text):
        if c in CHARS:v[CHARS.index(c)]=1
    for word in re.findall(r'[\w.%$€£+-]+',text):
        for i in range(len(word)-1):
            k=int.from_bytes(hashlib.sha256(word[i:i+2].encode()).digest()[:2],'little')%NB
            v[NC+k]=1
    v[NC+NB:]=torch.tensor([bool(text.strip()),any(c.isdigit() for c in text),bool(re.search(r'\d[.,]\d',text)), '%' in text,any(c in text for c in '$€£'),bool(re.search(r'-\d',text)),bool(re.search(r'\d[/-]\d',text)),any(c.isalpha() for c in text)])
    return v

def token_targets(u,words,grid):
    n=len(u);target=torch.zeros(n,DIM);line=torch.full((n,),-1,dtype=torch.long)
    dx,dy=.5/int(grid[2]),.5/int(grid[1])
    for w in words:
        x,y,bw,bh=w['box']
        mask=(u[:,0]>=x-dx)&(u[:,0]<=x+bw+dx)&(u[:,1]>=y-dy)&(u[:,1]<=y+bh+dy)
        if not mask.any():mask[((u-torch.tensor([x+bw/2,y+bh/2]))**2).sum(-1).argmin()]=True
        target[mask]=torch.maximum(target[mask],text_target(w['text']))
        line[mask]=w['line']
    return target,line

class EvidenceHead(nn.Module):
    def __init__(self,rms=1.):
        super().__init__();self.register_buffer('rms',torch.tensor(float(rms)))
        self.net=nn.Sequential(nn.Linear(768,192),nn.GELU(),nn.Linear(192,DIM))
    def forward(self,x):return self.net(x/self.rms)

def question_vectors(question):
    words=[s for s in re.findall(r'[\w.%$€£+-]+',question.lower()) if s not in STOP]
    if not words:words=[question]
    return torch.stack([text_target(s)[:NC+NB] for s in words])

def relevance(prob,question):
    q=question_vectors(question)
    # Supervised OCR output coordinates provide a shared, explicit text alphabet.
    return (F.normalize(prob[:,:NC+NB],dim=-1)@F.normalize(q,dim=-1).T).amax(-1)

def source_weights(prob,question):
    r=relevance(prob,question)
    return .3/len(r)+.7*torch.softmax(r/.1,dim=0)

def bernoulli_js(p,q):
    p=p.clamp(1e-6,1-1e-6);q=q.clamp(1e-6,1-1e-6);m=(p+q)/2
    return .5*(p*(p/m).log()+(1-p)*((1-p)/(1-m)).log()+q*(q/m).log()+(1-q)*((1-q)/(1-m)).log())

def region_read(x,u,centers,head,bandwidth=.04):
    d=((centers[:,None]-u[None])**2).sum(-1)
    val,idx=d.topk(min(8,len(u)),largest=False)
    w=torch.softmax(-val/(2*bandwidth**2),dim=-1)
    pooled=(x[idx]*w[:,:,None]).sum(1)
    conf=torch.exp(-val[:,0]/(2*bandwidth**2))
    p=head(pooled).sigmoid();p0=head(torch.zeros_like(pooled)).sigmoid()
    return conf[:,None]*p+(1-conf[:,None])*p0

def relation_states(p,pairs):
    # Spatially addressed text/number binding, not semantic table reasoning.
    a,b=p[pairs[:,0]],p[pairs[:,1]]
    at,an=a[:,-NT],a[:,-NT+1];bt,bn=b[:,-NT],b[:,-NT+1]
    states=torch.stack((at*(1-an)*bn,an*bt*(1-bn),an*bn,at*(1-an)*bt*(1-bn)),dim=-1)
    return states
