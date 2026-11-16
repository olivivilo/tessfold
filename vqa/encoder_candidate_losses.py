"""Encoder-only candidate proxies. No projector, Qwen, answers, or score inputs.

Coarsened Sinkhorn is explicitly an approximation; subspace retains all 768
feature coordinates. The expensive downstream model is used only by the cached
external evaluation, never by these losses or auxiliary head training.
"""
import math
import torch
from vqa.encoder_evidence import NC,NB,NT,source_weights,region_read,bernoulli_js,relation_states

def sqdist(a,b):return (a.square().sum(-1)[:,None]+b.square().sum(-1)[None]-2*a@b.T).clamp_min(0)

def coreset(z,u,a,grid=(32,16)):
    nx,ny=grid;idx=(u[:,0]*nx).long().clamp(0,nx-1)+nx*(u[:,1]*ny).long().clamp(0,ny-1)
    mass=torch.zeros(nx*ny,dtype=z.dtype).index_add(0,idx,a)
    keep=mass>1e-12
    v=torch.cat([z,.5*u],-1)
    sums=torch.zeros(nx*ny,v.shape[-1],dtype=z.dtype).index_add(0,idx,a[:,None]*v)
    points=sums[keep]/mass[keep,None];weights=mass[keep];weights=weights/weights.sum()
    distortion=(a*(v-sums[idx]/mass[idx,None].clamp_min(1e-30)).square().sum(-1)).sum()
    return points,weights,float(distortion)

def entropic_ot(x,a,y,b,epsilon=.05,iterations=150,tolerance=1e-6,max_iterations=1500):
    # Double-precision scaling; all exp exponents remain representable for
    # normalized encoder features. Fall back to stable log domain if needed.
    d=sqdist(x.double(),y.double());a=a.double();b=b.double()
    logk=-d/epsilon;k=logk.exp();v=torch.ones_like(b);tiny=1e-300
    error=float('inf');mode='scaling'
    for it in range(max_iterations):
        u=a/(k@v).clamp_min(tiny);v=b/(k.T@u).clamp_min(tiny)
        if (it+1)%25==0 and it+1>=iterations:
            error=float(torch.max(torch.abs(u*(k@v)-a)))
            if error<tolerance:break
    if not torch.isfinite(u).all() or not torch.isfinite(v).all() or float(d.max()/epsilon)>650:
        mode='log';lu=torch.zeros_like(a);lv=torch.zeros_like(b)
        for it in range(max_iterations):
            lu=a.log()-torch.logsumexp(logk+lv[None],dim=1)
            lv=b.log()-torch.logsumexp(logk+lu[:,None],dim=0)
            if (it+1)%25==0 and it+1>=iterations:
                plan=(lu[:,None]+logk+lv[None]).exp();error=float((plan.sum(1)-a).abs().max())
                if error<tolerance:break
        plan=(lu[:,None]+logk+lv[None]).exp()
    else:plan=u[:,None]*k*v[None]
    logratio=plan.clamp_min(tiny).log()-a.log()[:,None]-b.log()[None]
    value=(plan*d).sum()+epsilon*((plan*logratio).sum()-plan.sum()+1)
    error=max(float((plan.sum(1)-a).abs().max()),float((plan.sum(0)-b).abs().max()))
    return value.float(),{'marginal_max_error':error,'iterations':it+1,'mode':mode}

class EncoderCandidates:
    def __init__(self,x,u,question,head,projection,anchor_indices,grid=(32,16)):
        self.x=x;self.u=u;self.head=head;self.grid=grid;self.anchors=anchor_indices
        self.rms=float(head.rms);self.projection=projection
        # Linear random orthogonal map; one train-derived scalar scale for both.
        self.z=x@projection/(self.rms*math.sqrt(projection.shape[1]))
        with torch.no_grad():self.prob=head(x).sigmoid()
        self.a=source_weights(self.prob,question);self.unif=torch.full((len(x),),1/len(x))
        self.tu=coreset(self.z,u,self.unif,grid);self.tq=coreset(self.z,u,self.a,grid)
        self.otu,self.diagu=entropic_ot(*self.tu[:2],*self.tu[:2])
        self.otq,self.diagq=entropic_ot(*self.tq[:2],*self.tq[:2])
        self.otqu,self.diagqu=entropic_ot(*self.tq[:2],*self.tu[:2])
        self.identity_qsink=self.otqu-.5*self.otq-.5*self.otu
        self.ax=x[anchor_indices];self.au=self.unif[anchor_indices];self.au=self.au/self.au.sum()
        self.aq=self.a[anchor_indices];self.aq=self.aq/self.aq.sum()
        self.subself=self.subspace(x)
        nx,ny=16,8;yy,xx=torch.meshgrid((torch.arange(ny)+.5)/ny,(torch.arange(nx)+.5)/nx,indexing='ij')
        self.centers=torch.stack([xx.flatten(),yy.flatten()],-1)
        pairs=[]
        for j in range(ny):
            for i in range(nx):
                k=j*nx+i
                if i+1<nx:pairs.append((k,k+1))
                if j+1<ny:pairs.append((k,k+nx))
        self.pairs=torch.tensor(pairs)
        self.rp=region_read(x,u,self.centers,head)
        w=source_weights(self.rp,question);self.rw=w/w.sum()
        self.rel=relation_states(self.rp,self.pairs)
        self.relw=(self.rw[self.pairs[:,0]]+self.rw[self.pairs[:,1]])/2;self.relw=self.relw/self.relw.sum()

    def subspace(self,c):
        lam=.001*self.rms**2
        g=c.T@c/len(c)+lam*torch.eye(c.shape[1])
        chol=torch.linalg.cholesky(g)
        residual=lam*torch.cholesky_solve(self.ax.T.contiguous(),chol).T
        e=residual.square().sum(-1);den=self.ax.square().sum(-1)
        return torch.stack([(a*e).sum()/(a*den).sum().clamp_min(1e-12) for a in [self.au,self.aq]])

    def __call__(self,matrix):
        c=matrix.compress(self.x);cu=matrix.compress(self.u)
        z=c@self.projection/(self.rms*math.sqrt(self.projection.shape[1]))
        d=sqdist(self.z[self.anchors],z);du=sqdist(self.u[self.anchors],cu)
        losses={}
        for name,a in [('uniform',self.au),('question',self.aq)]:
            losses['coverage_'+name]=float((a*d.min(1).values).sum())
            losses['coverage_pos_'+name]=float((a*(d+.25*du).min(1).values).sum())
        sc=coreset(z,cu,torch.full((len(c),),1/len(c)),self.grid)
        otu,diag_u=entropic_ot(*self.tu[:2],*sc[:2]);otq,diag_q=entropic_ot(*self.tq[:2],*sc[:2]);ots,diag_s=entropic_ot(*sc[:2],*sc[:2])
        losses['sinkhorn_uniform']=float(otu-.5*self.otu-.5*ots)
        losses['sinkhorn_question_raw']=float(otq-.5*self.otq-.5*ots)
        losses['sinkhorn_question_excess']=losses['sinkhorn_question_raw']-float(self.identity_qsink)
        sub=self.subspace(c)
        losses.update(subspace_uniform=float(sub[0]),subspace_question=float(sub[1]),subspace_question_excess=float(sub[1]-self.subself[1]))
        sp=region_read(c,cu,self.centers,self.head)
        js=bernoulli_js(self.rp,sp)
        for name,w in [('uniform',torch.full_like(self.rw,1/len(self.rw))),('question',self.rw)]:
            losses['evidence_text_'+name]=float((w*js[:,:NC+NB].mean(-1)).sum())
            losses['evidence_number_'+name]=float((w*js[:,-NT+1:-1].mean(-1)).sum())
        rel=bernoulli_js(self.rel,relation_states(sp,self.pairs)).mean(-1)
        losses['evidence_relation_question']=float((self.relw*rel).sum())
        losses['evidence_combined_question']=losses['evidence_text_question']+losses['evidence_number_question']+losses['evidence_relation_question']
        diag={'ot_max_marginal_error':max(diag_u['marginal_max_error'],diag_q['marginal_max_error'],diag_s['marginal_max_error']),
              'ot_max_iterations':max(diag_u['iterations'],diag_q['iterations'],diag_s['iterations']),
              'coreset_student_bins':len(sc[0]),'coreset_student_distortion':sc[2],
              'coreset_teacher_distortion':self.tu[2],'coreset_teacher_question_distortion':self.tq[2],
              'question_ess_fraction':float(1/(self.a.square().sum()*len(self.a))),
              'subspace_teacher_baseline_uniform':float(self.subself[0]),'subspace_teacher_baseline_question':float(self.subself[1]),
              'sinkhorn_question_identity_baseline':float(self.identity_qsink),'student_m_ge_d':len(c)>=c.shape[1]}
        return c,losses,diag
