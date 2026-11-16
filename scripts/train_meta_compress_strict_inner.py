"""Same MetaCompress, inserted at a boundary INSIDE the frozen merger."""
from scripts import train_meta_compress_strict as runner
from vqa.meta_compress_dense_position_initialization import initialize
from vqa.meta_compress import meta_losses
from vqa.decoder_dependent_loss import common_prefix_logits
from scripts.eval_token_proxy import shortened_inputs

def student(engine,module,row,examples,cfg):
    prompt,raw,grid,x,response,teacher,tp=examples.get(row)
    merger=engine.model.vision_tower.merger
    source=merger.ln_q(raw)
    if cfg['insertion']=='post-gelu':source=merger.mlp[1](merger.mlp[0](source))
    elif cfg['insertion']!='post-norm':raise ValueError(cfg['insertion'])
    c,p=module(source.float(),grid,cfg['ratio'])
    sp=c.to(raw.dtype)
    sp=merger.mlp(sp) if cfg['insertion']=='post-norm' else merger.mlp[2](sp)
    si=shortened_inputs({'input_ids':prompt},len(c),engine.model.config.image_token_id)
    sl=common_prefix_logits(engine.model,si['input_ids'],sp[None],response)
    return meta_losses(p,teacher,sl),(c,p,sp,si,raw,grid,response,tp)

runner.initialize=initialize
runner.student=student
runner.SOURCES=runner.SOURCES+['scripts/train_meta_compress_strict_inner.py',
                             'vqa/meta_compress_dense_position_initialization.py']
if __name__=='__main__':runner.main()
