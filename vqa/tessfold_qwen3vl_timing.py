"""Native8B natural greedy unchanged; measure actual first selected token.

Isolated from the frozen training source. Visual setup includes4branch P@X,
representative-position selection, embedding lookup and visual scatter, rather
than incorrectly calling only router(P) the entire compression cost.
"""
import time
import torch
from vqa.tessfold_qwen3vl import NativeQwen

class TimingQwen(NativeQwen):
    @torch.no_grad()
    def greedy(self,data,p=None,reps=None,cap=64,fixed=False):
        self.first_token_timestamp=None
        torch.npu.synchronize();start=time.perf_counter()
        ids,embeds,positions,mask,deep=self.visual(data,p,reps)
        torch.npu.synchronize();self.visual_setup_ms=(time.perf_counter()-start)*1e3
        length=embeds.shape[1];out=self.model.model.language_model(inputs_embeds=embeds,position_ids=positions,
            attention_mask=torch.ones((1,length),dtype=torch.long,device=self.device),use_cache=True,
            visual_pos_masks=mask,deepstack_visual_embeds=deep)
        response=[];last=positions[:,:,-1:];eos=self.model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos
        for step in range(cap):
            logits=self.model.lm_head(out.last_hidden_state[:,-1:])[:,0]
            # HF min_new_tokens=cap suppresses EOS at ALL cap selected steps.
            if fixed:logits[:,eos]=-torch.inf
            token=logits.argmax(-1);value=int(token);response.append(value)
            if step==0:self.first_token_timestamp=time.perf_counter()
            if not fixed and value in eos:break
            if step==cap-1:break
            last=last+1
            out=self.model.model.language_model(input_ids=token[:,None],position_ids=last,past_key_values=out.past_key_values,
                attention_mask=torch.ones((1,length+step+1),dtype=torch.long,device=self.device),use_cache=True)
        return response
