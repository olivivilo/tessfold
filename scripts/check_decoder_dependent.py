"""Small independent oracles for entropy, KL direction and causal token alignment."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from vqa.token_proxy import random_matrix, BalancedMatrix
from vqa.decoder_dependent_loss import row_entropy, prediction_kl, common_prefix_logits


def main():
    torch.manual_seed(41); torch.set_num_threads(2)
    for n in [16,67,512]:
        for ratio in [.125,.5,.875]:
            p=random_matrix(torch.rand(n,2),ratio,'local',42)
            dense=p.dense().double()
            oracle=float(-torch.special.xlogy(dense,dense).sum(-1).mean())
            assert abs(row_entropy(p)-oracle)<2e-7
    identity=BalancedMatrix(torch.arange(16),torch.ones(16),16,0.)
    assert row_entropy(identity)==0
    a=torch.randn(1,5,73);b=torch.randn(3,5,73);lp=a.log_softmax(-1)
    oracle=torch.nn.functional.kl_div(b.log_softmax(-1),lp.exp().expand_as(b),reduction='none').sum(-1)
    torch.testing.assert_close(prediction_kl(lp,b),oracle)
    assert prediction_kl(lp,a).abs().max()==0
    c=Qwen3Config(vocab_size=79,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                  num_attention_heads=2,num_key_value_heads=1,head_dim=16)
    c.image_token_id=78
    class OfficialStyleWrapper(Qwen3ForCausalLM):
        def forward(self,input_ids=None,inputs_embeds=None,**kwargs):
            # Official MonkeyOCR's forward consumes ids locally and passes only
            # inputs_embeds to Qwen3; mirror that wrapper in this tiny oracle.
            return super().forward(input_ids=input_ids if inputs_embeds is None else None,
                                   inputs_embeds=inputs_embeds,**kwargs)
    model=OfficialStyleWrapper(c).eval()
    prompt=torch.tensor([[3,78,78,78,5,6]]);projection=torch.randn(2,3,32);answer=[11,12,13,14]
    with torch.no_grad():
        all_logits=common_prefix_logits(model,prompt,projection,answer)
        for t in range(len(answer)):
            ids=torch.cat((prompt,torch.tensor([answer[:t]],dtype=torch.long)),-1)
            embed=model.get_input_embeddings()(ids).clone();embed[0,1:4]=projection[0]
            one=model(inputs_embeds=embed,use_cache=False,logits_to_keep=1).logits
            torch.testing.assert_close(all_logits[0,t],one[0,0],atol=1e-6,rtol=1e-5)
        for i in range(2):
            single=common_prefix_logits(model,prompt,projection[i:i+1],answer)
            torch.testing.assert_close(all_logits[i],single[0],atol=1e-6,rtol=1e-5)
    print('PASS: sparse entropy vs dense; identity entropy and KL; full-vocab KL direction; causal response indexing; independent batch equivalence')


if __name__=='__main__':main()
