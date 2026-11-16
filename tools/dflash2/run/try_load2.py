import torch, time
from transformers import AutoTokenizer
from transformers.models.qwen4_exp import Qwen4ExpForCausalLM
p='/home/relja/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300'
t0=time.time()
try:
    m = Qwen4ExpForCausalLM.from_pretrained(p, dtype=torch.bfloat16).to('cuda')
    print("LOADED %.1fs"%(time.time()-t0), type(m).__name__, "mem %.1f GiB"%(torch.cuda.memory_allocated()/2**30))
    tok=AutoTokenizer.from_pretrained(p)
    ids=tok("The capital of France is", return_tensors='pt').to('cuda')
    out=m(**ids, output_hidden_states=True)
    print("forwarded: logits", tuple(out.logits.shape), "hidden", len(out.hidden_states), tuple(out.hidden_states[0].shape))
    # the tap candidate at layer 5: check width
    hs=out.hidden_states
    print("hidden_states[6] shape (after layer 5):", tuple(hs[6].shape), "=> last dim / hidden =", hs[6].shape[-1]//2560)
except Exception as e:
    print("FAILED %.1fs"%(time.time()-t0)); print(type(e).__name__, str(e)[:700])
