# /// script
# dependencies = [ "transformers", "accelerate" ]
# ///

# run on 2xH200 rented from primeintellect.ai

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

NAME = "meta-llama/Llama-3.3-70B-Instruct"
HISTORY = [
  {"role": "user", "content": "Is there a toaster emoji?"},
  {"role": "assistant", "content": "Yes, there is a toaster emoji:"}
]
TOPK = 5

tokenizer, model = AutoTokenizer.from_pretrained(NAME), AutoModelForCausalLM.from_pretrained(NAME, device_map="auto")
input_ids = tokenizer(
    tokenizer.apply_chat_template(HISTORY, continue_final_message=True, tokenize=False),
    return_tensors="pt",
).to(model.device)

model.generate(**input_ids, temperature=0.01, max_new_tokens=3, streamer=TextStreamer(tokenizer))
print()

logit_lens, topk = [], []
for _ in range(3):
    generated = torch.tensor([[ll[-1] for ll in logit_lens]]).to(model.device, dtype=input_ids.input_ids.dtype)
    out = model(input_ids=torch.hstack((input_ids.input_ids, generated)), output_hidden_states=True)
    logit_lens.append([])
    topk.append([])
    for hidden in out.hidden_states:
        ll = torch.log_softmax(model.lm_head(hidden[0, -1, :]), dim=-1)
        logit_lens[-1].append(ll.argmax().item())
        topk[-1].append(ll.topk(TOPK).indices.tolist())
    del out
    gc.collect()
    torch.cuda.empty_cache()

print(
  "         |   tokens                                               | tokens                    | token 0\n"
  f"         |   0                 1                 2                | (merged)                  | (topk {TOPK})\n"
  "---------|--------------------------------------------------------|---------------------------|--------------------------------------------------------------------------"
)
for i, x in enumerate(zip(*logit_lens)):
    if i % 4 == 0:
        print(f"Layer {i:>2} | ", end="")
        for token in x:
            print(f"{token:>7}{tokenizer.convert_ids_to_tokens(token)!r:<10} ", end="")
        print(" | ", end="")
        print(tokenizer.decode(list(x)).ljust(25), end = " | ")
        print([tokenizer.convert_ids_to_tokens(j) for j in topk[0][i]])
print()