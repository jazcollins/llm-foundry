import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

name = 'mosaicml/mpt-7b'

# Download config
config = AutoConfig.from_pretrained(name, trust_remote_code=True)
# (Optional) Use `triton` backend for fast attention. Defaults to `torch`.
config.attn_config['attn_impl'] = 'triton'
# (Optional) Change the `max_seq_len` allowed for inference
# config.max_seq_len = 4096

dtype = torch.bfloat16  # or torch.float32

# Download model source and weights
model = AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    torch_dtype=dtype,
    trust_remote_code=True)

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(name)

# Run text-generation pipeline
pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device='cuda:0',  # (Optional) to run on GPU 0
)
with torch.autocast('cuda', dtype=dtype):
    print(
        pipe('Here is a recipe for miso soup:\n',
            max_new_tokens=100,
            do_sample=True,
            use_cache=True))