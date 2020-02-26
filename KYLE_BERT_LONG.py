"""

Experiments with BERt and hyperpartisan

"""

import torch

from src.transformers import RobertaModel, RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
encoding = tokenizer.encode("Hello, my dog is cute")
decoding = tokenizer.decode(encoding)
assert decoding == '<s>Hello, my dog is cute</s>'
T1 = torch.tensor(encoding)  # dimension 8
T2 = T1.unsqueeze(0)  # dimensions (1 x 8)
outputs = model(T2)  # tuple of 2 elements
outputs[0].shape    # dimensions (1 x 8 x 768)
outputs[1].shape    # dimensions (1 x 768)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple



