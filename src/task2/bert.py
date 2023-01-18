import transformers
import torch
# ~/.cache/huggingface

config = transformers.BertConfig.from_json_file("data/pubmedbert/config.json")
# model = transformers.BertModel.from_pretrained("data/pubmedbert/pytorch_model.bin", config=config)
model = transformers.BertForSequenceClassification.from_pretrained("data/pubmedbert/pytorch_model.bin", config=config)
tokenizer = transformers.BertTokenizer("data/pubmedbert/vocab.txt")
# TODO: how do I use tokenizer_config.json?

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id])
