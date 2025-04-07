from transformers import BertTokenizer, BertForQuestionAnswering, BertModel
import torch
# https://modelscope.cn/models/AI-ModelScope/bert-base-cased/files
    # "pretrain_model_path":r"/Users/mac/Documents/bert-base-chinese",

tokenizer = BertTokenizer.from_pretrained(r"/Users/mac/Documents/bert-base-cased")
model = BertForQuestionAnswering.from_pretrained(r"/Users/mac/Documents/bert-base-cased")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
input_ids = tokenizer.encode(question, text)
token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

start_scores = output.start_logits
end_scores = output.end_logits

all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

print(all_tokens, 'all_tokens')

start = torch.argmax(start_scores)
end = torch.argmax(end_scores[start:])

answer = ' '.join(all_tokens[start:end+1])
print(start, end, answer, 'answer')
