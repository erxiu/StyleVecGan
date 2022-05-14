import torch
from transformers import AutoTokenizer, AutoModel, pipeline


def bert_and_token(model_path="E:/code/BERT/bert-base-chinese/"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    return model, tokenizer

if __name__ == '__main__':
    model, tokenizer = bert_and_token()
    print(model)
    sentence1 = "今天是个好日子，明天会更好，后天很快就来了"
    sentence2 = "明天会更好，后天很快就来了"
    inputs1 = tokenizer(sentence1, return_tensors="pt")
    inputs2 = tokenizer(sentence2, return_tensors="pt")
    print(inputs1)
    print(inputs2)

    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

    # 比较同一个字符在不同的句子当中的表示是不是相同的
    # print(outputs1.last_hidden_state.squeeze(0)[0])
    # print(outputs2.last_hidden_state.squeeze(0)[0])
    print(outputs1.last_hidden_state.squeeze(0)[-1]-outputs2.last_hidden_state.squeeze(0)[-1])
