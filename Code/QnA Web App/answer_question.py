import torch

# answer_question function is responsible for obtaining an answer to question from reference text 
# function takes 4 parameters 
def answer_question(question, reference, model, tokenizer):
    """
    Returns answer to given question by reference
    :param question: Input question for which answer to be found 
    :param reference: Input text/passage in which answer to be found 
    :param model: Model to use for prediction
    :param tokenizer: Tokenizer to use for the model
    :return: answer
    """

    # Tokenize the question and reference and assign IDs
    # tokenize both the question and the passage
    # tokenize, truncate them to a length of 512 and returns dict of {token_id, token_type_id, attention_mask}
    token_IDs = tokenizer.encode_plus(question, reference, max_length=512, truncation=True, return_tensors='pt')

    # Extract the tensor containing the token IDs from the dictionary
    input_tokens = token_IDs["input_ids"]
    token_type_ids = token_IDs["token_type_ids"]
    attention_mask = token_IDs["attention_mask"]
    # Make the model predict the start and end tokens of the answer
    # tokenized inputs passed to the model along with token_type_id and attention mask 
    model_output = model(input_tokens, token_type_ids=token_type_ids, attention_mask=attention_mask)
    # model outputs the start and end logits for each token in input. these scores represent likelihood of each token being the start or end of answer 
    start_scores, end_scores = model_output.start_logits, model_output.end_logits

    # Find the combination of start and end tokens that has the highest score
    # indices of token with highest start and end logits obtained 
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    # corresponding answer then extracted from input tokens 
    # extracted answer decoded back to natural language 
    answer = tokenizer.decode(input_tokens.squeeze()[answer_start:answer_end + 1].tolist())  # +1 to include last token
    # special tokens removed 
    special_tokens = ['[SEP]', '[CLS]', '[PAD]', '[UNK]']
    answer = ' '.join([word for word in answer.split() if word not in special_tokens])
    # processed answer returned 
    return answer