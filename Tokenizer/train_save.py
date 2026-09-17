path_1 = "/home/tuhin/python_codes/Tiny LLM/data/ptb/ptb.train.txt" #(5mb text file)
path_2 = "/home/tuhin/python_codes/Tiny LLM/data/AllCombined.txt"

from Tokenizer.tokenizer_modified import BPETokenizer

tok = BPETokenizer()

size = tok.train_BPE(path = path_1,vocab_size=4500,return_token_length=True)

train_text_2 = open(path_2).read()[:int(3e6)] #only taking 3x10^6 chars
offset = int(5e5)
tok.continue_training(None,train_text_2[:offset],7500,None) # increase the vocabulary

current_voc_size = len(tok.vocab)

# increase the vocabulary more
for i in range(5):
    tok.continue_training(None,train_text_2[(i+1)*offset:(i+2)*offset],current_voc_size +(i+1)*1000 ,None)

# save the learned vocabulary and the merge table(unsorted)
tok.save(path = "/home/tuhin/python_codes/Tiny LLM/Tokenizer/new_tok.json")