path = "/home/tuhin/python_codes/Tiny LLM/data/tokenizer_train.txt"
train_text_1 = open("/home/tuhin/python_codes/Tiny LLM/data/AllCombined.txt").read()
train_text_2 = open("/home/tuhin/python_codes/Tiny LLM/data/bpe_training_100k.txt").read()

from tokenizer import BPETokenizer

tokenizer = BPETokenizer(path = path)
tokenizer.train_BPE(vocab_size=1000)

offset = int(1e5)
for i in range(12):
    tokenizer.continue_training(text = train_text_1[i*offset:(i+1)*offset])

tokenizer.continue_training(text = train_text_2)


test_text = "hello my name is Tuhin , I love machine learning."

utf_enc = test_text.encode(errors="replace")
my_enc = tokenizer.encode_text(text= test_text)
decoded = tokenizer.decode_encoding(my_enc)

print(decoded == test_text)
print(f"compression : {len(my_enc)/len(utf_enc)}")


tokenizer.save("tokenizer.json")



loaded = BPETokenizer.load("tokenizer.json")

text = "The Standard Model describes fundamental particles."

a = tokenizer.encode_text(text)
b = loaded.encode_text(text)

assert a == b
assert tokenizer.decode_encoding(a) == loaded.decode_encoding(b)

print("Tokenizer successfully saved and loaded.")
print("Vocabulary:", len(loaded.vocab))