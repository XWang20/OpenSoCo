from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

def get_tokenizer():
    tokenizer_obj = Tokenizer.from_file("/data/private/wangxing/OpenSoCo/bm_train_codes/config/en_tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '</s>'
    tokenizer.unk_token = '<unk>'
    tokenizer.mask_token = '<mask>'
    tokenizer.sep_token = '</s>'
    tokenizer.cls_token = '<s>'
    tokenizer.bos_token = '<s>'
    return tokenizer
