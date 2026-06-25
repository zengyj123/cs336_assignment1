import regex as re
import pickle
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.id_for_bytes = { b: i for i,b in self.vocab.items() }
        self.merge_rank = {p: i for i,p in enumerate(merges)}

    def decode(self, ids):
        tokens_bytes = b"".join(self.vocab[id] for id in ids)
        return tokens_bytes.decode("utf-8", errors = "replace")
    
    def _merge_word(self, word):
        while True:
            pairs = list(zip(word, word[1:]))
            if not pairs:
                break
            best =  min(pairs, key = lambda p: self.merge_rank.get(p, float("inf")))
            if best not in self.merge_rank:
                break
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word)-1 and word[i] == best[0] and word[i+1] == best[1]:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
        return word  

    def encode(self, text):
        ids = []
        if self.special_tokens:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in specials) + ")"
            pieces = re.split(pattern, text)
        else:
            pieces = [text]
        for piece in pieces:
            if self.special_tokens and piece in self.special_tokens:
                ids.append(self.id_for_bytes[piece.encode("utf-8")])   # 特殊token：直接给id
            else:
                for m in re.finditer(PAT, piece):          # ① 预分词
                    pretoken = m.group()
                    word = tuple(bytes([b]) for b in pretoken.encode("utf-8"))   # ② 拆单字节
                    word = self._merge_word(word)         # ③ 套用merges
                    for token in word:                    # ④ 每个token查id
                        ids.append(self.id_for_bytes[token])
        return ids

    def encode_iterable(self, iterable):
        for chunk in iterable:               # 文件逐行给字符串
            for id in self.encode(chunk):    # 复用你写好的 encode
                yield id                      # 一个一个吐出去

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
