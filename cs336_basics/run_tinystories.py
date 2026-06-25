import pickle
import time
from cs336_basics.train_bpe import train_bpe
if __name__ == "__main__":
    start = time.perf_counter()
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe("tests/fixtures/tinystories_sample_5M.txt", vocab_size, special_tokens)
    elapsed = time.perf_counter() - start
    with open("ts_vocab.pkl", "wb") as f:    # "wb" = 写二进制
        pickle.dump(vocab, f)
    with open("ts_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    # 以后读回：vocab = pickle.load(open("ts_vocab.pkl", "rb"))
    longest = max(vocab.values(), key=len)
    print(repr(longest), len(longest))
    print(f"耗时 {elapsed:.1f} 秒")