import pickle
import time
import resource                              # ← 测内存
from cs336_basics.train_bpe import train_bpe

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1]                 # ← 读参数：路径
    vocab_size = int(sys.argv[2])            # ← 读参数：vocab（记得 int）
    prefix = sys.argv[3]                     # ← 读参数：前缀
    special_tokens = ["<|endoftext|>"]
    start = time.perf_counter()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    elapsed = time.perf_counter() - start
    with open(f"{prefix}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{prefix}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    longest = max(vocab.values(), key=len)
    print(repr(longest), len(longest))
    print(f"耗时 {elapsed:.1f} 秒")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"峰值内存 {peak/1024/1024:.2f} GB")
