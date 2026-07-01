import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

def compression_ratio(tokenizer, text):
    num_bytes = len(text.encode("utf-8"))
    num_tokens = len(tokenizer.encode(text))
    return num_bytes / num_tokens

if __name__ == "__main__":
    # 1. 加载训练好的分词器（from_files 在这儿派上用场！）
    # (a) OWT 分词器 on OWT
    owt_tok = Tokenizer.from_files("owt_vocab.pkl", "owt_merges.pkl", ["<|endoftext|>"])
    with open("data/owt_valid.txt") as f:
        owt_docs = f.read().split("<|endoftext|>")
    owt_sample = [d for d in owt_docs[:20] if d.strip()][:10]   # 取10篇非空
    owt_ratios = [compression_ratio(owt_tok, d) for d in owt_sample]
    print(f"OWT分词器 on OWT: {sum(owt_ratios)/len(owt_ratios):.2f} bytes/token")

    # (b) TinyStories 分词器 on OWT(跨分词器)
    ts_tok = Tokenizer.from_files("ts_vocab.pkl", "ts_merges.pkl", ["<|endoftext|>"])
    cross = [compression_ratio(ts_tok, d) for d in owt_sample]
    print(f"TinyStories分词器 on OWT: {sum(cross)/len(cross):.2f} bytes/token")
    """ # 3. 吞吐量、(d) 编码存盘...
    # 拿一段有代表性的文本（比如这 10 篇拼起来）计时编码
    sample_text = "".join(sample)
    start = time.perf_counter()
    tok.encode(sample_text)
    elapsed = time.perf_counter() - start
    throughput = len(sample_text.encode("utf-8")) / elapsed   # bytes/秒
    print(f"吞吐量: {throughput:.0f} bytes/秒")
    pile_bytes = 825 * 1024**3                      # 825GB 换算成字节
    seconds = pile_bytes / throughput
    print(f"编码 Pile (825GB) 约需 {seconds/3600:.1f} 小时")
    with open("data/TinyStoriesV2-GPT4-train.txt") as f:
        ids = np.fromiter(tok.encode_iterable(f), dtype=np.uint16)
    np.save("ts_train_ids.npy", ids)
    print(f"训练集 token 数: {len(ids)}")
    with open("data/TinyStoriesV2-GPT4-valid.txt") as f:
        ids = np.fromiter(tok.encode_iterable(f), dtype=np.uint16)
    np.save("ts_valid_ids.npy", ids) """