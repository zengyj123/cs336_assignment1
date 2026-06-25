import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

def compression_ratio(tokenizer, text):
    num_bytes = len(text.encode("utf-8"))
    num_tokens = len(tokenizer.encode(text))
    return num_bytes / num_tokens

if __name__ == "__main__":
    # 1. 加载训练好的分词器（from_files 在这儿派上用场！）
    tok = Tokenizer.from_files("ts_vocab.pkl", "ts_merges.pkl", ["<|endoftext|>"])

    # 2. 读数据、抽10篇文档、算压缩率
    with open("data/TinyStoriesV2-GPT4-valid.txt") as f:
        text = f.read()
    docs = text.split("<|endoftext|>")    # 按文档分隔符切
    sample = docs[:10]                     # 取前10篇
    ratios = [compression_ratio(tok, doc) for doc in sample]
    avg = sum(ratios) / len(ratios)
    print(f"TinyStories 平均压缩率: {avg:.2f} bytes/token")
    # 3. 吞吐量、(d) 编码存盘...
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