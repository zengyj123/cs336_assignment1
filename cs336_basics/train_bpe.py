import regex as re
from collections import Counter,defaultdict
import os
from multiprocessing import Pool
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_chunk(text, special_tokens):
    count = Counter()
    special_tokens_pattern = '|'.join(re.escape(tok) for tok in special_tokens)
    data = re.split(special_tokens_pattern, text)
    for paragraph in data:
        for m in re.finditer(PAT, paragraph):
            key = m.group().encode("utf-8")
            key = tuple(bytes([b]) for b in key)
            count[key] += 1
    return count
    


def train_bpe(input_path, vocab_size, special_tokens):
    num_processes = 16                       # 和 sbatch 的 --cpus-per-task 一致
    with open(input_path, 'rb') as f:        # ← 注意 'rb'（二进制），find_chunk_boundaries 要求
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
    # 并行：每块丢给 count_chunk
    with Pool(num_processes) as p:
        results = p.starmap(count_chunk, [(chunk, special_tokens) for chunk in chunks])
    # 合并多个 Counter 成一个
    count = Counter()
    for c in results:
        count.update(c)
    vocab = {i: bytes([i]) for i in range(256)}
    for idx, tok in enumerate(special_tokens, start=256):
        vocab[idx] = tok.encode('utf-8')
    num_merges = vocab_size - len(vocab)
    merges = []
    '''
    for _ in range(num_merges):
        pair_counts = Counter()
        for word, freq in count.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freq
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        new_count = Counter()
        for word,freq in count.items():
            i = 0
            new_word = []
            while i < len(word):
                if i <len(word) -1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            new_count[new_word] += freq
        count = new_count
    '''
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for word, freq in count.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)
    for _ in range(num_merges):
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))   # 直接选，不重数
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        affected = list(pair_to_words[best_pair])   
        for word in affected:
            freq = count[word]                       # 先拿到这个词的频次（权重）
            for pair in zip(word, word[1:]):         # 它【旧】的每个相邻对
                pair_counts[pair] -= freq            # 频次减回去
                pair_to_words[pair].discard(word)    # 从索引里移除这个词
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word)-1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] += freq            # 减 → 加
                pair_to_words[pair].add(new_word)    # discard → add
            del count[word]
            count[new_word] = freq

    return vocab, merges
    