import regex as re
from collections import Counter,defaultdict
def train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()
    special_tokens_pattern = '|'.join(re.escape(tok) for tok in special_tokens)
    data = re.split(special_tokens_pattern, data)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    count = Counter()
    for paragraph in data:
        for m in re.finditer(PAT, paragraph):
            key = m.group().encode("utf-8")
            key = tuple(bytes([b]) for b in key)
            count[key] += 1
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
    