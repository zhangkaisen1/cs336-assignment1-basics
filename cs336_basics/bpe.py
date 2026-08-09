import os
import sys
import regex as re
# from pretokenization_example import find_chunk_boundaries
from collections import defaultdict

#for test
i_path = "/home/kaisen_zhang/projects/cs336/assignments/assignment1-basics/tests/fixtures/corpus.en"
special_token=["<|endoftext|>"]

class BPETokenizer:

    def __init__(
        self,
        vocab,
        merges,
        special_tokens=None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []



class TrainBPE:
    def __init__(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens = None
    ):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []

        self._PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # GPT2正则化


    def pre_tokenization(self):
        
        # read file
        f = open(self.input_path, 'r')
        corpus = f.read()

        # delete special tokens
        special_pattern = "|".join(
            re.escape(token)
            for token in self.special_tokens
        )
        chunks = re.split(special_pattern, corpus)
        
        # vocabulary initialize
        vocab = {
            i: bytes([i])
            for i in range(256)
        }
        for i in range(len(self.special_tokens)):
            vocab[i + 256] = self.special_tokens[i].encode("utf-8")

        # get ids
        ids : list[list[int]] = []
        for chunk in chunks:
            for match in re.finditer(self._PAT, chunk):
                tokens = match.group().encode("utf-8")
                ids.extend([list(tokens)])
        
        # get pair counts
        pair_to_indices, counts = self._get_pair_counts(ids)
        
        return vocab, ids, pair_to_indices, counts
        

    def _get_pair_counts(
        self, 
        ids: list[list[int]]
    ) -> tuple[
            defaultdict[tuple[int, int], set], 
            defaultdict[tuple[int, int], int]
        ]:
        pair_to_indices = defaultdict(set)
        counts = defaultdict(int)
        for i, tokens in enumerate(ids):
            for pair in zip(tokens, tokens[1:]):
                pair_to_indices[pair].add(i)
                counts[pair] += 1
        return pair_to_indices, counts



    def compute_merge(
        self,
        vocab,
        merges,
        ids,
        pair_to_indices,
        counts
    ):
        # find most frequent pair 
        def rank(pair: tuple[int, int]) -> tuple[int, tuple[bytes, bytes]]:
            return counts[pair], (vocab[pair[0]], vocab[pair[1]])
        max_pair = max(counts, key=rank)
        merges.append(max_pair)
        new_token = vocab[max_pair[0]] + vocab[max_pair[1]]
        new_id = len(vocab)
        vocab[new_id] = new_token

        # alter pair,delete first, then merge, finally add again
        # re:https://github.com/donglinkang2021/cs336-assignment1-basics/blob/main/cs336_basics/bpe.py
        alter_ids = pair_to_indices[max_pair].copy()
        for i in alter_ids:
            token_ids = ids[i]
            if len(token_ids) < 2:
                continue
            for pair in zip(token_ids, token_ids[1:]):
                counts[pair] -= 1
                pair_to_indices[pair].discard(i)
                if counts[pair] == 0:
                    del counts[pair]
                    del pair_to_indices[pair]
                
        # merge pair
            new_token_ids = []
            j = 0
            while j < len(token_ids):
                if j < len(token_ids) - 1 and (token_ids[j], token_ids[j + 1]) == max_pair:
                    new_token_ids.append(new_id)
                    j += 2
                else:
                    new_token_ids.append(token_ids[j])
                    j += 1
            
        # add again
            for pair in zip(new_token_ids, new_token_ids[1:]):
                counts[pair] += 1
                pair_to_indices[pair].add(i)

            ids[i] = new_token_ids

        return 

            



        


    def train_bpe(
            self
    )-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab, ids, pair_to_indices, counts = self.pre_tokenization()
        merges : list[tuple[bytes, bytes]] = []
        train_num = self.vocab_size - len(vocab)
        for i in range(train_num):
            if not counts:
                break
            self.compute_merge(vocab, merges, ids, pair_to_indices, counts)

        merges = [(vocab[a], vocab[b]) for a, b in merges]
        return vocab, merges


    # def train_bpe(
    #     self,
    #     num_processes, int = 4
    # )-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    #     # initialize vacabulary
    #     vocab = {
    #         i: bytes([i])
    #         for i in range(256)
    #     }
    #     for i in range(len(self.special_tokens)):
    #         vocab[i + 256] = self.special_tokens[i].encode("utf-8")

    #     # pre_token**
    #     with open(self.input_path, 'r') as f:
    #         boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>"encode("utf-8"))
        
    #     for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")

def main():
    tbpe = TrainBPE(i_path, 500, special_token)
    vocab, merges = tbpe.train_bpe()
    print(type(merges))
    return 0


if __name__ == "__main__":
    main()





 
