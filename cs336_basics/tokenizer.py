import os
import regex as re
from typing import Iterable, Iterator

# for test
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],  
        merges: list[tuple[bytes, bytes]],  
        special_tokens: list[str] | None = None  
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}  # bytes: int

        self._PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # GPT2正则化

    # def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None): 
    '''
    Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the 
    same format that your BPE training code output) and (optionally) a list of special tokens. 
    This method should accept the following additional parameters:
    vocab_filepath: str  
    merges_filepath: str  
    special_tokens: list[str] | None = None  
    '''
    def _get_id(self, token : bytes) -> int:
        for id, t in self.vocab.items():
            #print(f"token = {token}, vocab = {t}")
            if (t == token):
                return id
        return -1


    def _merge_pair(
        self, token_bytes: list[bytes], pair: tuple[bytes, bytes], new_token: bytes
    ) -> list[bytes]:
        new_token_bytes = []
        i = 0
        while i < len(token_bytes):
            if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i+1]) == pair:
                new_token_bytes.append(new_token)
                i += 2
            else:
                new_token_bytes.append(token_bytes[i])
                i += 1
        return new_token_bytes

    def encode(self, text: str) -> list[int]: # Encode an input text into a sequence of token IDs.
    # pre_tokenization
        if self.special_tokens:
            special_pattern = "|".join(
                re.escape(token)
                for token in self.special_tokens
            )

            chunks = re.split(
                f"({special_pattern})",
                text
            )
        else:
            chunks = [text]
        # encode
        encoded_result : list[int] = []

        for chunk in chunks:
            # deal with special tokens
            if chunk in self.special_tokens:
                token_id = self._get_id(chunk.encode("utf-8"))
                encoded_result.append(token_id)
                continue
            
            for match in re.finditer(self._PAT, chunk):
                pre_tokens = match.group().encode("utf-8")
                
                pre_token = [bytes([b]) for b in pre_tokens]
                #print(pre_token)
                while len(pre_token) >= 2:
                    pairs = list(zip(pre_token, pre_token[1:]))
                    #print(pairs)
                    max_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))
                    #print(max_pair)
                    if max_pair not in self.merge_ranks:
                        break
                    # print(pre_tokens)
                    new_token = max_pair[0] + max_pair[1]
                    #print(pre_token)
                    pre_token = self._merge_pair(pre_token, max_pair, new_token)
                    #print(pre_token)

                for token in pre_token:
                    token_id = self.vocab_reversed.get(token)
                    if token_id is not None:
                        encoded_result.append(token_id)

        return encoded_result

                # b = 0
                # e = len(pre_tokens)
                # while b < e:
                #     id = self._get_id(pre_tokens[b: e])
                #     if id == -1:
                #         e -= 1
                #     else:
                #         encoded_result.append(id)
                #         b = e
                #         e = len(pre_tokens)

    def decode(self, ids: list[int]) -> str: # Decode a sequence of token IDs into text.
        decoded_bytes = b"".join(
            self.vocab[id] for id in ids
        )
        # attention decoder of invaild input (e.g.0, 1, etc.)
        return decoded_bytes.decode("utf-8", errors="replace")


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id


'''
    Given an iterable of 
    strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is 
    required for memory-efficient tokenization of large files that we cannot directly load into 
    memory.
'''
    



def main():
    from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    corpus_path = FIXTURES_PATH / "german.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    ids = tokenizer.encode(corpus_contents)
    #print(ids)
    assert tokenizer.decode(ids) == corpus_contents


if __name__ == "__main__":
    main()
'''
[32423, 1004,...00, 2059, ...]
[32423, 406, ...00, 2059, ...]
'''