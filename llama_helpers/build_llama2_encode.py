#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs


# -----------------------------------------------------------------------------
# Python implementation of Karpathy's encode(...)
# -----------------------------------------------------------------------------
class TokenIndex:

    def __init__(self, s: str, idx: int):
        self.str = s
        self.id = idx


def compare_tokens(a: TokenIndex, b: TokenIndex) -> int:
    # simple lex compare
    return (a.str > b.str) - (a.str < b.str)


def python_encode(vocab: List[str], vocab_scores: List[float], max_token_length: int, text: str, bos_id: int,
                  eos_id: int, do_bos: bool, do_eos: bool) -> Tuple[List[int], int]:
    # Build and sort TokenIndex array once
    sorted_vocab = [TokenIndex(s, i) for i, s in enumerate(vocab)]
    sorted_vocab.sort(key = lambda ti: ti.str)

    def str_lookup(buf: str) -> int:
        # binary search: but for simplicity do linear (small N)
        for ti in sorted_vocab:
            if ti.str == buf:
                return ti.id
        return -1

    # allocate tokens, n_tokens
    tokens: List[int] = []
    # optional BOS
    if do_bos:
        tokens.append(bos_id)

    # dummy_prefix = ID of " " if present
    if text and text[0] != "\0":
        prefix_id = str_lookup(" ")
        if prefix_id >= 0:
            tokens.append(prefix_id)

    # UTF-8 codepoints
    buf = bytearray()
    for c in text.encode("utf-8"):
        # continuation bytes start with 0b10xxxxxx (0x80..0xBF)
        if (c & 0xC0) != 0x80:
            # new codepoint
            buf.clear()
        buf.append(c)
        # peek next char
        # if next is continuation and buf too long continue
        # else flush
        # but since we're in pure Python, detect when either next is non-cont or end
        # We'll simple flush whenever the next in text.encode isn't continuation:
        # (we lack lookahead here, so decode per codepoint via utf-8)
        try:
            next_byte = None
        except:
            next_byte = None
        # when the next is not continuation or buf len>4:
        if len(buf) and ((len(buf) > 4) or ((buf[-1] & 0xC0) != 0x80)):
            s = bytes(buf).decode("utf-8", errors = "ignore")
            idx = str_lookup(s)
            if idx >= 0:
                tokens.append(idx)
            else:
                # byte‐fallback: +3 offset for first three reserved tokens
                for b in buf:
                    tokens.append(b + 3)
            buf.clear()

    # merge best pairs
    while True:
        best_score = -1e10
        best_idx = -1
        best_id = -1
        for i in range(len(tokens) - 1):
            a = vocab[tokens[i]] + vocab[tokens[i + 1]]
            idx = str_lookup(a)
            if idx >= 0 and vocab_scores[idx] > best_score:
                best_score = vocab_scores[idx]
                best_id = idx
                best_idx = i
        if best_idx < 0:
            break
        # merge
        tokens[best_idx] = best_id
        del tokens[best_idx + 1]

    # optional EOS
    if do_eos:
        tokens.append(eos_id)

    return tokens, len(tokens)


# -----------------------------------------------------------------------------
# Script entry‐point
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-file", required = True, help = "Path to vocab.txt (ID, piece, score per line)")
    p.add_argument("--prompt", required = True, help = "Input text prompt (UTF-8)")
    p.add_argument("--bos-id", type = int, required = True)
    p.add_argument("--eos-id", type = int, required = True)
    p.add_argument("--max-tokens", type = int, default = 512)
    p.add_argument("--out-dir", default = "encode_out")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok = True)

    # 1) load vocab
    pieces: List[str] = []
    scores: List[float] = []
    with open(args.vocab_file, encoding = "utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(",\t")
            if len(parts) != 3:
                continue
            idx, piece, sc = parts
            # print(f"idx={idx}, piece={piece}, sc={sc}")
            i = int(idx)
            if i >= len(pieces):
                pieces.extend([""] * (i + 1 - len(pieces)))
                scores.extend([0.0] * (i + 1 - len(scores)))
            pieces[i] = piece
            scores[i] = float(sc)

    # 2) run Python encoder
    tokens, n_tok = python_encode(
        vocab = pieces,
        vocab_scores = scores,
        max_token_length = max(len(p) for p in pieces),
        text = args.prompt,
        bos_id = args.bos_id,
        eos_id = args.eos_id,
        do_bos = True,
        do_eos = True,
    )

    print("Python encoded tokens:", tokens)
    print("n_tokens =", n_tok)

    # 3) save inputs.npz
    #   - input_bytes: raw UTF-8 bytes of prompt
    input_bytes = np.frombuffer(args.prompt.encode("utf-8"), dtype = np.uint8)
    np.savez(os.path.join(args.out_dir, "inputs.npz"),
             input_bytes = input_bytes,
             max_length = np.array(args.max_tokens, dtype = np.int32))

    # 4) save outputs.npz
    tok_arr = np.array(tokens, dtype = np.int32)
    np.savez(os.path.join(args.out_dir, "outputs.npz"),
             token_ids = tok_arr,
             n_tokens = np.array(n_tok, dtype = np.int32))

    # 5) build ONNX graph
    N = args.max_tokens
    W = max(len(p) for p in pieces)
    ib = gs.Variable("input_bytes", dtype = np.uint8, shape = [len(input_bytes)])
    ml = gs.Variable("max_length", dtype = np.int32, shape = [1])
    to = gs.Variable("token_ids", dtype = np.int32, shape = [len(tokens)])
    nt = gs.Variable("n_tokens", dtype = np.int32, shape = [1])

    node = gs.Node(op = "Encode",
                   inputs = [ib, ml],
                   outputs = [to, nt],
                   attrs = {
                       "vocabSize": len(pieces),
                       "maxPieceLen": W,
                       "seqLen": N,
                       "byteFallbackOffset": 3,
                       "unkId": 0,
                       "bosId": args.bos_id,
                       "bosName": args.prompt[0] != "\0",
                       "eosName": args.prompt[-1] != "\0",
                       "eosId": args.eos_id,
                       "prefixId": pieces.index(" "),
                       "vocabPieces": pieces,
                       "vocabIds": list(range(len(pieces))),
                       "vocabScores": scores,
                   })

    graph = gs.Graph(nodes = [node], inputs = [ib, ml], outputs = [to, nt])
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx.save(model, os.path.join(args.out_dir, "network.onnx"))
    print("✅ Wrote ONNX graph to", args.out_dir)


if __name__ == "__main__":
    main()
