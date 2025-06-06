#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs


# -----------------------------------------------------------------------------
# Python “reference” implementations of encode(...) and decode(...)
# -----------------------------------------------------------------------------
class TokenIndex:

    def __init__(self, s: str, idx: int):
        self.str = s
        self.id = idx


def python_encode(vocab: List[str], vocab_scores: List[float], max_token_length: int, text: str, bos_id: int,
                  eos_id: int, do_bos: bool, do_eos: bool) -> Tuple[List[int], int]:
    """
    Python‐side reference for Karpathy’s BPE encode(…) routine.
    Returns (list_of_token_ids, n_tokens)
    """
    # --- Build & sort TokenIndex array
    sorted_vocab = [TokenIndex(s, i) for i, s in enumerate(vocab)]
    sorted_vocab.sort(key = lambda ti: ti.str)

    def str_lookup(buf: str) -> int:
        for ti in sorted_vocab:
            if ti.str == buf:
                return ti.id
        return -1

    tokens: List[int] = []
    if do_bos:
        tokens.append(bos_id)

    # “dummy prefix” = ID of space if text not empty
    if text and text[0] != "\0":
        prefix_id = str_lookup(" ")
        if prefix_id >= 0:
            tokens.append(prefix_id)

    # UTF‐8 codepoint by codepoint
    buf = bytearray()
    for byte_val in text.encode("utf-8"):
        if (byte_val & 0xC0) != 0x80:
            buf.clear()
        buf.append(byte_val)

        # flush if next byte isn’t a continuation or buf grew > 4
        if (buf[-1] & 0xC0) != 0x80 or len(buf) > 4:
            try:
                codepoint_str = bytes(buf).decode("utf-8")
            except:
                codepoint_str = None

            if codepoint_str is not None:
                idx2 = str_lookup(codepoint_str)
            else:
                idx2 = -1

            if idx2 >= 0:
                tokens.append(idx2)
            else:
                for b in buf:
                    tokens.append(b + 3)
            buf.clear()

    # Merge (BPE) pass
    while True:
        best_score = -1e10
        best_idx = -1
        best_id = -1
        for i in range(len(tokens) - 1):
            pair = vocab[tokens[i]] + vocab[tokens[i + 1]]
            idx2 = str_lookup(pair)
            if idx2 >= 0 and vocab_scores[idx2] > best_score:
                best_score = vocab_scores[idx2]
                best_idx = i
                best_id = idx2
        if best_idx < 0:
            break
        tokens[best_idx] = best_id
        del tokens[best_idx + 1]

    if do_eos:
        tokens.append(eos_id)

    return tokens, len(tokens)


def python_decode(vocab: List[str], token_ids: List[int]) -> List[str]:
    """
    A trivial “decode” reference: map each token‐ID back to its vocab piece.
    """
    return [vocab[t] for t in token_ids]


# -----------------------------------------------------------------------------
# Script entry‐point: builds Encode→Slice→Slice→Concat→Decode ONNX
# plus NPZ files for testing
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description = "Build an ONNX graph that does Encode→Slice→Slice→Concat→Decode "
                                     "and dump inputs.npz / outputs.npz for testing.")
    parser.add_argument("--vocab-file",
                        required = True,
                        help = "Path to vocab.txt (each line: ID, piece, score separated by \",\\t\").")
    parser.add_argument("--prompt", required = True, help = "Input text prompt (UTF‐8).")
    parser.add_argument("--bos-id", type = int, required = True, help = "Integer index of the <BOS> token.")
    parser.add_argument("--eos-id", type = int, required = True, help = "Integer index of the <EOS> token.")
    parser.add_argument("--max-tokens",
                        type = int,
                        default = 512,
                        help = "Maximum output length (upper bound) for the tokenizer.")
    parser.add_argument("--out-dir",
                        default = "tokenizer_out",
                        help = "Directory to write: inputs.npz, outputs.npz, network.onnx")
    args = parser.parse_args()

    # ─── Step 1) Load vocab.txt → pieces[id], scores[id]
    pieces: List[str] = []
    scores: List[float] = []
    with open(args.vocab_file, encoding = "utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split(",\t")
            if len(parts) != 3:
                continue
            idx_str, piece, score_str = parts
            idx = int(idx_str)
            sc = float(score_str)
            if idx >= len(pieces):
                pieces.extend([""] * (idx + 1 - len(pieces)))
                scores.extend([0.0] * (idx + 1 - len(scores)))
            pieces[idx] = piece
            scores[idx] = sc

    # ─── Step 2) Run Python reference encode → (tokens_list, n_tok)
    tokens_list, n_tok = python_encode(vocab = pieces,
                                       vocab_scores = scores,
                                       max_token_length = max(len(p) for p in pieces),
                                       text = args.prompt,
                                       bos_id = args.bos_id,
                                       eos_id = args.eos_id,
                                       do_bos = True,
                                       do_eos = True)
    print("Python‐reference encoded tokens:", tokens_list)
    print("Python‐reference n_tokens =", n_tok)

    # ─── Step 3) Run Python reference decode → decoded_pieces
    decoded_pieces = python_decode(pieces, tokens_list)
    print("Python‐reference decoded pieces:", decoded_pieces)

    # ─── Step 4) Save inputs.npz / outputs.npz
    odir = args.out_dir
    os.makedirs(odir, exist_ok = True)

    input_bytes = np.frombuffer(args.prompt.encode("utf-8"), dtype = np.uint8)
    np.savez(os.path.join(odir, "inputs.npz"),
             input_bytes = input_bytes,
             max_length = np.array([args.max_tokens], dtype = np.int32))

    decoded_bytes = []
    for piece in decoded_pieces:
        b = piece.encode("utf-8")
        decoded_bytes.extend(b)
    out_arr = np.array(decoded_bytes, dtype = np.uint8)

    np.savez(os.path.join(odir, "outputs.npz"), decoded_pieces = out_arr)
    print(f"Saved reference outputs.npz (decoded_pieces length = {out_arr.shape[0]})")

    # ─── Step 5) Build the ONNX graph for Encode→Slice→Slice→Concat→Decode ───────
    N = args.max_tokens
    W = max(len(p) for p in pieces)

    # 5a) Graph inputs “input_bytes_tensor” & “max_length_tensor”
    input_bytes_var = gs.Variable(name = "input_bytes", dtype = np.uint8, shape = [int(input_bytes.shape[0])])
    max_length_var = gs.Variable(name = "max_length", dtype = np.int32, shape = [1])

    # 5b) Encode node outputs “token_ids_tensor” (shape [N]) and “n_tokens_tensor” (shape [1])
    token_ids_var = gs.Variable(name = "token_ids", dtype = np.int32, shape = [N])
    n_tokens_var = gs.Variable(name = "n_tokens", dtype = np.int32, shape = [1])

    encode_node = gs.Node(op = "Encode",
                          inputs = [input_bytes_var, max_length_var],
                          outputs = [token_ids_var, n_tokens_var],
                          attrs = {
                              "vocabSize": len(pieces),
                              "maxPieceLen": W,
                              "seqLen": N,
                              "byteFallbackOffset": 3,
                              "unkId": 0,
                              "bosId": args.bos_id,
                              "bosName": int(args.prompt[0] != "\0"),
                              "eosId": args.eos_id,
                              "eosName": int(args.prompt[-1] != "\0"),
                              "prefixId": pieces.index(" "),
                              "vocabPieces": pieces,
                              "vocabIds": list(range(len(pieces))),
                              "vocabScores": scores,
                          })

    # 5c) Slice #1: token_ids_tensor[0 : n_tok] → ids_slice_tensor
    slice1_starts = gs.Constant(name = "slice1_starts", values = np.array([0], dtype = np.int64))
    slice1_ends = gs.Constant(name = "slice1_ends", values = np.array([n_tok], dtype = np.int64))
    slice1_axes = gs.Constant(name = "slice1_axes", values = np.array([0], dtype = np.int64))
    slice1_steps = gs.Constant(name = "slice1_steps", values = np.array([1], dtype = np.int64))

    ids_slice = gs.Variable(name = "ids", dtype = np.int32, shape = [int(n_tok)])
    slice1_node = gs.Node(op = "Slice",
                          inputs = [token_ids_var, slice1_starts, slice1_ends, slice1_axes, slice1_steps],
                          outputs = [ids_slice])

    # 5d) Slice #2: token_ids_tensor[0 : (n_tok−1)] → prev_no_bos_tensor
    slice2_starts = gs.Constant(name = "slice2_starts", values = np.array([0], dtype = np.int64))
    slice2_ends = gs.Constant(name = "slice2_ends", values = np.array([n_tok - 1], dtype = np.int64))
    slice2_axes = gs.Constant(name = "slice2_axes", values = np.array([0], dtype = np.int64))
    slice2_steps = gs.Constant(name = "slice2_steps", values = np.array([1], dtype = np.int64))

    prev_no_bos = gs.Variable(name = "prev_no_bos", dtype = np.int32, shape = [int(n_tok - 1)])
    slice2_node = gs.Node(op = "Slice",
                          inputs = [token_ids_var, slice2_starts, slice2_ends, slice2_axes, slice2_steps],
                          outputs = [prev_no_bos])

    # 5e) Concat([bos_const, prev_no_bos_tensor], axis=0) → prev_with_bos_tensor
    bos_const = gs.Constant(name = "bos_const", values = np.array([args.bos_id], dtype = np.int32))
    prev_with_bos = gs.Variable(name = "prev_ids", dtype = np.int32, shape = [int(n_tok)])
    concat_node = gs.Node(op = "Concat",
                          inputs = [bos_const, prev_no_bos],
                          outputs = [prev_with_bos],
                          attrs = {"axis": 0})

    # 5f) Decode(prev_with_bos_tensor, ids_slice_tensor) → pieces_out_tensor
    pieces_out = gs.Variable(name = "pieces", dtype = np.uint8, shape = [int(out_arr.shape[0])])
    decode_node = gs.Node(op = "Decode",
                          inputs = [prev_with_bos, ids_slice],
                          outputs = [pieces_out],
                          attrs = {
                              "vocab": pieces,
                              "byte_pieces": [chr(i) for i in range(256)],
                              "bos_id": args.bos_id,
                              "eos_id": args.eos_id,
                              "maxPieceLen": W,
                              "nTokens": n_tok
                          })

    # ─── Glue them all together into one GS Graph ─────────────────────────────────
    graph = gs.Graph(nodes = [encode_node, slice1_node, slice2_node, concat_node, decode_node],
                     inputs = [input_bytes_var, max_length_var],
                     outputs = [pieces_out, n_tokens_var])

    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    onnx.save(model, os.path.join(odir, "network.onnx"))
    print("✅ Wrote ONNX graph to", os.path.join(odir, "network.onnx"))

    print(f"\nAll files written into “{odir}/”:")
    print(" • inputs.npz   (contains input_bytes, max_length)")
    print(" • outputs.npz  (contains decoded_pieces as uint8 array)")
    print(" • network.onnx (the Encode→Slice→Slice→Concat→Decode model)\n")


if __name__ == "__main__":
    main()
