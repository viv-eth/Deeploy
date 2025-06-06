#!/usr/bin/env python3
import argparse
import os
import re
from typing import List

import numpy as np
import onnx
import onnx_graphsurgeon as gs

_HEX_RE = re.compile(r"^<0x([0-9A-F]{2})>$")


class Tokenizer:

    def __init__(self, vocab: List[str], byte_pieces: List[str], bos_id: int):
        self.vocab = vocab
        self.byte_pieces = byte_pieces
        self.bos_id = bos_id


def decode(tokenizer: Tokenizer, prev_token: int, token: int) -> str:
    """
    Python equivalent of the C `decode` function:

        char* decode(Tokenizer* t, int prev_token, int token)

    - Strips leading space if this token follows BOS.
    - Recognizes "<0xXX>" byte‐piece tokens and returns the corresponding raw byte string.
    - Otherwise returns the vocab piece as is.
    """
    piece = tokenizer.vocab[token]

    # Following BOS, strip a single leading space if present
    if prev_token == tokenizer.bos_id and piece.startswith(" "):
        piece = piece[1:]

    # If piece is of the form "<0xHH>", map it through byte_pieces
    m = _HEX_RE.match(piece)
    if m:
        byte_val = int(m.group(1), 16)
        # Return the single-byte string from byte_pieces
        return tokenizer.byte_pieces[byte_val]

    return piece


def main():
    parser = argparse.ArgumentParser(description = "Build Decode test artifacts using vocab.txt and encode outputs")
    parser.add_argument("--vocab-file", required = True, help = "Path to vocab.txt (ID, piece, score per line)")
    parser.add_argument("--encode-out",
                        required = True,
                        help = "Directory of Encode outputs (must contain outputs.npz)")
    parser.add_argument("--bos-id", type = int, required = True, help = "Token ID used as BOS for decode logic")
    parser.add_argument("--eos-id", type = int, required = True, help = "Token ID used as EOS for decode logic")
    parser.add_argument("--max-tokens",
                        type = int,
                        default = 512,
                        help = "Maximum sequence length (pad or truncate token_ids to this)")
    parser.add_argument("--out-dir",
                        default = "decode_out",
                        help = "Output directory for network.onnx, inputs.npz, outputs.npz")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok = True)

    # 1) Load vocabulary from vocab.txt
    pieces = []
    vocab_file_path = args.vocab_file
    with open(vocab_file_path, encoding = "utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(",\t", 2)
            if len(parts) != 3:
                continue
            idx_str, piece_field, _ = parts
            idx = int(idx_str)
            piece = piece_field  # keep spaces/commas intact
            if idx >= len(pieces):
                pieces.extend([""] * (idx + 1 - len(pieces)))
            pieces[idx] = piece

    # 2) Load Encode outputs: token_ids & n_tokens
    enc = np.load(os.path.join(args.encode_out, "outputs.npz"))
    n_tokens = enc["n_tokens"]
    raw_ids = enc["token_ids"]

    prev_ids = np.concatenate([
        [args.bos_id],  # BOS
        raw_ids[:n_tokens - 1]  # Previous token IDs
    ])
    ids = raw_ids[:n_tokens]  # Current token IDs

    # Print the full input arrays
    print(f"prev_ids: {prev_ids}")
    # Print the decoded previous token IDs
    print(f"Decoded prev_ids: {[pieces[i] for i in prev_ids]}")
    print("Length of prev_ids:", len(prev_ids))
    print(f"ids: {ids}")
    # Print the decoded current token IDs
    print(f"Decoded ids: {[pieces[i] for i in ids]}")
    print("Length of ids:", len(ids))

    # 3) Build input arrays
    np.savez(os.path.join(args.out_dir, "inputs.npz"), prev_ids = prev_ids, ids = ids)

    # 4) Implementation of Python decode() function
    byte_pieces = [chr(i) for i in range(256)]
    tokenizer = Tokenizer(pieces, byte_pieces, args.bos_id)

    decoded = []
    for prev_token, token in zip(prev_ids, ids):
        piece = decode(tokenizer, prev_token, token)
        for ch in piece:
            decoded.append(ord(ch))
    out_arr = np.array(decoded, dtype = np.uint8)
    # Print the decoded output as actual string
    decoded_str = "".join(chr(c) for c in out_arr)
    print(f"Decoded output: {decoded_str}")
    np.savez(os.path.join(args.out_dir, "outputs.npz"), decoded_pieces = out_arr)

    print(f"Decoded output shape: {out_arr.shape}")

    # 5) Build and export the ONNX graph with a single Decode node
    # ----------------------------------------------------------------------------
    # Create graph variables
    maxPieceLen = max(len(piece) for piece in pieces)
    print(f"maxPieceLen: {maxPieceLen}")
    prev_var = gs.Variable("prev_ids", dtype = np.int32, shape = [int(n_tokens)])
    curr_var = gs.Variable("ids", dtype = np.int32, shape = [int(n_tokens)])
    out_var = gs.Variable("pieces", dtype = np.uint8, shape = out_arr.shape)
    print(f"Input shape: {prev_var.shape}, {curr_var.shape}")
    print(f"Output shape: {out_var.shape}")

    # Create Decode node
    decode_node = gs.Node(op = "Decode",
                          inputs = [prev_var, curr_var],
                          outputs = [out_var],
                          attrs = {
                              "vocab": pieces,
                              "byte_pieces": byte_pieces,
                              "bos_id": args.bos_id,
                              "eos_id": args.eos_id,
                              "maxPieceLen": maxPieceLen,
                              "nTokens": int(n_tokens)
                          })

    # Create graph
    graph = gs.Graph(nodes = [decode_node], inputs = [prev_var, curr_var], outputs = [out_var])
    graph.cleanup().toposort()
    # Export the graph to ONNX format
    model = gs.export_onnx(graph)
    onnx.save(model, os.path.join(args.out_dir, "network.onnx"))
    print(f"✅ Wrote network.onnx in {args.out_dir}")


if __name__ == "__main__":
    main()
