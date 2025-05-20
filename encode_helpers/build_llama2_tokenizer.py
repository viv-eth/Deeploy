#!/usr/bin/env python3
import argparse
import math
import os
import re
import struct

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from sentencepiece import SentencePieceProcessor

_hex_fallback = re.compile(r"^<0x([0-9A-F]{2})>$")


def format_c_string_content(s: str) -> str:
    """
    Formats a string's content into a Python string representing the
    literal characters needed inside C string literal quotes.
    Handles specific tokens and escapes necessary characters.
    """
    # Handle <0xNN> token - produces literal characters '\xNN'
    m = _hex_fallback.match(s)
    if m:
        hex_val = m.group(1)
        return f"\\x{hex_val}"  # Python string '\\xNN' for literal '\xNN' in C

    out = []
    for b in s.encode("utf-8"):
        # Literal printable ASCII chars (except " and \)
        if 0x20 <= b <= 0x7E and b not in (0x5C, 0x22):
            out.append(chr(b))
        # C standard escapes for special characters
        elif b == 0x22:  # double-quote
            out.append(r'\"')  # Python string '\\""' for literal '\"' in C
        elif b == 0x5C:  # backslash
            out.append(r'\\')  # Python string '\\\\' for literal '\\' in C
        elif b == 0x0A:  # newline
            out.append(r'\n')  # Python string '\\n' for literal '\n' in C
        elif b == 0x0D:  # carriage return
            out.append(r'\r')  # Python string '\\r' for literal '\r' in C
        # Hex escape for all other bytes
        else:
            # Python string '\\xNN' for literal '\xNN' in C
            out.append(f"\\x{b:02x}")
    return "".join(out)


# Helper function to get raw bytes from token ID (for Python gold standard logic)
def get_piece_bytes_from_id(token_id, pieces, byte_fallback_offset):
    if token_id >= byte_fallback_offset and token_id < byte_fallback_offset + 256:
        # It's a byte fallback ID
        byte_val = token_id - byte_fallback_offset
        return bytes([byte_val])
    elif 0 <= token_id < len(pieces):
        # Lookup the piece string by original ID and encode to bytes
        return pieces[token_id].encode('utf-8')
    else:
        # Should not happen with correct logic, but handle defensively
        # print(f"Warning: get_piece_bytes_from_id received invalid token_id: {token_id}")
        return b""  # Return empty bytes for invalid ID


# Python implementation of the Unigram tokenization logic for gold standard output
# This simulates the two-phase process of the C encode function (initial lookup/fallback + merges).
def tokenize_unigram(text, pieces, scores, unk_id, byte_fallback_offset, prefix_id, max_piece_len, bos, eos):
    # Map raw piece (bytes) to its ID and score for efficient lookup during tokenization
    piece_bytes_map = {p.encode('utf-8'): (i, scores[i]) for i, p in enumerate(pieces)}

    gold_tokens = []

    # Add optional BOS (=1) token, if desired
    if bos:
        # Assumes BOS ID is 1 based on common SentencePiece usage and your C code.
        gold_tokens.append(1)  # Assuming BOS ID is 1

    # Add optional dummy prefix token (" "), if desired and text is not empty
    # Simulate the C code's logic: add prefix if text is not empty using the determined PREFIX_ID.
    if text and text[0] != '\0':
        # Verify that the piece for PREFIX_ID is indeed a space, as expected by the C code.
        prefix_piece_str = ""
        if prefix_id >= 0 and prefix_id < len(pieces):
            prefix_piece_str = pieces[prefix_id]

        if prefix_piece_str == " ":  # Add prefix if the piece for PREFIX_ID is indeed a space
            gold_tokens.append(prefix_id)
        else:
            # This case might indicate an issue with the loaded vocabulary or prefix_id assumption.
            # print(f"Warning: Piece for PREFIX_ID {prefix_id} is '{prefix_piece_str}', not ' '. Skipping prefix token.")
            pass  # Follow the C code's logic based on PREFIX_ID

    # Step 1: Initial tokenization (codepoint or byte fallback)
    # Simulate the C code's byte-by-byte accumulation and lookup logic.
    text_bytes = text.encode('utf-8')
    i = 0
    while i < len(text_bytes):
        str_buffer_bytes = b""
        str_len = 0  # Length in bytes of the current buffer
        c = text_bytes[i]  # Get the current byte

        # Accumulate the current byte
        str_buffer_bytes += bytes([c])
        str_len += 1
        i += 1  # Move to the next byte position in the input text

        # Continue accumulating bytes while the *next* byte is a continuation byte AND the buffer
        # length is less than the maximum expected length for an initial piece/character (often 4 for UTF-8,
        # or max_piece_len in your original C code logic). The encode function used str_len < 4.
        while i < len(text_bytes) and ((text_bytes[i] & 0xC0)
                                       == 0x80) and str_len < 4:  # Using 4 as per encode function's limit
            str_buffer_bytes += bytes([text_bytes[i]])
            str_len += 1
            i += 1

        # At this point, str_buffer_bytes contains the bytes for a potential initial token.
        # This sequence ends because the next byte was not a continuation, or the buffer reached 4 bytes.

        # Look up this exact byte sequence in the vocabulary pieces (using the byte map).
        lookup_info = piece_bytes_map.get(str_buffer_bytes)

        if lookup_info is not None:
            # Found a match in the vocabulary
            token_id, _ = lookup_info
            gold_tokens.append(token_id)
        else:
            # No direct vocabulary match for the accumulated bytes, perform byte fallback
            for b in str_buffer_bytes:
                gold_tokens.append(b + byte_fallback_offset)

    # Step 2: Merge best pairs (Unigram greedy merge)
    # This simulates the C code's merge loop.
    current_tokens = list(gold_tokens)  # Start with the initial tokens

    while True:
        best_score = -math.inf  # Use math.inf for negative infinity
        best_id = -1
        best_idx = -1
        T = len(current_tokens)  # Current number of tokens

        if T < 2:
            break  # Cannot merge if less than 2 tokens

        for i in range(T - 1):
            id1 = current_tokens[i]
            id2 = current_tokens[i + 1]

            # Get the raw byte sequences for the two tokens being considered for merge.
            # Handle byte fallback IDs correctly when retrieving bytes.
            piece1_bytes = get_piece_bytes_from_id(id1, pieces, byte_fallback_offset)
            piece2_bytes = get_piece_bytes_from_id(id2, pieces, byte_fallback_offset)
            merged_bytes = piece1_bytes + piece2_bytes

            # Ensure the merged string is not excessively long before lookup.
            # The limit is usually the maximum piece length defined by the tokenizer.
            if len(merged_bytes) > max_piece_len:
                continue  # Skip this merge candidate

            # Look up the concatenated byte sequence in the vocabulary pieces map.
            lookup_info = piece_bytes_map.get(merged_bytes)

            if lookup_info is not None:
                merge_id, sc = lookup_info[0], lookup_info[1]
                # The greedy strategy is to pick the merge with the highest score.
                if sc > best_score:
                    best_score = sc
                    best_id = merge_id
                    best_idx = i  # Store the index of the first token in the pair

        if best_idx < 0:
            break  # No valid merges were found in this iteration with a score > -infinity

        # Perform the merge: Replace the pair (at best_idx and best_idx+1) with the merged token ID.
        current_tokens[best_idx] = best_id
        # Remove the second token of the pair by popping it.
        current_tokens.pop(best_idx + 1)
        # The list length (and token count) decreases by 1 automatically.

    # Add optional EOS (=2) token, if desired
    if eos:
        # Assumes EOS ID is 2 based on common SentencePiece usage and your C code.
        gold_tokens.append(2)  # Assuming EOS ID is 2

    return current_tokens


def main():
    p = argparse.ArgumentParser(description = "Build an ONNX graph with a single custom Encode node (Unigram-style).")
    p.add_argument("bin_path", help = "SentencePiece tokenizer .bin")
    p.add_argument("-s", "--spm-model", default = "tokenizer.model", help = "SentencePiece .model file")
    p.add_argument("-O", "--out-dir", default = ".", help = "Where to write network.onnx, inputs.npz, outputs.npz")
    p.add_argument("-t", "--test-string", default = "Blue Kitty!", help = "Example text to tokenize")
    p.add_argument("-b", "--byte-fallback-offset", type = int, default = 3, help = "Offset for byte fallback tokens")
    p.add_argument("-m", "--max-tokens", type = int, default = 512, help = "Maximum number of output tokens")
    p.add_argument("--add-bos", action = "store_true", help = "Add a beginning-of-sequence (BOS) token")
    p.add_argument("--add-eos", action = "store_true", help = "Add an end-of-sequence (EOS) token")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok = True)

    # 1) mmap in the .bin header + entries exactly as build_tokenizer() would
    pieces = []
    scores = []
    with open(args.bin_path, "rb") as f:
        # 1a) header: max piece length
        hdr = f.read(4)
        if len(hdr) < 4:
            raise RuntimeError("`.bin` too short to contain header")
        max_piece_len = struct.unpack("<I", hdr)[0]
        L = max_piece_len

        # figure out how many bytes remain
        total = os.fstat(f.fileno()).st_size
        pos = f.tell()

        # 1b) read until EOF
        while pos < total:
            sb = f.read(4)
            if len(sb) < 4:
                raise RuntimeError(f"unexpected EOF reading score at offset {pos}")
            sc = struct.unpack("<f", sb)[0]

            lb = f.read(4)
            if len(lb) < 4:
                raise RuntimeError(f"unexpected EOF reading length at offset {pos+4}")
            length = struct.unpack("<i", lb)[0]

            rb = f.read(length)
            if len(rb) < length:
                raise RuntimeError(f"unexpected EOF reading piece@{pos+8}")
            piece = rb.decode("utf-8")

            scores.append(sc)
            pieces.append(piece)

            pos = f.tell()

    V = len(pieces)
    print(f"Loaded {V} tokens from .bin (max_piece_len={L})")
    try:
        prefix_id = pieces.index(" ")
    except ValueError:
        raise RuntimeError("space-prefix token not found in .bin vocabulary")

    # 2) SentencePiece only to get unk_id & prefix_id
    sp = SentencePieceProcessor(model_file = args.spm_model)
    unk_id = sp.unk_id()

    # 3) sort lexicographically by raw text
    raw = list(zip(pieces, range(len(pieces)), scores))
    raw.sort(key = lambda x: x[0])

    vocabPieces = [format_c_string_content(p) for p, _, _ in raw]
    # # save vocabPieces to a .txt file
    # with open(os.path.join(args.out_dir, "vocab_pieces.txt"), "w", encoding="utf-8") as f:
    #     for piece in vocabPieces:
    #         f.write(f"{piece}\n")
    print("Wrote vocab pieces →", os.path.join(args.out_dir, "vocab_pieces.txt"))
    vocabIds = [np.int64(i) for _, i, _ in raw]
    vocabScores = [float(s) for *_, s in raw]

    # 4) build ONNX graph
    inp = gs.Variable("input_bytes", dtype = np.uint8, shape = (args.max_tokens,))
    out_ids = gs.Variable("token_ids", dtype = np.int32, shape = (args.max_tokens,))
    out_n = gs.Variable("n_tokens", dtype = np.int32, shape = ())

    encode = gs.Node(op = "Encode",
                     name = "encode_node",
                     inputs = [inp],
                     outputs = [out_ids, out_n],
                     attrs = {
                         "vocabSize": np.int64(V),
                         "maxPieceLen": np.int64(max_piece_len),
                         "byteFallbackOffset": np.int64(3),
                         "unkId": np.int64(unk_id),
                         "prefixId": np.int64(prefix_id),
                         "vocabPieces": vocabPieces,
                         "vocabIds": vocabIds,
                         "vocabScores": vocabScores,
                     })

    graph = gs.Graph(nodes = [encode], inputs = [inp], outputs = [out_ids, out_n]).cleanup().toposort()

    model = gs.export_onnx(graph)
    onnx.save(model, os.path.join(args.out_dir, "network.onnx"))
    print("Wrote ONNX →", os.path.join(args.out_dir, "network.onnx"))

    # 5) dump inputs
    raw_bytes = args.test_string.encode("utf-8")[:args.max_tokens]
    in_arr = np.zeros((args.max_tokens,), dtype = np.uint8)
    in_arr[:len(raw_bytes)] = np.frombuffer(raw_bytes, dtype = np.uint8)
    np.savez(os.path.join(args.out_dir, "inputs.npz"), input_bytes = in_arr)
    print("Wrote inputs →", os.path.join(args.out_dir, "inputs.npz"))

    # # 6) gold tokens via same codepoint+fallback
    # vmap = {p: i for p, i, _ in raw}
    # toks = []
    # for ch in args.test_string:
    #     if ch in vmap:
    #         toks.append(vmap[ch])
    #     else:
    #         for b in ch.encode("utf-8"):
    #             toks.append(b + 3)
    # out_arr = np.zeros((L,), dtype=np.int32)
    # out_arr[:len(toks)] = toks
    # np.savez(os.path.join(args.out_dir, "outputs.npz"), token_ids=out_arr)
    # print("Wrote gold →", os.path.join(args.out_dir, "outputs.npz"))

    # 6) Generate gold tokens using the full Unigram tokenization algorithm (initial + merges)
    # This is the core change: using the tokenize_unigram function to get the correct expected output.
    gold_tokens = tokenize_unigram(
        args.test_string,
        pieces,  # original pieces list (strings)
        scores,  # original scores list (floats)
        unk_id,
        args.byte_fallback_offset,
        prefix_id,
        max_piece_len,  # Pass max_piece_len
        args.add_bos,  # Pass BOS flag
        args.add_eos  # Pass EOS flag
    )

    print(f"Generated gold tokens for '{args.test_string}': {gold_tokens}")

    # Save gold tokens into outputs.npz, padding with zeros or truncating to MAX_OUTPUT_TOKENS
    out_arr = np.zeros((args.max_tokens,), dtype = np.int32)
    if len(gold_tokens) > args.max_tokens:
        print(
            f"Warning: Gold token count ({len(gold_tokens)}) exceeds MAX_OUTPUT_TOKENS ({args.max_tokens}). Truncating gold output in NPZ."
        )
        out_arr[:args.max_tokens] = gold_tokens[:args.max_tokens]
        n_gold_tokens = args.max_tokens  # Report truncated count
    else:
        out_arr[:len(gold_tokens)] = gold_tokens
        n_gold_tokens = len(gold_tokens)  # Report actual count

    outputs_npz_path = os.path.join(args.out_dir, "outputs.npz")
    # Save both the token IDs array and the actual number of tokens.
    np.savez(outputs_npz_path, token_ids = out_arr, n_tokens = np.array(n_gold_tokens, dtype = np.int32))
    print("Wrote outputs →", outputs_npz_path)


if __name__ == "__main__":
    main()
