# ----------------------------------------------------------------------
#
# File: EncodeTemplate.py
#
# Last edited: 05.05.2025
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Author: Viviane Potocnik (vivianep@iis.ee.ethz.ch), ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate(r"""

/* ——————————————————————
    sorted_vocab & scores
   —————————————————————— */

typedef struct { const char *str; int id; } TokenIndex;

// actual number of valid entries
#define N ${vocabSize}

static const TokenIndex sorted_vocab[N] = {
% for lit, idx in vocabItems:
    {${lit}, ${idx} }${"," if not loop.last else ""}
% endfor
};

static const float vocab_scores[N] = {
% for sc in vocabScores:
    ${"%.9g" % sc}${"," if not loop.last else ""}
% endfor
};

static const char *vocab_by_id[N] = {
% for piece in vocabPiecesById:
    ${piece}${"," if not loop.last else ""}
% endfor
};

// Debug printing macro: enable by defining ENCODE_DEBUG
#ifndef ENCODE_DEBUG
    #define DEBUG_PRINT(...) ((void)0)
#else
    #define DEBUG_PRINT(...) printf(__VA_ARGS__)
#endif

BEGIN_SINGLE_CORE

  /* Variables needed by both phases or the outer scope */
  char str_buffer[${maxPieceLen*2+3}]; // Buffer for merge operations (twice the max piece length plus safety)
  char *text = (char *)${input_bytes}; /* C‐string input */
  int current_token_count = 0; // Counter for output tokens
  char *p = ${input_bytes}; // Pointer to the current position in the input string
  bool is_first_token = true; // Flag to handle the special first token prefix rule

    /* —— 1) Greedy Longest Match Segmentation with First Token Prefix —— */
    while (*p != '\0' && current_token_count < 1024) {
        int best_match_id = -1;
        size_t best_match_len = 0; // Length in bytes of the input substring matched

        // Max substring length to check
        size_t max_check_len = ${maxPieceLen};

        for (size_t len = 1; len <= max_check_len; ++len) {
            if (*(p + len - 1) == '\0' && len > 1) break;
            char lookup_buffer[${maxPieceLen*2+3} + 1];
            size_t lookup_len = 0;
            int current_id = -1;
            if (is_first_token) {
                if (${prefixId} < 0 || ${prefixId} >= N) {
                    DEBUG_PRINT("Warning: prefix token ID %d out of bounds (size %d)\n", ${prefixId}, N);
                    break;
                }
                const char *prefix_str = vocab_by_id[${prefixId}];
                size_t prefix_len = strlen(prefix_str);
                if (prefix_len + len < sizeof(lookup_buffer)) {
                    memcpy(lookup_buffer, prefix_str, prefix_len);
                    memcpy(lookup_buffer + prefix_len, p, len);
                    lookup_len = prefix_len + len;
                    lookup_buffer[lookup_len] = '\0';
                    current_id = str_lookup(lookup_buffer, (const TokenIndex *)sorted_vocab, N);
                }
            }
            int current_id_no_prefix = -1;
            if (len < sizeof(lookup_buffer)) {
                memcpy(lookup_buffer, p, len);
                lookup_len = len;
                lookup_buffer[lookup_len] = '\0';
              current_id_no_prefix = str_lookup(lookup_buffer, (const TokenIndex *)sorted_vocab, N);
            }
            if (is_first_token && current_id >= 0) {
                if (len > best_match_len) {
                    best_match_id = current_id;
                    best_match_len = len;
                }
            } else if (current_id_no_prefix >= 0) {
                if (len > best_match_len) {
                    best_match_id = current_id_no_prefix;
                    best_match_len = len;
                }
            }
            if (*(p + len - 1) == '\0') break;
        }

        if (best_match_id != -1) {
            if (best_match_id >= 0 && best_match_id < N) {
                DEBUG_PRINT("Greedy match '%s' -> id=%d\n", vocab_by_id[best_match_id], best_match_id);
            }
            ${token_ids}[current_token_count++] = best_match_id;
            p += best_match_len;
            is_first_token = false;
        } else {
            unsigned char b = (unsigned char)*p;
            int fb = b + ${byteFallbackOffset};
            DEBUG_PRINT("byte-fallback '%c' -> id=%d\n", b, fb);
            ${token_ids}[current_token_count++] = fb;
            p += 1;
            is_first_token = false;
        }
    }
  *${n_tokens} = current_token_count;

  /* —— 2) merge-best-pair loop —— */
    while (1) {
        float best_score = -INFINITY;
        int best_id = -1, best_idx = -1;
        int T = *${n_tokens};
        DEBUG_PRINT("--- merge iteration, n_tokens=%d\n", T);
        if (T < 2) {
            DEBUG_PRINT("no more merges (less than 2 tokens)\n");
            break;
        }
        for (int i = 0; i < T - 1; ++i) {
            if (${token_ids}[i] < 0 || ${token_ids}[i] >= N || ${token_ids}[i+1] < 0 || ${token_ids}[i+1] >= N) {
                DEBUG_PRINT("trying merge invalid tokens: id1=%d, id2=%d\n", ${token_ids}[i], ${token_ids}[i+1]);
                continue;
            }
            const char *a = vocab_by_id[${token_ids}[i]];
            const char *b = vocab_by_id[${token_ids}[i + 1]];
            DEBUG_PRINT("trying merge '%s' + '%s'\n", a, b);
            size_t la = strlen(a), lb = strlen(b);
            if (la + lb >= sizeof(str_buffer)) {
                DEBUG_PRINT("  concat '%s%s' too long for buffer (size %zu)\n", a, b, sizeof(str_buffer));
                continue;
            }
            memcpy(str_buffer, a, la);
            memcpy(str_buffer + la, b, lb);
            str_buffer[la + lb] = '\0';
            int merge_id = str_lookup(str_buffer, (const TokenIndex *)sorted_vocab, N);
            DEBUG_PRINT("  concat '%s' -> id=%d\n", str_buffer, merge_id);
            if (merge_id >= 0 && merge_id < N) {
                float sc = vocab_scores[merge_id];
                DEBUG_PRINT("  score[%d]=%f\n", merge_id, sc);
                if (sc > best_score) {
                    best_score = sc;
                    best_id    = merge_id;
                    best_idx   = i;
                }
            } else {
                DEBUG_PRINT("  Warning: merge_id %d out of bounds (size %d)\n", merge_id, N);
            }
        }
        if (best_idx < 0) {
            DEBUG_PRINT("no more merges (no valid pair found)\n");
            break;
        }
        DEBUG_PRINT("perform merge at idx=%d, id=%d, score=%f\n", best_idx, best_id, best_score);
        ${token_ids}[best_idx] = best_id;
        for (int j = best_idx + 1; j < T - 1; ++j) {
            ${token_ids}[j] = ${token_ids}[j + 1];
        }
        (*${n_tokens})--;
    }
    DEBUG_PRINT("Final concated string: ");
    int T = *${n_tokens};
    for (int i = 0; i < T; ++i) {
        if (${token_ids}[i] >= 0 && ${token_ids}[i] < N) {
            DEBUG_PRINT("%s ", vocab_by_id[${token_ids}[i]]);
        } else {
            DEBUG_PRINT("[INVALID_TOKEN_ID:%d] ", ${token_ids}[i]);
        }
    }
    DEBUG_PRINT("\nFinal tokens: ");
    for (int i = 0; i < T; ++i) {
        if (${token_ids}[i] >= 0 && ${token_ids}[i] < N) {
            DEBUG_PRINT("%d ", ${token_ids}[i]);
        } else {
            DEBUG_PRINT("%d (INVALID) ", ${token_ids}[i]);
        }
    }
    DEBUG_PRINT("\n");

END_SINGLE_CORE

""")
