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
    flat arrays for sorted vocab, scores & reverse mapping
   —————————————————————— */

#define N ${vocabSize}

/* sorted list of (piece, id) for blind scan */
static const char *sorted_vocab_str[N] = {
% for lit, idx in vocabItems:
    ${lit}${"," if not loop.last else ""}
% endfor
};
static const int sorted_vocab_id[N] = {
% for lit, idx in vocabItems:
    ${idx}${"," if not loop.last else ""}
% endfor
};

/* merge scores */
static const float vocab_scores[N] = {
% for sc in vocabScores:
    ${"%.9g" % sc}${"," if not loop.last else ""}
% endfor
};

/* reverse lookup from id→string */
static const char *vocab_by_id[N] = {
% for piece in vocabPiecesById:
    ${piece}${"," if not loop.last else ""}
% endfor
};

BEGIN_SINGLE_CORE

    /* local flat buffer large enough for worst-case (seqLen entries) */
    int flat_ids[${sequenceLength}];
    *${n_tokens} = 0;

    /* 1) optional BOS */
    if (${bosName}) {
        flat_ids[(*${n_tokens})++] = ${bosId};
    }

    /* 2) optional dummy-prefix (space) */
    if (*${inputBytes} != '\0') {
        flat_ids[(*${n_tokens})++] = ${prefixId};
    }

    /* 3) greedy longest-match tokenization with byte-fallback */
    {
        char *str_buffer = malloc((${maxPieceLen}*2 + 3) * sizeof(char));
        size_t str_len = 0;
        for (const unsigned char *c = (const unsigned char*)${inputBytes}; *c; ++c) {
            if (((*c) & 0xC0) != 0x80) {
                str_len = 0;
            }
            str_buffer[str_len++] = (char)*c;
            str_buffer[str_len]   = '\0';
            if ((((unsigned char)*(c+1)) & 0xC0) == 0x80 && str_len < 4) {
                continue;
            }

            /* inline str_lookup */
            int id = -1;
            for (int i = 0; i < N; ++i) {
                if (strcmp(str_buffer, sorted_vocab_str[i]) == 0) {
                    id = sorted_vocab_id[i];
                    break;
                }
            }

            if (id >= 0 && id != ${unkId}) {
                flat_ids[(*${n_tokens})++] = id;
            } else {
                for (size_t j = 0; j < str_len; ++j) {
                    unsigned char b = (unsigned char)str_buffer[j];
                    int outb = b + ${byteFallbackOffset};
                    flat_ids[(*${n_tokens})++] = outb;
                }
            }
        }
        free(str_buffer);
    }

    /* 4) score-based pairwise merging */
    {
        char *str_buffer = malloc((${maxPieceLen}*2 + 3) * sizeof(char));
        while (1) {
            float best_score = -INFINITY;
            int   best_id    = -1;
            int   best_idx   = -1;
            int   T          = *${n_tokens};
            if (T < 2) {
                break;
            }

            for (int i = 0; i < T-1; ++i) {
                sprintf(str_buffer, "%s%s",
                        vocab_by_id[ flat_ids[i] ],
                        vocab_by_id[ flat_ids[i+1] ]);
                int m = -1;
                for (int k = 0; k < N; ++k) {
                    if (strcmp(str_buffer, sorted_vocab_str[k]) == 0) {
                        m = sorted_vocab_id[k];
                        break;
                    }
                }
                if (m >= 0 && vocab_scores[m] > best_score) {
                    best_score = vocab_scores[m];
                    best_id    = m;
                    best_idx   = i;
                }
            }

            if (best_idx < 0) {
                break;
            }

            flat_ids[best_idx] = best_id;
            for (int j = best_idx+1; j < T-1; ++j) {
                flat_ids[j] = flat_ids[j+1];
            }
            (*${n_tokens})--;
        }
        free(str_buffer);
    }

    /* 5) optional EOS */
    if (${eosName}) {
        flat_ids[(*${n_tokens})++] = ${eosId};
    }

    /* 6) finally copy exactly n_tokens entries into the real ONNX output */
    for (int i = 0; i < *${n_tokens}; ++i) {
        ${tokenIds}[i] = flat_ids[i];
    }

END_SINGLE_CORE
""")
