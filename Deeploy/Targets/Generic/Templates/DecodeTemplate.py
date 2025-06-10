# ----------------------------------------------------------------------
#
# File: DecodeTemplate.py
#
# Last edited: 20.05.2025
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Author: Viviane Potocnik, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate(r"""

/* ——————————————————————
    vocab pieces & byte table
   —————————————————————— */

#define V ${vocabSize}

static const unsigned char byte_pieces[512] = {
% for b in bytePiecesFlat:
    ${b}${"," if not loop.last else ""}
% endfor
};

BEGIN_SINGLE_CORE

    size_t off = 0;

    // only iterate over the actual token count, not the padded max
    for (int i = 0; i < ${nTokens}; ++i) {
        int32_t prev = ${prevIds}[i];
        int32_t tok  = ${ids}[i];
        if (tok == ${eosId}) break;
        const char* piece = vocab_by_id[tok];

        /* strip single leading space after BOS */
        if (prev == ${bosId} && piece[0] == ' ') {
            piece += 1;
        }

        /* raw‐byte tokens (“<0xHH>”) */
        {
            unsigned char byte_val;
            if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
                piece = (const char*)(byte_pieces + byte_val * 2);
            }
        }

        /* flatten every character of this piece into the output */
        while (*piece) {
            ${outPieces}[off++] = (uint8_t)*piece++;
        }
    }

    // print the decoded string once
    for (size_t j = 0; j < off; ++j) {
        printf("%c", ${outPieces}[j]);
    }
    printf("\n");
    ${outPieces}[off] = '\0'; // Null‐terminate the output string

END_SINGLE_CORE

""")