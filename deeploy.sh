#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Viviane Potocnik <vivianep@iis.ee.ethz.ch>

# Function to prompt user to override an environment variable
prompt_override() {
    local var_name="$1"
    local default_value="$2"
    local current_value="${!var_name}"
    
    if [ -z "$current_value" ]; then
        echo "Variable $var_name is not set. Using default value: $default_value."
        export "$var_name=$default_value"
    else
        echo "Variable $var_name is currently set to: $current_value"
        echo -n "Do you want to override it with the default value ($default_value)? (y/N): "
        read answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            export "$var_name=$default_value"
            echo "$var_name set to $default_value."
        else
            echo "Keeping current value for $var_name."
        fi
    fi
}

# Check if DEEPLOY_INSTALL_DIR is set; if not, prompt the user for it.
if [ -z "$DEEPLOY_INSTALL_DIR" ]; then
    echo "DEEPLOY_INSTALL_DIR is not set."
    echo -n "Please enter the DEEPLOY_INSTALL_DIR path to use for Deeploy defaults: "
    read input_dir
    if [ -z "$input_dir" ]; then
        echo "No DEEPLOY_INSTALL_DIR provided. Exiting."
        exit 1
    fi
    export DEEPLOY_INSTALL_DIR="$input_dir"
fi

echo "Using DEEPLOY_INSTALL_DIR: $DEEPLOY_INSTALL_DIR"
echo "-----------------------------------------"

# Prompt for each tool-specific variable.
prompt_override "PULP_SDK_HOME" "${DEEPLOY_INSTALL_DIR}/pulp-sdk"
prompt_override "LLVM_INSTALL_DIR" "${DEEPLOY_INSTALL_DIR}/llvm"
prompt_override "PULP_RISCV_GCC_TOOLCHAIN" "/PULP_SDK_IS_A_MESS"
prompt_override "MEMPOOL_HOME" "${DEEPLOY_INSTALL_DIR}/mempool"
prompt_override "CMAKE" "/usr/bin/cmake"
prompt_override "QEMU_HOME" "${DEEPLOY_INSTALL_DIR}/qemu"
prompt_override "BANSHEE_HOME" "${DEEPLOY_INSTALL_DIR}/banshee"

echo "-----------------------------------------"
echo "Final environment configuration:"
echo "PULP_SDK_HOME = ${PULP_SDK_HOME}"
echo "LLVM_INSTALL_DIR = ${LLVM_INSTALL_DIR}"
echo "PULP_RISCV_GCC_TOOLCHAIN = ${PULP_RISCV_GCC_TOOLCHAIN}"
echo "MEMPOOL_HOME = ${MEMPOOL_HOME}"
echo "CMAKE = ${CMAKE}"
echo "QEMU_HOME = ${QEMU_HOME}"
echo "BANSHEE_HOME = ${BANSHEE_HOME}"
echo "-----------------------------------------"

# Update PATH by adding directories if not already present.
for dir in "${QEMU_HOME}/bin" "${BANSHEE_HOME}" "$HOME/.cargo/bin"; do
    case ":$PATH:" in
        *":${dir}:"*) ;;  # Directory already in PATH
        *) PATH="${dir}:$PATH" ;;
    esac
done
export PATH

# Source the PULP SDK configuration script if available.
if [ -f "${PULP_SDK_HOME}/configs/siracusa.sh" ]; then
    echo "Sourcing ${PULP_SDK_HOME}/configs/siracusa.sh"
    source "${PULP_SDK_HOME}/configs/siracusa.sh"
else
    echo "Warning: ${PULP_SDK_HOME}/configs/siracusa.sh not found."
fi
