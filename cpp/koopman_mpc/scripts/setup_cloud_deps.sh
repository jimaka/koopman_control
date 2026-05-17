#!/usr/bin/env bash
# 云端/CI 缺失 C++ 工具链时安装构建依赖
set -euo pipefail
if command -v g++ >/dev/null 2>&1; then
    if echo 'int main(){}' | g++ -x c++ - -o /tmp/_koopman_gxx_test 2>/dev/null; then
        rm -f /tmp/_koopman_gxx_test
        echo "g++ OK"
        exit 0
    fi
fi
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential g++ cmake curl
fi
