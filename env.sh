#!/bin/bash
# ==========================================
# 安全 pip 安装脚本（遇错跳过 + 记录日志）
# 使用方法: ./safe_pip_install.sh requirements.txt
# ==========================================

REQ_FILE=$1
LOG_FILE="install_log_$(date +%Y%m%d_%H%M%S).txt"

if [ -z "$REQ_FILE" ]; then
    echo "❌ 请输入 requirements.txt 文件路径"
    echo "用法: ./safe_pip_install.sh requirements.txt"
    exit 1
fi

echo "📦 开始逐包安装，日志保存到: $LOG_FILE"
echo "======================================" > "$LOG_FILE"

FAILED_LIST=()

while read -r pkg || [ -n "$pkg" ]; do
    if [[ -z "$pkg" || "$pkg" == \#* ]]; then
        continue  # 跳过空行和注释
    fi
    echo "🚀 正在安装: $pkg"
    pip install "$pkg" --default-timeout=100 --retries 3 >>"$LOG_FILE" 2>&1
    if [ $? -ne 0 ]; then
        echo "⚠️ 安装失败: $pkg"
        FAILED_LIST+=("$pkg")
        echo "[FAILED] $pkg" >> "$LOG_FILE"
    else
        echo "✅ 成功安装: $pkg"
        echo "[OK] $pkg" >> "$LOG_FILE"
    fi
done < "$REQ_FILE"

echo
echo "======================================"
echo "📊 安装完成，日志文件: $LOG_FILE"
if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo "❌ 以下包安装失败:"
    for p in "${FAILED_LIST[@]}"; do
        echo "  - $p"
    done
else
    echo "🎉 所有包安装成功！"
fi
