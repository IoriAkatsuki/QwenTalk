#!/usr/bin/env bash
# flash_moe_launch.sh — Flash-MoE 风格 expert offload 启动配置生成器
# 仅产生 launch 命令字符串，不自动重启 llmsrv（用户手动决定时机）
#
# 模型: Qwen3.6-35B-A3B Q4_K_S, 41 layers, 256 routed experts
# 关键 tensor: blk.{0..40}.ffn_{gate,up,down}_exps.weight
#
# 用法:
#   ./flash_moe_launch.sh             # 显示三档 profile 推荐命令
#   ./flash_moe_launch.sh A           # 输出 Profile A (fit-UMA 当前)
#   ./flash_moe_launch.sh B           # 输出 Profile B (释放 ~2.4GB)
#   ./flash_moe_launch.sh C           # 输出 Profile C (flash-stream)
#   ./flash_moe_launch.sh --dry-run B # 输出命令 + meminfo 估算
#
# 不会执行 llama-server，只 echo 命令

set -u

LLAMA_BIN=${LLAMA_BIN:-/home/intel/llama_mtp/build_sycl/bin}
MODEL=${MODEL:-/home/intel/models/qwen3.6-mtp-q4ks.gguf}
PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}

# 当前 llmsrv 已知最优基线
COMMON_FLAGS="-m $MODEL -ngl 99 -fa 1 -ub 64 -t 2 -ctk f16 -ctv f16 --port $PORT --host $HOST"

# 三档 profile 的 -ot 正则
# A: 无 offload，全 GPU
RE_A=""
# B: 仅末 11 层 (30..40) experts 走 CPU，释放 ~2.4GB
RE_B='blk\.(3[0-9]|40)\.ffn_.*_exps\.weight=CPU'
# C: 前 5 层 experts 留 GPU，其余 36 层走 CPU，释放 ~7.2GB（为 Q5_K_M 25GB 预演）
RE_C='blk\.([5-9]|[12][0-9]|3[0-9]|40)\.ffn_.*_exps\.weight=CPU'

print_meminfo() {
    echo "--- 当前内存 ---"
    grep -E "MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree" /proc/meminfo 2>/dev/null | head -10
    if command -v free >/dev/null 2>&1; then
        echo ""
        free -h
    fi
    echo ""
}

estimate_offload() {
    # Q4_K_S 单 expert tensor (256 experts 打包) ≈ 67 MB
    # 单层 3 tensors = 201 MB
    local profile=$1
    case "$profile" in
        A) echo "  → GPU expert 占用: ~8.2 GB (全 41 层 × 201 MB)" ;;
        B) echo "  → GPU expert 占用: ~6.0 GB；CPU 接管 11 层 × 201 MB = 2.2 GB" ;;
        C) echo "  → GPU expert 占用: ~1.0 GB；CPU 接管 36 层 × 201 MB = 7.2 GB" ;;
    esac
}

build_cmd() {
    local profile=$1
    local regex=""
    case "$profile" in
        A) regex="$RE_A" ;;
        B) regex="$RE_B" ;;
        C) regex="$RE_C" ;;
        *) echo "未知 profile: $profile" >&2; return 1 ;;
    esac

    local cmd="$LLAMA_BIN/llama-server $COMMON_FLAGS"
    if [ -n "$regex" ]; then
        cmd="$cmd --override-tensor '$regex'"
    fi
    echo "$cmd"
}

show_profile() {
    local p=$1
    local desc=""
    case "$p" in
        A) desc="fit-UMA baseline (当前 llmsrv 配置，无 expert offload)" ;;
        B) desc="spillover-late: 末 11 层 expert → CPU，释放 ~2.4GB GPU/UMA" ;;
        C) desc="flash-stream: 仅前 5 层 GPU expert，其余 → CPU/mmap (Q5_K_M 预演)" ;;
    esac
    echo "=========================================="
    echo "Profile $p: $desc"
    echo "=========================================="
    estimate_offload "$p"
    echo ""
    echo "命令:"
    build_cmd "$p"
    echo ""
}

main() {
    local profile=""
    local dry_run=0

    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run) dry_run=1 ;;
            A|B|C) profile=$1 ;;
            -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
            *) echo "未知参数: $1" >&2; exit 1 ;;
        esac
        shift
    done

    if [ -z "$profile" ]; then
        echo "Flash-MoE 启动配置三档（Qwen3.6-35B-A3B Q4_K_S）"
        echo ""
        show_profile A
        show_profile B
        show_profile C
        echo "提示: 跑 './flash_moe_launch.sh <A|B|C>' 单独输出，加 --dry-run 含内存估算"
        echo "不会自动启动 llama-server，请手动 stop llmsrv 后执行选定命令"
        return 0
    fi

    if [ "$dry_run" -eq 1 ]; then
        print_meminfo
    fi

    show_profile "$profile"

    if [ "$dry_run" -eq 1 ]; then
        echo "--- 启动前检查清单 ---"
        echo "  [ ] swap 已 swapoff/swapon 重置（防 5GB swap 干扰）"
        echo "  [ ] llmsrv 已 systemctl stop（避免端口冲突）"
        echo "  [ ] echo 3 > /proc/sys/vm/drop_caches（测冷启动）"
        echo "  [ ] iostat -xmt 1 sda 后台监控 (PC801 IO)"
        echo ""
    fi
}

main "$@"
