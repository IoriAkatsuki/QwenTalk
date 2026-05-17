#!/bin/bash
# AI 工作负载启动器 — 按 cgroup slice 分配前后台
# 用法: ai-launcher.sh [inference|perception|background] <command>

SLICE="${1:?用法: $0 [inference|perception|background] <command>}"
shift
CMD="$@"

case "$SLICE" in
    inference)
        # P-core 0,1 + 高优先级 + nice -10
        echo "启动 [推理前台] CPU 0-1: $CMD"
        systemd-run --slice=ai-inference.slice --scope \
            nice -n -10 taskset -c 0,1 "$@"
        ;;
    perception)
        # E-core 2-9 + 中等优先级
        echo "启动 [感知管线] CPU 2-9: $CMD"
        systemd-run --slice=ai-perception.slice --scope \
            taskset -c 2-9 "$@"
        ;;
    background)
        # LP E-core 10-13 + 低优先级
        echo "启动 [后台任务] CPU 10-13: $CMD"
        systemd-run --slice=ai-background.slice --scope \
            nice -n 10 taskset -c 10-13 "$@"
        ;;
    *)
        echo "未知 slice: $SLICE"
        echo "用法: $0 [inference|perception|background] <command>"
        exit 1
        ;;
esac
