#!/usr/bin/env python3
"""冒烟测试：在指定设备上加载模型，跑一次对话，测延迟和吞吐。"""
import argparse
import time
import openvino_genai as ov_genai


def streamer(subword: str) -> bool:
    print(subword, end="", flush=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-d", "--device", default="GPU", choices=["CPU", "GPU", "NPU"])
    parser.add_argument("-p", "--prompt", default="用中文一句话介绍英特尔 Core Ultra 处理器。")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"[INFO] Loading {args.model} on {args.device} ...")
    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(args.model, args.device)
    t_load = time.perf_counter() - t0
    print(f"[INFO] Loaded in {t_load:.2f}s")

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = args.max_tokens
    cfg.do_sample = True
    cfg.temperature = 0.7
    cfg.top_p = 0.9

    print(f"\n👤 {args.prompt}\n🤖 ", end="", flush=True)
    t1 = time.perf_counter()
    pipe.start_chat()
    result = pipe.generate(args.prompt, cfg, streamer)
    t_gen = time.perf_counter() - t1
    pipe.finish_chat()

    n_tokens = len(result.tokens[0]) if hasattr(result, "tokens") else args.max_tokens
    print(f"\n\n[STATS] generate: {t_gen:.2f}s, ~{n_tokens / t_gen:.1f} tok/s (rough)")


if __name__ == "__main__":
    main()
