import openvino_genai as ov_genai
import time
import sys

def test_llm_with_genai():
    llm_model_path = "Qwen3-8B-nf4-ov"
    
    # 测试不同的提示词
    prompts = [
        "你好，请用中文介绍一下你自己",
        "请解释一下人工智能的基本概念",
        "写一个关于春天的小诗"
    ]
    
    print("正在初始化LLM模型...")
    print(f"模型路径: {llm_model_path}")
    
    try:
        # 首先尝试NPU
        print("尝试在NPU上加载模型...")
        pipe = ov_genai.LLMPipeline(llm_model_path, "NPU")
        device = "NPU"
        print("✓ NPU加载成功!")
        
    except Exception as npu_error:
        print(f"✗ NPU加载失败: {npu_error}")
        print("尝试在CPU上加载模型...")
        
        try:
            pipe = ov_genai.LLMPipeline(llm_model_path, "CPU")
            device = "CPU"
            print("✓ CPU加载成功!")
        except Exception as cpu_error:
            print(f"✗ CPU加载也失败: {cpu_error}")
            return
    
    # 配置生成参数
    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = 128
    generation_config.temperature = 0.7
    generation_config.top_p = 0.9
    generation_config.do_sample = True
    generation_config.apply_chat_template = False
    
    print(f"\n使用设备: {device}")
    print(f"生成配置:")
    print(f"  - max_new_tokens: {generation_config.max_new_tokens}")
    print(f"  - temperature: {generation_config.temperature}")
    print(f"  - top_p: {generation_config.top_p}")
    print(f"  - do_sample: {generation_config.do_sample}")
    
    # 测试多个提示词
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(prompts)}")
        print(f"输入: {prompt}")
        print("正在生成...")
        
        start_time = time.time()
        
        try:
            result = pipe.generate(prompt, generation_config)
            end_time = time.time()
            
            print(f"\n生成结果:")
            print(result)
            print(f"\n生成时间: {end_time - start_time:.2f} 秒")
            
            # 计算tokens/秒
            tokens_per_second = generation_config.max_new_tokens / (end_time - start_time)
            print(f"生成速度: {tokens_per_second:.2f} tokens/秒")
            
        except Exception as gen_error:
            print(f"生成失败: {gen_error}")
            continue
        
        # 添加分隔符
        print("-" * 60)

def test_streaming_generation():
    """测试流式生成"""
    llm_model_path = "Qwen3-8B-nf4-ov"
    prompt = "请详细解释深度学习的基本原理"
    
    print("\n" + "="*60)
    print("测试流式生成")
    print("="*60)
    
    try:
        # 尝试加载模型
        try:
            pipe = ov_genai.LLMPipeline(llm_model_path, "NPU")
            device = "NPU"
        except:
            pipe = ov_genai.LLMPipeline(llm_model_path, "CPU")
            device = "CPU"
        
        print(f"使用设备: {device}")
        print(f"输入: {prompt}")
        print("流式输出:")
        print("-" * 40)
        
        # 配置流式生成
        generation_config = ov_genai.GenerationConfig()
        generation_config.max_new_tokens = 200
        generation_config.temperature = 0.7
        generation_config.apply_chat_template = False
        
        start_time = time.time()
        
        # 流式生成（如果支持）
        try:
            for chunk in pipe.generate(prompt, generation_config):
                print(chunk, end='', flush=True)
        except:
            # 如果不支持流式，使用普通生成
            result = pipe.generate(prompt, generation_config)
            print(result)
        
        end_time = time.time()
        print(f"\n\n生成完成，用时: {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"流式生成测试失败: {e}")

def check_model_info():
    """检查模型信息"""
    llm_model_path = "Qwen3-8B-nf4-ov"
    
    print("\n" + "="*60)
    print("检查模型信息")
    print("="*60)
    
    try:
        # 加载模型获取信息
        pipe = ov_genai.LLMPipeline(llm_model_path, "CPU")  # 使用CPU获取信息更稳定
        
        # 尝试获取tokenizer信息
        print("模型加载成功")
        
        # 测试tokenizer
        test_text = "Hello, world!"
        print(f"测试tokenizer: '{test_text}'")
        
    except Exception as e:
        print(f"无法获取模型信息: {e}")

if __name__ == "__main__":
    try:
        # 主要测试
        test_llm_with_genai()
        
        # 流式生成测试
        test_streaming_generation()
        
        # 模型信息检查
        check_model_info()
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()