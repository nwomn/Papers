# -*- coding: utf-8 -*-

"""
一个用于加载 Hugging Face Causal LM 模型并计算其参数量的脚本。

如何使用:
1.  保存此文件为 `count_params.py`。
2.  确保已安装必要的库:
    pip install torch transformers accelerate bitsandbytes
3.  从命令行运行:
    # 示例 1: 计算 Llama-3-8B 的参数
    python count_params.py meta-llama/Meta-Llama-3-8B

    # 示例 2: 计算一个较小的模型，如 GPT-2
    python count_params.py gpt2

    # 示例 3: 计算需要信任远程代码的模型 (如 Falcon)
    python count_params.py tiiuae/falcon-7b --trust-remote-code

    # 示例 4: 以 bfloat16 格式加载以节省内存
    python count_params.py mistralai/Mixtral-8x7B-Instruct-v0.1 --torch_dtype bfloat16

    # 示例 5: 加载本地模型检查点
    python count_params.py /path/to/your/local/model
"""

import argparse
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import torch
import fla
import sys
# 导入并注册MOB模型
try:
    sys.path.insert(0, '/public/liguoqi/shirl')
    sys.path.insert(0, '/public/liguoqi/shirl/models')
    # 添加mob路径
    mob_liger_gla_path = '/public/liguoqi/shirl/mob_liger_gla'
    sys.path.insert(0, mob_liger_gla_path)
    import mob_liger_gla
    print("✓ MOB_Liger_GLA模型已在openicl_infer中成功导入和注册")
    
    mob_gdn_path = '/public/liguoqi/shirl/models/mob_gated_deltanet'
    sys.path.insert(0, mob_gdn_path)
    import mob_gated_deltanet
    print("✓ MOB_Gated_DeltaNet模型已在openicl_infer中成功导入和注册")
    
    import mom
    print("✓ MOM模型已在openicl_infer中成功导入和注册")
    
except ImportError as e:
    print(f"⚠️ 模型导入失败: {e}")
from transformers import AutoModelForCausalLM

def format_params(num_params: int) -> str:
    """将参数数量格式化为 K, M, B 单位"""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f} B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f} M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f} K"
    else:
        return str(num_params)
    
def count_and_print_parameters(model: torch.nn.Module):
    """
    精确计算并打印模型的可训练参数量。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print("模型参数详情")
    print("="*50)
    print(f"  总参数量:       {total_params:,} ({format_params(total_params)})")
    print(f"  可训练参数量:   {trainable_params:,} ({format_params(trainable_params)})")
    print("="*50 + "\n")
    return total_params

def analyze_model_parameters(model_name_or_path: str, torch_dtype: str, trust_remote_code: bool):
    """
    加载模型并计算其参数量。
    """
    print("="*60)
    print(f"模型: {model_name_or_path}")
    print(f"数据类型 (torch_dtype): {torch_dtype}")
    print(f"信任远程代码 (trust_remote_code): {trust_remote_code}")
    print("="*60)

    try:
        print("\n[INFO] 正在加载模型... 这可能需要一些时间，具体取决于模型大小和网络速度。")
        
        # 使用 AutoModelForCausalLM 加载模型
        # device_map='auto' 会自动利用可用的硬件 (GPU/CPU)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto",
            trust_remote_code=trust_remote_code,
            device_map="auto" 
        )
        
        print("[SUCCESS] 模型加载成功！")

        count_and_print_parameters(model)
        
        # 打印模型占用的显存（如果在GPU上）
        if torch.cuda.is_available() and model.device.type == 'cuda':
            # model.get_memory_footprint() 方法可以估算模型大小
            footprint_bytes = model.get_memory_footprint()
            print(f"  模型内存占用 (估算): {footprint_bytes / (1024**3):.2f} GB")
            
        print("-" * 20 + "\n")

    except OSError as e:
        print(f"\n[ERROR] 加载模型失败。请检查模型名称或路径是否正确。")
        print(f"  错误详情: {e}")
    except Exception as e:
        print(f"\n[ERROR] 发生未知错误。")
        print(f"  错误详情: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加载 Hugging Face Causal LM 模型并计算其参数量。")
    
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="/9950backfile/wkk/exp/transformer-1B-100B/checkpoint-50016",
        help="要分析的 Hugging Face 模型名称或本地检查点路径。"
    )
    
    parser.add_argument(
        "--torch_dtype", 
        type=str, 
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="加载模型时使用的数据类型，'auto' 会自动选择。"
    )
    
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="当模型需要执行自定义代码时，必须设置此项。"
    )
    
    args = parser.parse_args()
    
    analyze_model_parameters(args.model_name_or_path, args.torch_dtype, args.trust_remote_code)