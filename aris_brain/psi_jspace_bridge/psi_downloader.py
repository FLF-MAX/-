"""
PSI Model Downloader — 下载并配置任意开源大模型用于 PSI 植入
============================================================

支持: DeepSeek V4 / K2 / Llama / Qwen / Mistral / Falcon
后端: HuggingFace Hub → GGUF 转换
"""

import os
import sys
import json
import subprocess
from typing import Optional, List, Dict

# ═══════════════════════════════════════════════════════════
# 模型仓库映射
# ═══════════════════════════════════════════════════════════

MODELS = {
    # DeepSeek V4 系列
    "deepseek-v4": {
        "repo": "deepseek-ai/DeepSeek-V4",
        "gguf_available": True,
        "quantizations": ["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
        "description": "DeepSeek V4 基础版",
        "params": "~671B (MoE, 37B active)",
        "min_ram_gb": 48,
    },
    "deepseek-v4-q4": {
        "repo": "deepseek-ai/DeepSeek-V4-GGUF",
        "gguf_available": True,
        "quantizations": ["Q4_K_M"],
        "description": "DeepSeek V4 Q4_K_M GGUF",
        "params": "~671B quantized → ~120GB",
        "min_ram_gb": 128,
    },

    # DeepSeek V3 (可用替代)
    "deepseek-v3": {
        "repo": "deepseek-ai/DeepSeek-V3",
        "gguf_available": True,
        "quantizations": ["Q4_K_M", "Q5_K_M", "Q8_0"],
        "description": "DeepSeek V3 (671B MoE)",
        "params": "671B (37B active)",
        "min_ram_gb": 48,
    },

    # K2 (传闻中的 1.6T)
    "k2": {
        "repo": "placeholder/k2-1.6t",  # 实际 repo 等待确认
        "gguf_available": False,  # 可能还没有 GGUF
        "quantizations": [],
        "description": "K2 1.6T 参数模型",
        "params": "~1.6T",
        "min_ram_gb": 320,  # F16: ~3.2TB, only Q2 is feasible
    },

    # Qwen 2.5 系列
    "qwen2.5-72b": {
        "repo": "Qwen/Qwen2.5-72B-Instruct-GGUF",
        "gguf_available": True,
        "quantizations": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
        "description": "Qwen 2.5 72B 指令版",
        "params": "72B",
        "min_ram_gb": 48,
    },
    "qwen2.5-32b": {
        "repo": "Qwen/Qwen2.5-32B-Instruct-GGUF",
        "gguf_available": True,
        "quantizations": ["Q4_K_M", "Q5_K_M", "Q8_0"],
        "description": "Qwen 2.5 32B 指令版",
        "params": "32B",
        "min_ram_gb": 24,
    },

    # Llama 3.x
    "llama3.1-405b": {
        "repo": "meta-llama/Llama-3.1-405B-Instruct-GGUF",
        "gguf_available": True,
        "quantizations": ["Q4_K_M", "Q5_K_M", "Q6_K", "IQ4_XS"],
        "description": "Llama 3.1 405B 指令版",
        "params": "405B",
        "min_ram_gb": 256,
    },
    "llama3.1-70b": {
        "repo": "meta-llama/Llama-3.1-70B-Instruct-GGUF",
        "gguf_available": True,
        "quantizations": ["Q4_K_M", "Q5_K_M", "Q8_0", "IQ4_XS"],
        "description": "Llama 3.1 70B 指令版",
        "params": "70B",
        "min_ram_gb": 48,
    },

    # Mistral / Codestral
    "mistral-large-2": {
        "repo": "mistralai/Mistral-Large-Instruct-2407-GGUF",
        "gguf_available": True,
        "quantizations": ["Q4_K_M", "Q5_K_M", "Q8_0"],
        "description": "Mistral Large 2 (123B)",
        "params": "123B",
        "min_ram_gb": 80,
    },
}

# 模型下载目录：可通过 LAAP_MODELS_DIR 环境变量覆盖
# 默认放在项目根目录的同级 laap_models 文件夹下
_DEFAULT_MODELS_DIR = str(Path(__file__).resolve().parents[3] / "laap_models")
INSTALL_DIR = os.path.abspath(os.environ.get("LAAP_MODELS_DIR", _DEFAULT_MODELS_DIR))


def list_available_models() -> Dict:
    """列出所有可下载的模型"""
    return {
        name: {
            "description": info["description"],
            "params": info["params"],
            "quantizations": info["quantizations"],
            "min_ram_gb": info["min_ram_gb"],
            "gguf": info["gguf_available"],
        }
        for name, info in MODELS.items()
    }


def download_model(model_name: str, 
                   quantization: str = "Q4_K_M",
                   output_dir: str = None):
    """
    下载模型到本地。
    
    使用 HuggingFace CLI 下载 GGUF 文件。
    
    Args:
        model_name: 模型名称 (MODELS 字典的 key)
        quantization: 量化级别
        output_dir: 输出目录 (默认: LAAP_MODELS_DIR 环境变量或项目根同级 laap_models/)
    
    Returns:
        下载的模型文件路径
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available: {list(MODELS.keys())}")

    info = MODELS[model_name]
    out_dir = output_dir or INSTALL_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 构建 huggingface 下载命令
    if info["gguf_available"]:
        # GGUF 格式 — 直接下载单个文件
        repo = info["repo"]
        
        # 文件名模式: 通常是 {model_name}-{quant}.gguf
        base_name = repo.split("/")[-1]
        filename = f"{base_name}-{quantization}.gguf"

        cmd = [
            "huggingface-cli", "download",
            repo,
            filename,
            "--local-dir", out_dir,
            "--local-dir-use-symlinks", "False",
        ]

        print(f"下载模型: {model_name} ({info['description']})")
        print(f"  仓库: {repo}")
        print(f"  文件: {filename}")
        print(f"  路径: {out_dir}")
        print(f"  命令: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  错误: {result.stderr}")
            return None

        model_path = os.path.join(out_dir, filename)
        print(f"  下载完成: {model_path}")
        return model_path
    else:
        print(f"模型 {model_name} 没有 GGUF 格式可用。需要先从原始权重转换。")
        print(f"  1. 从 {info['repo']} 克隆原始权重")
        print(f"  2. 使用 convert.py 转换为 GGUF")
        print(f"  3. 使用 quantize.py 量化")
        return None


def setup_llamacpp_psi(model_path: str, psi_enabled: bool = True):
    """
    为已下载的模型配置 PSI 植入运行环境。
    
    创建启动脚本，自动加载 psi_sampler.py。
    """
    script_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)

    script_content = f'''"""
PSI 启动脚本 — {model_name}
自动加载 PSI 采样调制器。
"""
import sys
import os

# 添加桥接器路径
BRIDGE_DIR = r"{BRIDGE_DIR}"
sys.path.insert(0, BRIDGE_DIR)

from psi_sampler import PsiSampler, PsiLlamaCppWrapper

# PSI 全局实例
psi_sampler = PsiSampler()
psi_enabled = {str(psi_enabled).lower()}

def wrap_model(llm_instance):
    """包装 llama_cpp.Llama 实例"""
    return PsiLlamaCppWrapper(llm_instance, psi_enabled=psi_enabled)

if __name__ == "__main__":
    print(f"PSI 植入就绪 — 使用 wrap_model(your_llm) 包装")
    print(f"  psi_enabled={{psi_enabled}}")
    print(f"  psi_state.json at {{BRIDGE_DIR}}/psi_state.json")
'''

    script_path = os.path.join(script_dir, "psi_launch.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content.replace("{BRIDGE_DIR}", BRIDGE_DIR)
                               .replace("{model_name}", model_name)
                               .replace("{str(psi_enabled).lower()}", str(psi_enabled).lower()))

    print(f"启动脚本已创建: {script_path}")
    print(f"使用方法:")
    print(f"  from llama_cpp import Llama")
    print(f"  llm = Llama(r'{model_path}')")
    print(f"  from psi_launch import wrap_model")
    print(f"  psi_llm = wrap_model(llm)")
    print(f"  result = psi_llm.generate('你好')")
    return script_path


if __name__ == "__main__":
    print("=" * 60)
    print("PSI Model Downloader — 可用模型")
    print("=" * 60)

    models = list_available_models()
    for name, info in models.items():
        print(f"\n  {name}")
        print(f"    {info['description']}")
        print(f"    参数: {info['params']}")
        print(f"    量化: {', '.join(info['quantizations'])}")
        print(f"    最小 RAM: {info['min_ram_gb']}GB")
        print(f"    GGUF: {'✓' if info['gguf'] else '✗'}")

    print(f"\n{'=' * 60}")
    print("安装目录:", INSTALL_DIR)
    print("使用: python3 psi_downloader.py <model_name> [quantization]")
    print("例如: python3 psi_downloader.py qwen2.5-72b Q4_K_M")
