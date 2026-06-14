#!/usr/bin/env python
import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# Resolve standard paths so that we import generator and models from original sources
original_sources_path = str(Path(__file__).parent.parent / "sources" / "MisoTTS")
if original_sources_path not in sys.path:
    sys.path.append(original_sources_path)

# ANSI terminal colors
BOLD = "\033[1m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
RESET = "\033[0m"


def main():
    print(f"\n{BOLD}{CYAN}=== MisoTTS to MLX Weight Converter ==={RESET}\n")

    # 1. Resolve local cache or download the safetensors file
    repo_id = "MisoLabs/MisoTTS"
    filename = "model.safetensors"
    
    print(f"{BLUE}ℹ Resolving model file in Hugging Face Hub Cache...{RESET}")
    try:
        model_file = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"{GREEN}✔ Resolved cached model checkpoint file:{RESET} {model_file}\n")
    except Exception as e:
        print(f"{RED}✘ Failed to resolve model file: {e}{RESET}")
        print(f"{YELLOW}Please run 'uv run python miso_mlx/miso_mlx_cli.py download' first to cache the weights.{RESET}")
        sys.exit(1)

    # 2. Check for safetensors support
    try:
        from safetensors import safe_open
    except ImportError:
        print(f"{RED}✘ The 'safetensors' package is required.{RESET}")
        print(f"{YELLOW}Install it using: pip install safetensors{RESET}")
        sys.exit(1)

    # 3. Read the safetensors metadata and tensor headers
    print(f"{BLUE}ℹ Reading safetensors metadata and mapping parameter names...{RESET}")
    
    # We will group the tensors to analyze the model topology
    backbone_tensors = []
    decoder_tensors = []
    embedding_tensors = []
    head_tensors = []
    
    total_params = 0
    
    with safe_open(model_file, framework="pt", device="cpu") as f:
        keys = f.keys()
        for key in sorted(keys):
            tensor_slice = f.get_slice(key)
            shape = list(tensor_slice.get_shape())
            dtype = tensor_slice.get_dtype()
            
            # Calculate total parameters
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            total_params += num_elements

            tensor_info = {"key": key, "shape": shape, "dtype": dtype, "elements": num_elements}
            
            if key.startswith("backbone."):
                backbone_tensors.append(tensor_info)
            elif key.startswith("decoder."):
                decoder_tensors.append(tensor_info)
            elif key.startswith("text_embeddings.") or key.startswith("audio_embeddings."):
                embedding_tensors.append(tensor_info)
            else:
                head_tensors.append(tensor_info)

    # Print a high-level summary of model parameters
    print(f"{BOLD}{GREEN}✔ Successfully analyzed model topology!{RESET}\n")
    print(f"Total Tensors Found: {len(keys)}")
    print(f"Total Parameters: {total_params / 1e9:.2f} Billion ({total_params:,} parameters)")
    
    print("\n" + "-" * 60)
    print(f"{BOLD}{BLUE}Model Sub-Network Breakdown:{RESET}")
    print("-" * 60)
    print(f"1. {BOLD}Backbone (8B Llama):{RESET} {len(backbone_tensors)} tensors")
    print(f"2. {BOLD}Decoder (300M Llama):{RESET} {len(decoder_tensors)} tensors")
    print(f"3. {BOLD}Embeddings:{RESET} {len(embedding_tensors)} tensors")
    print(f"4. {BOLD}Output Heads / Projections:{RESET} {len(head_tensors)} tensors")
    print("-" * 60)

    # 4. Generate the PyTorch-to-MLX Key Mapping Blueprint
    # Save the output relative to miso_mlx directory
    mlx_weights_dir = Path(__file__).parent / "mlx_weights"
    mlx_weights_dir.mkdir(exist_ok=True)
    mapping_report_path = mlx_weights_dir / "pytorch_to_mlx_mapping.txt"
    
    print(f"\n{BLUE}ℹ Generating the complete PyTorch to MLX translation mapping file...{RESET}")
    
    # Construct complete dictionary mapping
    mapping = {
        # Embeddings & Heads
        "text_embeddings.weight": "text_embeddings.weight",
        "audio_embeddings.weight": "audio_embeddings.weight",
        "projection.weight": "projection.weight",
        "codebook0_head.weight": "codebook0_head.weight",
        "audio_head": "audio_head",
    }
    
    # Backbone 8B
    for i in range(32):
        mapping[f"backbone.layers.{i}.sa_norm.scale"] = f"backbone.layers.{i}.attention_norm.weight"
        mapping[f"backbone.layers.{i}.mlp_norm.scale"] = f"backbone.layers.{i}.ffn_norm.weight"
        mapping[f"backbone.layers.{i}.attn.q_proj.weight"] = f"backbone.layers.{i}.attention.wq.weight"
        mapping[f"backbone.layers.{i}.attn.k_proj.weight"] = f"backbone.layers.{i}.attention.wk.weight"
        mapping[f"backbone.layers.{i}.attn.v_proj.weight"] = f"backbone.layers.{i}.attention.wv.weight"
        mapping[f"backbone.layers.{i}.attn.output_proj.weight"] = f"backbone.layers.{i}.attention.wo.weight"
        mapping[f"backbone.layers.{i}.mlp.w1.weight"] = f"backbone.layers.{i}.feed_forward.w1.weight"
        mapping[f"backbone.layers.{i}.mlp.w2.weight"] = f"backbone.layers.{i}.feed_forward.w2.weight"
        mapping[f"backbone.layers.{i}.mlp.w3.weight"] = f"backbone.layers.{i}.feed_forward.w3.weight"
    mapping["backbone.norm.scale"] = "backbone.norm.weight"
    
    # Decoder 300M
    for i in range(8):
        mapping[f"decoder.layers.{i}.sa_norm.scale"] = f"decoder.layers.{i}.attention_norm.weight"
        mapping[f"decoder.layers.{i}.mlp_norm.scale"] = f"decoder.layers.{i}.ffn_norm.weight"
        mapping[f"decoder.layers.{i}.attn.q_proj.weight"] = f"decoder.layers.{i}.attention.wq.weight"
        mapping[f"decoder.layers.{i}.attn.k_proj.weight"] = f"decoder.layers.{i}.attention.wk.weight"
        mapping[f"decoder.layers.{i}.attn.v_proj.weight"] = f"decoder.layers.{i}.attention.wv.weight"
        mapping[f"decoder.layers.{i}.attn.output_proj.weight"] = f"decoder.layers.{i}.attention.wo.weight"
        mapping[f"decoder.layers.{i}.mlp.w1.weight"] = f"decoder.layers.{i}.feed_forward.w1.weight"
        mapping[f"decoder.layers.{i}.mlp.w2.weight"] = f"decoder.layers.{i}.feed_forward.w2.weight"
        mapping[f"decoder.layers.{i}.mlp.w3.weight"] = f"decoder.layers.{i}.feed_forward.w3.weight"
    mapping["decoder.norm.scale"] = "decoder.norm.weight"

    with open(mapping_report_path, "w") as out:
        out.write("============================================================\n")
        out.write("MISOTTS PYTORCH TO MLX PARAMETER KEY TRANSLATION MAP TEMPLATE\n")
        out.write("============================================================\n\n")
        out.write("This file maps MisoTTS PyTorch state_dict layers to standard\n")
        out.write("MLX model layer structures.\n\n")
        
        for pt_key, mlx_key in mapping.items():
            out.write(f"PyTorch: {pt_key:<55} -> MLX Target: {mlx_key}\n")

    print(f"{BOLD}{GREEN}✔ Generated key mapping translation template at:{RESET} {mapping_report_path.resolve()}")
    
    # 5. Perform the actual weight translation and serialization
    print(f"{BLUE}ℹ Translating and serializing weights to MLX native format...{RESET}")
    try:
        import mlx.core as mx
        
        mlx_weights = {}
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for pt_key, mlx_key in mapping.items():
                if pt_key in f.keys():
                    tensor = f.get_tensor(pt_key)
                    # Convert to PyTorch numpy array, then initialize MLX array to bfloat16
                    arr = mx.array(tensor.numpy(), dtype=mx.bfloat16)
                    mlx_weights[mlx_key] = arr
        
        output_weights_path = mlx_weights_dir / "model.safetensors"
        mx.save_safetensors(str(output_weights_path), mlx_weights)
        print(f"{BOLD}{GREEN}✔ Fully converted MLX weights generated and saved to:{RESET} {output_weights_path.resolve()}\n")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{YELLOW}⚠ Could not execute real weight translation: {e}{RESET}")
        print(f"{YELLOW}Ensure you have downloaded the weights using 'uv run python miso_mlx/miso_mlx_cli.py download' first.{RESET}\n")


if __name__ == "__main__":
    main()
