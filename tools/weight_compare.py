import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,  
    'ytick.labelsize': 16,  
    'legend.fontsize': 14
})

def load_state_dict(filepath):
    state = torch.load(filepath, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        return state['model']
    return state

def compute_layer_diff(orig_state, concept_state, eps=1e-18):
    diff_self = []
    diff_cross = []
    diff_other = []
    exclude_layers = [
        "cfg_uncond",
        "pos_start",
        "text_norm.weight",
        "lvl_embed.weight",
        "word_embed.weight",
        "word_embed.bias",
        "shared_ada_lin.1.weight",
        "shared_ada_lin.1.bias",
        "head_nm.ada_lin.1.weight",
        "head_nm.ada_lin.1.bias",
        "head.weight",
        "head.bias",
    ]
    
    for key, orig_weight in orig_state.items():
        if key not in concept_state:
            continue  
        concept_weight = concept_state[key]
        if not (isinstance(orig_weight, torch.Tensor) and isinstance(concept_weight, torch.Tensor)):
            continue

        diff_tensor = torch.abs(orig_weight - concept_weight) / (torch.abs(orig_weight) + eps)
        diff_value = diff_tensor.mean().item()
        
        if 'sa' in key:
            diff_self.append(diff_value)
        elif 'ca' in key:
            diff_cross.append(diff_value)
        elif key not in exclude_layers:
            diff_other.append(diff_value)
    
    avg_self = np.mean(diff_self) if diff_self else 0
    avg_cross = np.mean(diff_cross) if diff_cross else 0
    avg_other = np.mean(diff_other) if diff_other else 0
    
    return {'self_attention': avg_self, 'cross_attention': avg_cross, 'other': avg_other}

def main():
    parser = argparse.ArgumentParser(description="Visualize the layer-wise differences between state_dicts using a graph.")
    parser.add_argument('--original', type=str, default="weights/infinity_2b_reg.pth",
                        help="")
    parser.add_argument('--concept_dir', type=str, default="./ckpts/local_ti1_full",
                        help="")
    parser.add_argument('--output', type=str, default="diff_comparison.pdf",
                        help="")
    args = parser.parse_args()

    print(f"Loading original weight from {args.original}")
    orig_state = load_state_dict(args.original)

    concept_results = {}
    
    for item in os.listdir(args.concept_dir):
        item_path = os.path.join(args.concept_dir, item)
        if os.path.isdir(item_path):
            pth_files = [f for f in os.listdir(item_path) if f.endswith('.pth')]
            if not pth_files:
                print(f"No .pth files found in {item_path}")
                continue
            pth_files.sort()
            concept_file = os.path.join(item_path, pth_files[-1])
            concept_name = item
        elif item.endswith('.pth'):
            concept_file = item_path
            concept_name = os.path.splitext(item)[0]
        else:
            continue
        
        print(f"Loading concept weight for '{concept_name}' from {concept_file}")
        concept_state = load_state_dict(concept_file)
        diff_dict = compute_layer_diff(orig_state, concept_state)
        concept_results[concept_name] = diff_dict
    
    if not concept_results:
        print("No concept weights found. Exiting.")
        return

    avg_self = np.mean([diff['self_attention'] for diff in concept_results.values()])
    avg_cross = np.mean([diff['cross_attention'] for diff in concept_results.values()])
    avg_ffn = np.mean([diff['other'] for diff in concept_results.values()])  # "other"를 FFN으로 간주

    categories = ['Self-Attention', 'Cross-Attention', 'FFN']
    avg_values = [avg_self, avg_cross, avg_ffn]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(categories, avg_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    
    ax.set_ylabel('Average Ratio of Weight Diff.')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=16)  
    
    fig.tight_layout()
    plt.savefig(os.path.join("./local_analyze_result", args.output), dpi=300)
    print(f"Graph saved to {args.output}")
    plt.show()

if __name__ == '__main__':
    main()
