import torch
import lpips

class LPIPSEvaluator(object):
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)

    @torch.no_grad()
    def compute_pairwise_distance(self, src_images: torch.Tensor) -> float:
        n = src_images.shape[0]
        total_distance = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = self.lpips_fn(src_images[i].unsqueeze(0).to(self.device),
                                  src_images[j].unsqueeze(0).to(self.device))
                total_distance += d.item() 
                count += 1
        avg_distance = total_distance / count if count > 0 else 0.0
        return avg_distance

if __name__ == "__main__":
    evaluator = LPIPSEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    src_images = torch.randn(4, 3, 224, 224)
    
    src_images = (src_images - src_images.min()) / (src_images.max() - src_images.min()) * 2 - 1
    
    avg_lpips = evaluator.compute_pairwise_distance(src_images)
    print("Average LPIPS distance:", avg_lpips)
