import torch
import torch.nn.functional as F
from torchvision.datasets import CocoCaptions
import clip
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import json
from tqdm import tqdm
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.system import DistillSystem
def get_center_path(clip_name):
   
    base_dir = "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results"
    
    
    if "ViT-B" in clip_name:
        
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"
    elif "ViT-L" in clip_name:
        
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_vitl14_160.npy"
    else:
        
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"
OmegaConf.register_new_resolver("select_path", get_center_path)
@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def evaluate_final_model(config: DictConfig):
    
    print(OmegaConf.to_yaml(config))
   
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    
    
    checkpoint_path = "/data/clc/APSE-IPIK/NEW_APSEIPIK/pth_new/coco/RSUM=445.48_vitb32.ckpt"
    
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    system = DistillSystem.load_from_checkpoint(checkpoint_path, strict=False)
    model = system.model
    model.to(device)
    
    
    
   
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    
    
    clip_model, preprocess = clip.load(config.model.original_clip_name, device=device)
    
    
    coco_root = config.dataset.get('root', '/data/clc/Long-CLIP/data/coco/val2017')
    coco_annfile = config.dataset.get('annFile', '/data/clc/Long-CLIP/data/coco/annotations/captions_val2017.json')
    
    coco = CocoCaptions(root=coco_root, annFile=coco_annfile)
    
    num_samples = 5000  
    
    
    
    image_features = []
    text_features = []
   
    total_infer_time = 0.0  
    total_infer_count = 0   
    torch.cuda.reset_peak_memory_stats() 
    torch.cuda.empty_cache()
    # ==========================
    with torch.no_grad():
        
        for i, (image, captions) in enumerate(tqdm(coco, desc="提取特征")):
            if i >= num_samples:
                break
            
           
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            if i > 10: torch.cuda.synchronize() 
            start_t = time.time()
            
            image_feature = model.encode_image(image_input)
            
            if i > 10: 
                torch.cuda.synchronize()
                total_infer_time += time.time() - start_t
                total_infer_count += 1
            
            # image_feature = model.encode_image(image_input)
            # image_feature = clip_model.encode_image(image_input)
            image_features.append(image_feature.float().cpu())
            
            
            captions = captions[:5]  
            for caption in captions:
                text_input = clip.tokenize([caption]).to(device)
                
                if i > 10: torch.cuda.synchronize()
                start_t = time.time()
                
                # text_feature = model.enhance_text(text_input)
                # text_feature = model.enhance_text(text_input)
                text_feature = model.encode_text(text_input)
                if i > 10:
                    torch.cuda.synchronize()
                    total_infer_time += time.time() - start_t
                    total_infer_count += 1
                text_features.append(text_feature.float().cpu())
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
    avg_latency = (total_infer_time / total_infer_count) * 1000 if total_infer_count > 0 else 0 # ms
    
    
    # ================================
    # 合并特征
    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    image_features = image_features.float()
    text_features = text_features.float()
    
   
    
    # 归一化
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    
    # 计算相似度矩阵
    similarity = image_features @ text_features.T
    
    
    
    i2t_r1, i2t_r5, i2t_r10 = evaluate_i2t(similarity, num_samples) 
    
   
    t2i_r1, t2i_r5, t2i_r10 = evaluate_t2i(similarity, num_samples)
    
    
    rsum = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10
    
    
    results = {
        'i2t_r1': i2t_r1,
        'i2t_r5': i2t_r5, 
        'i2t_r10': i2t_r10,
        't2i_r1': t2i_r1,
        't2i_r5': t2i_r5,
        't2i_r10': t2i_r10,
        'rsum': rsum
    }
    
    
    print("\n" + "="*70)
    print("result")
    print("="*70)
    
    print(f"I2T:")
    print(f"  R@1:  {i2t_r1:.4f}")
    print(f"  R@5:  {i2t_r5:.4f}") 
    print(f"  R@10: {i2t_r10:.4f}")
    print(f"T2I:")
    print(f"  R@1:  {t2i_r1:.4f}")
    print(f"  R@5:  {t2i_r5:.4f}")
    print(f"  R@10: {t2i_r10:.4f}")
    print(f"RSUM:   {rsum:.4f}")
    

    output_file = f"val_at_mscoco_RSUM={rsum:.4f}.json"
    with open(output_file, 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    

    return results

def evaluate_i2t(similarity, num_samples):

    pred_true_r1 = 0
    pred_true_r5 = 0 
    pred_true_r10 = 0
    
    for i in range(num_samples):
        pred = similarity[i]
        
        # R@1
        top1_indices = pred.argsort(descending=True)[:1]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in top1_indices:
                pred_true_r1 += 1
                break
        
        # R@5
        top5_indices = pred.argsort(descending=True)[:5]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in top5_indices:
                pred_true_r5 += 1
                break
        
        # R@10
        top10_indices = pred.argsort(descending=True)[:10]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in top10_indices:
                pred_true_r10 += 1
                break
    
    return pred_true_r1*100/num_samples, pred_true_r5*100/num_samples, pred_true_r10*100/num_samples

def evaluate_t2i(similarity, num_samples):

    similarity_t2i = similarity.T
    
    pred_true_r1 = 0
    pred_true_r5 = 0
    pred_true_r10 = 0
    
    for i in range(num_samples * 5):
        pred = similarity_t2i[i]
        true_index = i // 5 
        
        # R@1
        top1_indices = pred.argsort(descending=True)[:1]
        if true_index in top1_indices:
            pred_true_r1 += 1
        
        # R@5  
        top5_indices = pred.argsort(descending=True)[:5]
        if true_index in top5_indices:
            pred_true_r5 += 1
        
        # R@10
        top10_indices = pred.argsort(descending=True)[:10]
        if true_index in top10_indices:
            pred_true_r10 += 1
    
    return pred_true_r1*100/(num_samples*5), pred_true_r5*100/(num_samples*5), pred_true_r10*100/(num_samples*5)

if __name__ == "__main__":
    evaluate_final_model()