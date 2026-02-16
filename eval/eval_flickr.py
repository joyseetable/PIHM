import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import clip
from tqdm import tqdm
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("feature-extraction", model="Amirhossein75/Image-Contrastive-CLIP-Flickr30k")
# Load model directly
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

processor = AutoProcessor.from_pretrained("Amirhossein75/Image-Contrastive-CLIP-Flickr30k")
pre_model = AutoModelForZeroShotImageClassification.from_pretrained("Amirhossein75/Image-Contrastive-CLIP-Flickr30k")


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.models.newmodel import TwoStageTextEnhancementModel
from systems.system import DistillSystem

def load_flickr30k_from_json(json_file, image_root, max_images=1000):

    with open(json_file, 'r') as f:
        data = json.load(f)["images"]
    
    dataset = []
    count = 0
    
    for item in data:
        if count >= max_images:
            break
            

        filename = item.get('filename', '')
        image_path = os.path.join(image_root, filename)
        sentences = item.get('sentences', [])
        
    
        captions = [sentence['raw'] for sentence in sentences]
        
        if os.path.exists(image_path) and len(captions) >= 5:
            dataset.append((image_path, captions[:5]))  
            count += 1
        
    
    
    return dataset

def get_text_features(model, dataset, device, batch_size=64):
    
    feature_list = []
    
    
    all_captions = []
    for img_path, captions in dataset:
        all_captions.extend(captions[:5])  
    
    
    
    
    with torch.no_grad():
        for i in range(0, len(all_captions), batch_size):
            end_idx = min(i + batch_size, len(all_captions))
            text_batch = all_captions[i:end_idx]
            
            
            text_tokens = clip.tokenize(text_batch, truncate=True).to(device)
            
            
            text_features = model.encode_text(text_tokens)
            
            
            
            text_features = text_features.float().cpu()
            feature_list.append(text_features)
            
            
    
    text_features = torch.cat(feature_list, dim=0)
    return text_features

def get_image_features(model, preprocess, dataset, device, max_images=1000):
    
    img_feature_list = []
    
    
    dataset = dataset[:max_images]
    
    with torch.no_grad():
        for i, (img_path, captions) in enumerate(tqdm(dataset, desc="image features")):
            try:
                image = Image.open(img_path).convert('RGB')
                image_input = preprocess(image).unsqueeze(0).to(device)
                
                
                img_feature = model.encode_image(image_input)
                
                
                
                img_feature = img_feature.float().cpu()
                img_feature_list.append(img_feature)
                
                
                if (i + 1) % 50 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f" {e}")
                continue

    img_features = torch.cat(img_feature_list, dim=0)
    return img_features

def evaluate_i2t(similarity, num_images):
    
    pred_true_r1 = 0
    pred_true_r5 = 0 
    pred_true_r10 = 0
    
    for i in range(num_images):
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
    
    i2t_r1 = pred_true_r1 / num_images
    i2t_r5 = pred_true_r5 / num_images
    i2t_r10 = pred_true_r10 / num_images
    
    print(f"Image→Text R@1: {i2t_r1:.4f}")
    print(f"Image→Text R@5: {i2t_r5:.4f}")
    print(f"Image→Text R@10: {i2t_r10:.4f}")
    
    return i2t_r1*100, i2t_r5*100, i2t_r10*100

def evaluate_t2i(similarity, num_images):
    
    similarity_t2i = similarity.T
    num_texts = num_images * 5
    
    pred_true_r1 = 0
    pred_true_r5 = 0
    pred_true_r10 = 0
    
    for i in range(num_texts):
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
    
    t2i_r1 = pred_true_r1 / num_texts
    t2i_r5 = pred_true_r5 / num_texts
    t2i_r10 = pred_true_r10 / num_texts
    
    print(f"Text→Image R@1: {t2i_r1:.4f}")
    print(f"Text→Image R@5: {t2i_r5:.4f}")
    print(f"Text→Image R@10: {t2i_r10:.4f}")
    
    return t2i_r1*100, t2i_r5*100, t2i_r10*100
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
def evaluate_flickr30k_final(config: DictConfig):
    
    
    print(OmegaConf.to_yaml(config))
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
   
    MAX_IMAGES = 1000
    
    
    checkpoint_path = "/data/clc/APSE-IPIK/NEW_APSEIPIK/outputs/baseline_full_finetune_ratio0.3/flickr30k/checkpoints/best-epoch=epoch=28-rsum=val/RSUM=132.91.ckpt"
    
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    system = DistillSystem.load_from_checkpoint(checkpoint_path, strict=False)
    model = system.model
    model.to(device)
    # model.training_stage = 2
    
    
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
    
    
 
    _, preprocess = clip.load(config.model.original_clip_name, device=device)
    

    json_file = '/data/clc/APSE-IPIK/NEW_APSEIPIK/DATA/flickr30k/annotations/test.json'
    image_root = '/data/clc/APSE-IPIK/NEW_APSEIPIK/DATA/flickr30k/images'
    

    if not os.path.exists(json_file):
        possible_json_paths = [
            './data/flickr30k/val.json',
            '/data/flickr30k/val.json',
            '/data/clc/Long-CLIP/data/flickr30k/val.json'
        ]
        for path in possible_json_paths:
            if os.path.exists(path):
                json_file = path
                break
    
    if not os.path.exists(image_root):
        possible_image_paths = [
            './data/flickr30k-images',
            '/data/flickr30k-images',
            '/data/clc/Long-CLIP/data/flickr30k-images'
        ]
        for path in possible_image_paths:
            if os.path.exists(path):
                image_root = path
                break
    

    

    dataset = load_flickr30k_from_json(json_file, image_root, max_images=MAX_IMAGES)
    
    if len(dataset) == 0:

        return
    

    

    

    text_features = get_text_features(model, dataset, device)

    

    image_features = get_image_features(model, preprocess, dataset, device, max_images=MAX_IMAGES)
    # image_features = get_image_features(pre_model, preprocess, dataset, device, max_images=MAX_IMAGES)
    

    

    expected_text_features = len(dataset) * 5

    

    image_features = image_features.float()
    text_features = text_features.float()
    

    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    

    similarity = image_features @ text_features.T

    


    
    num_images = len(dataset)
    

    i2t_r1, i2t_r5, i2t_r10 = evaluate_i2t(similarity, num_images) 
    

    t2i_r1, t2i_r5, t2i_r10 = evaluate_t2i(similarity, num_images)
    

    rsum = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10
    

    results = {
        'i2t_r1': float(i2t_r1),
        'i2t_r5': float(i2t_r5), 
        'i2t_r10': float(i2t_r10),
        't2i_r1': float(t2i_r1),
        't2i_r5': float(t2i_r5),
        't2i_r10': float(t2i_r10),
        'rsum': float(rsum),
        'num_images': num_images,
        'num_texts': num_images * 5
    }
    

    

    output_file = f"val_at_flickr30k_RSUM={rsum:.4f}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    

    
    return results

if __name__ == "__main__":
    evaluate_flickr30k_final()