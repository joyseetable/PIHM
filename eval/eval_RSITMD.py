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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.system import DistillSystem

def load_rsitmd_from_json(json_file, image_root, split='test'):

    with open(json_file, 'r') as f:
        data = json.load(f)["images"]
    
    dataset = []
    
    for item in data:

        if item.get('split') != split:
            continue
            
        filename = item.get('filename', '')
        image_path = os.path.join(image_root, filename)
        
        sentences = item.get('sentences', [])

        captions = [sentence['raw'] for sentence in sentences]
        

        if os.path.exists(image_path):
            if len(captions) >= 5:
                dataset.append((image_path, captions[:5]))  


    

    return dataset

def get_text_features(model, dataset, device, batch_size=128):

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

def get_image_features(model, preprocess, dataset, device, batch_size=128):

    img_feature_list = []
    
    with torch.no_grad():

        batch_imgs = []
        
        for i, (img_path, captions) in enumerate(tqdm(dataset, desc="image features")):
            try:
                
                image = Image.open(img_path).convert('RGB')
                image_input = preprocess(image)
                batch_imgs.append(image_input)
                
                
                if len(batch_imgs) == batch_size or i == len(dataset) - 1:
                    img_tensor = torch.stack(batch_imgs).to(device)
                    img_feature = model.encode_image(img_tensor)
                    img_feature = img_feature.float().cpu()
                    img_feature_list.append(img_feature)
                    batch_imgs = [] 
                    
            except Exception as e:
                print(f" {e}")
                continue

    img_features = torch.cat(img_feature_list, dim=0)
    return img_features

def evaluate_metrics(similarity, num_images):

    # Image -> Text
    i2t_r1, i2t_r5, i2t_r10 = 0, 0, 0
    for i in range(num_images):
        pred = similarity[i]
        
        true_indices = set(range(i * 5, (i + 1) * 5))
        
       
        indices = pred.argsort(descending=True)
        
        # Check R@1
        if indices[0].item() in true_indices: i2t_r1 += 1
        
        # Check R@5
        
        top5 = indices[:5].tolist()
        if any(idx in true_indices for idx in top5): i2t_r5 += 1
            
        # Check R@10
        top10 = indices[:10].tolist()
        if any(idx in true_indices for idx in top10): i2t_r10 += 1

    i2t_r1 /= num_images
    i2t_r5 /= num_images
    i2t_r10 /= num_images

    # Text -> Image
    t2i_r1, t2i_r5, t2i_r10 = 0, 0, 0
    similarity_t2i = similarity.T
    num_texts = num_images * 5
    
    for i in range(num_texts):
        pred = similarity_t2i[i]
        true_img_idx = i // 5
        
        indices = pred.argsort(descending=True)
        
        if indices[0].item() == true_img_idx: t2i_r1 += 1
        if true_img_idx in indices[:5].tolist(): t2i_r5 += 1
        if true_img_idx in indices[:10].tolist(): t2i_r10 += 1

    t2i_r1 /= num_texts
    t2i_r5 /= num_texts
    t2i_r10 /= num_texts

    return (i2t_r1*100, i2t_r5*100, i2t_r10*100), (t2i_r1*100, t2i_r5*100, t2i_r10*100)


def get_center_path(clip_name):

    base_dir = "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results"
    if "ViT-B" in clip_name:
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/RSI_clustering_results/cluster_centers_vitb32_RSI_32.npy"
        # return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"
    elif "ViT-L" in clip_name:
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_vitl14_160.npy"
    else:
        return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"


if not OmegaConf.has_resolver("select_path"):
    OmegaConf.register_new_resolver("select_path", get_center_path)

@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def evaluate_rsitmd(config: DictConfig):
    
    print(OmegaConf.to_yaml(config))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    

    rsitmd_json_file = '/data/clc/data/RSIMTD/dataset_RSITMD.json' 
    rsitmd_image_root = '/data/clc/data/RSIMTD/images'
    
   
    CKPT_PATH = "/data/clc/APSE-IPIK/NEW_APSEIPIK/pth_new/rsitmd/RSUM=62.75.ckpt"
    # ===========================================

    if not os.path.exists(rsitmd_json_file):
        print(f"错误: 找不到 json 文件: {rsitmd_json_file}")
        
        possible_paths = [
            './data/RSITMD/dataset_RSITMD.json',
            '../data/RSITMD/dataset_RSITMD.json'
        ]
        for p in possible_paths:
            if os.path.exists(p):
                rsitmd_json_file = p
                
                break


    USEMYMODEL = True
    if USEMYMODEL:

 
        checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)

        system = DistillSystem.load_from_checkpoint(CKPT_PATH, strict=False)
        model = system.model
        model.to(device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        _, preprocess = clip.load(config.model.original_clip_name, device=device)
    else:
       
        model, preprocess = clip.load(config.model.original_clip_name, device=device)
   
    # checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # system = DistillSystem.load_from_checkpoint(checkpoint_path, strict=False)
    # model = system.model
    # model.to(device)
    
   
    if USEMYMODEL:
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
        print("模型加载成功")
    

    

    dataset = load_rsitmd_from_json(rsitmd_json_file, rsitmd_image_root, split='test')
    
    if len(dataset) == 0:

        return


    text_features = get_text_features(model, dataset, device)
    

    image_features = get_image_features(model, preprocess, dataset, device)
    

    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    


    similarity = image_features @ text_features.T
    

    num_images = len(dataset)
    (i2t_r1, i2t_r5, i2t_r10), (t2i_r1, t2i_r5, t2i_r10) = evaluate_metrics(similarity, num_images)
    
    rsum = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10

    print("\n" + "="*50)
    print(f"RSITMD Test Set Evaluation (Images: {num_images})")
    print("="*50)
    print(f"Image-to-Text:")
    print(f"R@1:  {i2t_r1:.2f}")
    print(f"R@5:  {i2t_r5:.2f}")
    print(f"R@10: {i2t_r10:.2f}")
    print("-" * 20)
    print(f"Text-to-Image:")
    print(f"R@1:  {t2i_r1:.2f}")
    print(f"R@5:  {t2i_r5:.2f}")
    print(f"R@10: {t2i_r10:.2f}")
    print("-" * 20)
    print(f"RSUM: {rsum:.2f}")
    print("="*50)
    

    results = {
        'dataset': 'RSITMD',
        'split': 'test',
        'i2t': {'r1': i2t_r1, 'r5': i2t_r5, 'r10': i2t_r10},
        't2i': {'r1': t2i_r1, 'r5': t2i_r5, 'r10': t2i_r10},
        'rsum': rsum
    }
    
    output_file = f"eval_rsitmd_RSUM={rsum:.2f}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    evaluate_rsitmd()