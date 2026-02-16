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
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.system import DistillSystem


def load_rsicd_with_ids(json_file, image_root, max_images=1000):
    with open(json_file, 'r') as f:

        content = json.load(f)
        data = content.get("images", content)
    
    dataset = []
    count = 0
    

    target_split = 'test' 


    for idx, item in enumerate(data):
        if count >= max_images:
            break
            
        split = item.get('split', '')
        if split != target_split:
            continue

        filename = item.get('filename', '')

        img_id = item.get('imgid', idx)
        

        base_path = os.path.join(image_root, filename)
        final_path = base_path
        if not os.path.exists(base_path):
            file_stem = os.path.splitext(filename)[0]
            for ext in ['.tif', '.jpg', '.png']:
                p = os.path.join(image_root, file_stem + ext)
                if os.path.exists(p):
                    final_path = p
                    break
        
        sentences = item.get('sentences', [])
        captions = [sentence['raw'] for sentence in sentences]
        
        if os.path.exists(final_path) and len(captions) >= 5:

            dataset.append((final_path, captions[:5], img_id))
            count += 1
        else:

            pass 
            

    return dataset


def calculate_metrics_by_id(image_features, text_features, image_ids, device):


    

    # unique_ids, unique_indices = np.unique(image_ids_np, return_index=True)
    image_ids_np = np.array(image_ids)
    unique_ids, unique_indices = np.unique(image_ids_np, return_index=True)
    

    unique_image_features = image_features[unique_indices].to(device)
    unique_ids_tensor = torch.tensor(unique_ids, device=device)
    

    all_text_features = text_features.to(device)
    

    ids_list_expanded = []
    for img_id in image_ids:
        ids_list_expanded.extend([img_id] * 5)
    all_text_ids_tensor = torch.tensor(ids_list_expanded, device=device)
    
    print(f"Unique Images (Gallery): {len(unique_ids)}")
    print(f"Total Queries (Text): {len(all_text_ids_tensor)}")


    unique_image_features = F.normalize(unique_image_features, dim=-1)
    all_text_features = F.normalize(all_text_features, dim=-1)
    
    logits_per_text = torch.matmul(all_text_features, unique_image_features.t())
    logits_per_image = logits_per_text.t()


    ground_truth_mask = (all_text_ids_tensor.unsqueeze(1) == unique_ids_tensor.unsqueeze(0))


    def calc_r_k(logits, mask, prefix):

        _, top_indices = logits.topk(10, dim=1)

        extracted_corrects = torch.gather(mask, 1, top_indices)
        
        res = {}
        for k in [1, 5, 10]:
            corrects_at_k = extracted_corrects[:, :k].any(dim=1)
            score = corrects_at_k.float().mean().item() * 100
            print(f"{prefix} R@{k}: {score:.2f}")
            res[f'{prefix}_r{k}'] = score
        return res
    print("--- Image to Text ---")
    i2t_res = calc_r_k(logits_per_image, ground_truth_mask.t(), "Image->Text")
    
    print("--- Text to Image ---")
    t2i_res = calc_r_k(logits_per_text, ground_truth_mask, "Text->Image")
    
    
    
    return {**t2i_res, **i2t_res}


def extract_all_features(model, preprocess, dataset, device):
    img_feats = []
    txt_feats = []
    img_ids = []
    

    # Batch size
    bs = 64
    

    paths = [x[0] for x in dataset]
    captions_group = [x[1] for x in dataset] # List of lists
    ids = [x[2] for x in dataset]
    

    with torch.no_grad():
        for i in tqdm(range(0, len(paths), bs), desc="Images"):
            batch_paths = paths[i:i+bs]
            batch_imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(preprocess(img))
                except:
                    batch_imgs.append(torch.zeros(3, 224, 224))
            
            if batch_imgs:
                img_input = torch.stack(batch_imgs).to(device)
                feat = model.encode_image(img_input)
                img_feats.append(feat.cpu())

            if i == 0:
                feat_sum = feat[0].sum().item()
                feat_head = feat[0, :5].cpu().numpy().tolist()

    

    flat_captions = [c for group in captions_group for c in group]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(flat_captions), bs), desc="Texts"):
            batch_txts = flat_captions[i:i+bs]
            tokens = clip.tokenize(batch_txts, truncate=True).to(device)
            feat = model.encode_text(tokens)
            txt_feats.append(feat.cpu())

    all_img_feats = torch.cat(img_feats, dim=0)
    all_txt_feats = torch.cat(txt_feats, dim=0)
    
    return all_img_feats, all_txt_feats, ids

# Hydra Resolver
def get_center_path(clip_name):

    USERSI = True
    if "ViT-B" in clip_name:
        if USERSI:
            return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/RSI_clustering_results/cluster_centers_vitb32_RSI_32.npy"
    return "/data/clc/APSE-IPIK/NEW_APSEIPIK/offline/clustering_results/cluster_centers_160.npy"

OmegaConf.register_new_resolver("select_path", get_center_path)

@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(config: DictConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    # checkpoint_path = "/data/clc/APSE-IPIK/NEW_APSEIPIK/pth_new/rsicd/RSUM=36.09.ckpt"
    checkpoint_path = "/data/clc/APSE-IPIK/NEW_APSEIPIK/pth_new/ucm/RSUM=81.10.ckpt"
    # checkpoint_path = "/data/clc/APSE-IPIK/NEW_APSEIPIK/pth_new/rsitmd/RSUM=62.75.ckpt"
    print(f"Checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}
    
    system = DistillSystem.load_from_checkpoint(checkpoint_path, strict=True)
    model = system.model.to(device)
    
    # Strict False Check
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:

        critical = [k for k in missing if "adapter" in k or "prompt" in k]

    model.load_state_dict(state_dict, strict=False)      
    model.eval()
    
    # CLIP Preprocess
    clipmodel, preprocess = clip.load(config.model.original_clip_name, device=device)
    

    USE_RSICD = False
    if USE_RSICD:
        json_file = '/data/clc/data/RSICD/annocations/dataset_rsicd.json'
        image_root = '/data/clc/data/RSICD/images'
    USE_UCM = True
    if USE_UCM:
        json_file = '/data/clc/data/UCM_captions/dataset.json'
        image_root = '/data/clc/data/UCM_captions/imgs'
    USE_rsitmd = False
    if USE_rsitmd:
        json_file = '/data/clc/data/RSIMTD/dataset_RSITMD.json'
        image_root = '/data/clc/data/RSIMTD/images'
    dataset = load_rsicd_with_ids(json_file, image_root, max_images=30000)
    

    img_feats, txt_feats, img_ids = extract_all_features(model, preprocess, dataset, device)
    
    print(f"Img Feats: {img_feats.shape}")
    print(f"Txt Feats: {txt_feats.shape}")
    print(f"IDs count: {len(img_ids)}")
    

    results = calculate_metrics_by_id(img_feats, txt_feats, img_ids, device)
    
    rsum = sum(results.values())
    print(f"\n[Final Result] RSUM: {rsum:.2f}")

if __name__ == "__main__":
    main()