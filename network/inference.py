import torch
from tqdm import tqdm
import pandas as pd
from dataset import Inference_Dataset
import argparse
import torchvision
import re
import torch.nn.functional as F
from torchcam.methods import SmoothGradCAMpp
import numpy as np
from torchvision.transforms import transforms

def main(args):
    torch.cuda.set_device(args.cuda)
    
    lag_order, horizon =  re.findall(r'\d+', args.model_checkpoint.split('/')[-2])
    lag_order = int(lag_order); horizon = int(horizon)
    image_model_name = args.model_checkpoint.split('/')[-3]
    image_name = args.model_checkpoint.split('/')[-4]
    match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})', args.model_checkpoint.split('/')[-1])
    if match:
        train_start = match.group(1)
        train_end = match.group(2)
    else:
        raise ValueError('the format of model checkpoint is false')
    
    if image_model_name == 'resnet18':
        model = torchvision.models.resnet18(num_classes=2)
    elif image_model_name == 'resnet34':
        model = torchvision.models.resnet34(num_classes=2)
    elif image_model_name == 'resnet50':
        model = torchvision.models.resnet50(num_classes=2)
    elif image_model_name == 'resnet101':
        model = torchvision.models.resnet101(num_classes=2)
    elif image_model_name == 'resnet152':
        model = torchvision.models.resnet152(num_classes=2)
    else:
        raise ValueError('model type error')
    model = model.cuda()
    
    model_weights = torch.load(
        f'{args.model_checkpoint}/network_best.pth', map_location=f'cuda:{args.cuda}'
    )
    weights_dict = {}
    for k, v in model_weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    model.load_state_dict(weights_dict)
    model.eval()
    
    oos_data_info = np.load(f'{args.data_root}/data_info/test_{image_name}_{train_start}_{train_end}_{lag_order}_{horizon}.npy')
    infer_dataset = Inference_Dataset(lag_order, horizon, oos_data_info)
    
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=args.batch,
        num_workers=args.worker,
        pin_memory=True,
    )
    
    iter_results = []
    
    CAM_extractor = SmoothGradCAMpp(model, 'layer4')
    time_transform = transforms.Resize((1, lag_order), antialias=True)
    
    for dates, stocks, images, labels in tqdm(infer_loader):
                
        out = model(images.cuda())
        positive_CAM = CAM_extractor(1, out)[0].mean(axis=1).unsqueeze(1)
        negative_CAM = CAM_extractor(0, out)[0].mean(axis=1).unsqueeze(1)
        
        with torch.no_grad():
            probs = F.softmax(out, dim=-1)
            pred = torch.argmax(probs, dim=-1)
                
            tem_index = pd.MultiIndex.from_tuples(zip(stocks, dates), names = ['c_code', 'date'])
            inference_tem = pd.DataFrame(columns = [f'CAM_{i}' for i in range(lag_order)] + ['pred_prob', 'pred', 'label'],
                                         index = tem_index)
            inference_tem['pred_prob'] = probs[:, -1].cpu().numpy()
            inference_tem['pred'] = pred.cpu().numpy()
            inference_tem['label'] = labels.numpy()
            
            mask = (pred.unsqueeze(-1).unsqueeze(-1)).expand_as(positive_CAM)
            pred_CAM = time_transform(positive_CAM * mask + negative_CAM * (1 - mask))
            
            inference_tem[[f'CAM_{lag_order - i - 1}' for i in range(lag_order)]] = (pred_CAM.squeeze(1)).cpu().numpy()
            
            iter_results.append(inference_tem)
    
    inference_table = pd.concat(iter_results, axis = 0)
    
    return inference_table

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=int, default=0,
                        help='ID of GPU device to be used')
    parser.add_argument('--batch', type=int, default=180, help='Batch size')
    parser.add_argument(
        '--model-checkpoint',
        type=str,
        default='Your/Model/Checkpoint/Path/Here', 
        help='Path of Save Model')
    parser.add_argument(
        '--worker',
        type=int,
        default=4,
        help='Number of processes used for loading data',
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='Your/Data/Path/Here',
        help='Directory of data',
    )

    args = parser.parse_args()

    inference_table = main(args)
    inference_table.to_csv(f'{args.model_checkpoint}/inference_table.csv.gz', compression = 'gzip')