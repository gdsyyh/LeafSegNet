import torch
import os,cv2
from tqdm import tqdm
from metrics import eval
from torch.utils.data import DataLoader
from metrics import Metrics
from torch.utils.data import Dataset, DataLoader
from train import DATASET
from torchvision import transforms
from model.UNet import UNet
from model.UNetPlusPlus import NestedUNet
from model.core.res_unet_plus import ResUnetPlusPlus
from model.core.res_unet import ResUnet
from model.CGNet import Context_Guided_Network as CGNet
from model.STUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from model.STUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from model.MACUNet import MACUNet
from model.LMFFNet import LMFFNet
from model.MANet import MANet
from model.seg_model import Seg_Network as TwoStage

from utils import load_data

#计算IOU要在sigmod之后，返回值为1个
def test():
    print('loading data......')
    path = "dataset"
    (train_x, train_y), (valid_x, valid_y) = load_data(path)
    val_total_batch = int(len(valid_x) / 1)
    valid_dataset = DATASET(valid_x, valid_y, (256, 256), transform=None)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    modelDicName = 'ViT_seg_2_'
    # model = UNet(classes=1)
    # model = CGNet(classes=1)
    # model = NestedUNet(num_classes=1, input_channels=3, deep_supervision=False)
    # model = ResUnetPlusPlus(3)

    # 以下双引号部分为ST-UNet
    # config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
    # config_vit.n_classes = 1
    # config_vit.n_skip = 3
    # if "R50-ViT-B_16".find('R50') != -1:
    #     config_vit.patches.grid = (
    #     int(256 / 16), int(256 / 16))
    # in_channels = config_vit['decoder_channels'][-1]
    # model = ViT_seg(config_vit, img_size=256, num_classes=config_vit.n_classes)
    # model = TwoStage(in_channels=3, out_channels=1)

    # model = MACUNet(3, 1)
    # model = LMFFNet(classes=1)
    model = MANet(3, 1)


    model.load_state_dict(torch.load('log/'+modelDicName+'checkpoint.pth'))
    if not os.path.exists("results\\"+modelDicName):
        os.makedirs(r'results\\'+modelDicName)
    model.cuda()
    model.eval()
    # metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean', 'Dice'])
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_loader):

            x = x.to(torch.device('cuda'), dtype=torch.float32)
            y = y.to(torch.device('cuda'), dtype=torch.float32)

            pred = model(x)
            pred = torch.sigmoid(pred)
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean, Dice= eval(pred, y, 0.5)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision,
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly,
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean, Dice = Dice
                        )
            #################################
            # pred = torch.sigmoid(output)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred_draw = pred.clone().detach()
            mask_draw = y.clone().detach()
            img_id = i
            img_numpy = pred_draw.cpu().detach().numpy()[0][0]
            img_numpy[img_numpy == 1] = 255
            cv2.imwrite(f'results/'+modelDicName+"/"+str(img_id)+'_pred.png', img_numpy)
            mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
            mask_numpy[mask_numpy == 1] = 255
            cv2.imwrite(f'results/'+modelDicName+"/"+str(img_id)+'_gt.png', mask_numpy)
            #################################
    metrics_result = metrics.mean(val_total_batch)

    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
          'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, Dice: %.4f'
          % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
             metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
             metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean'], metrics_result['Dice']))

if __name__ == '__main__':
    test()
    print('Done')
