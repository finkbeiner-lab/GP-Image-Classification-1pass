import argparse
import numpy as np
from TensorDataset import *
import ABNN_WSI
from torch.utils.data.sampler import SubsetRandomSampler
from modelsMinMax import *
import torch.optim as optim
import random
import torchvision
from sklearn.metrics import *
import unet


def main(args):
    #inizializaiont of the environment and set of seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    shuffle=True
    #if mode is TRAIN
    if args.mode == "TRAIN":
        if args.segment=="True":
            unet.run_segmentation(args)

        if args.classify=="True":
            ABNN_WSI.run_classification(args)
        

    #if mode is TENSOR
    if args.mode == "TENSOR":
        model=None
        # if the model chosen for the creation of the tensor is RESNET18
        if args.model_type == "RESNET18":
            #if the original pretained model is chosen
            if args.model_pretrained:
                model = torchvision.models.resnet18(pretrained=True)
            # if a new pretained model is chosen
            else:
                model = torchvision.models.resnet18()
                model.fc = nn.Linear(512, 3)
                model.load_state_dict(torch.load(args.model_path))
            #parameters are not trained
            for param in model.parameters():
                param.requires_grad = False
            #last layer for the classification is deleted: only features are extracted to create the tensor
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            model.cuda(device=device)
        # if the model chosen for the creation of the tensor is RESNET34
        if args.model_type == "RESNET34":
            # if the original pretained model is chosen
            if args.model_pretrained:
                model = torchvision.models.resnet34(pretrained=True)
            # if a new pretrained model is chosen
            else:
                model = torchvision.models.resnet34()
                model.fc = nn.Linear(512, 3)
                model.load_state_dict(torch.load(args.model_path))
            # parameters are not trained
            for param in model.parameters():
                param.requires_grad = False
            # last layer for the classification is deleted: only features are extracted to create the tensor
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            model.cuda(device=device)
        #creation of the tensors
        ABNN_WSI.tensors_creation(model,args,device=device)
    print("END")


if __name__ == '__main__':
    MODEL_PATH = "path-to-save/load-the-model"
    DATA_DIR = "path-for-loading-images/tensors"
    SAVE_DIR = "path-to-save-tensors"
    parser = argparse.ArgumentParser(description='Training a model')

    # General parameters
    parser.add_argument('--model_type', choices=['RESNET18','RESNET34'],default="RESNET34",help="Models used to create the Tensor_U [RESNET18,RESNET34] ")
    parser.add_argument('--model_pretrained', help='if original pretrained model this parameter should be set to True')
    parser.add_argument('--model_path', default=MODEL_PATH, help='path of the model saved for each epoch')
    parser.add_argument('--model_path_fin', default=MODEL_PATH, help='path of the final saved model')
    parser.add_argument('--data_dir',default=DATA_DIR, help='path of the train dataset')
    parser.add_argument('--val_dir', default=DATA_DIR, help='path of the validation dataset')
    parser.add_argument('--test_dir', default=DATA_DIR, help='path of the test dataset')
    parser.add_argument('--aug_dir', default=DATA_DIR, help='path of the first dataset for the augmentation')
    parser.add_argument('--aug_dir2', default=DATA_DIR, help='path of the second dataset for the augmentation')
    parser.add_argument('--save_dir', default=SAVE_DIR, help='path of the directory where tensors will be saved')
    parser.add_argument('--mode', choices=['TRAIN','TENSOR'], default="TRAIN", help="possible options: TRAIN and TENSOR")
    parser.add_argument('--seed', type=int, default=1, help='Seed value')
    parser.add_argument('--gpu_list', default="0", help='number of the GPU that will be used')
    parser.add_argument('--debug', action='store_true', help='for debug mode')
    parser.add_argument('--ext', default='pth', help='extension of the structure to load: svs/png for images (mode=TENSORS) and pth for tensors (mode=TRAIN)')

    # Training parameters
    parser.add_argument('--patch_size', type=int, default=224, help='Patch Size')
    parser.add_argument('--patch_scale', type=int, default=224, help='Patch Scale')
    """ 
    parser.add_argument('--num_epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
    
    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--filters_out', type=int, default=64, help='number of Attention Map Filters')
    parser.add_argument('--filters_in', type=int, default=512, help='number of Input Map Filters')
    """
    parser.add_argument('--json_path', help='Path to annotated json files for WSIs')
    parser.add_argument('--image_save_dir', help='Path to save cropped images')
    parser.add_argument('--mask_save_dir', help='Path to save cropped masks')
    parser.add_argument('--images_dir', help='Path to save image and mask folder')

    parser.add_argument('--segment', type=str, default="False", help='if doing segmentation - True else False')
    parser.add_argument('--classify', type=str, default="True", help='if doing classification - True else False')



    args = parser.parse_args()
    print(args)
    main(args)


#python ABNN_WSI.py --mode TRAIN --data_dir '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/train/' --val_dir /gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/val/ --test_dir /gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/test/ --aug_dir '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/train/' --aug_dir2 '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/train/' --gpu_list 0 --seed 0 --model_path '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/GigaPixel-paper/model_epochs' --model_path_fin '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/GigaPixel-paper/final' --batch_size 8 --learning_rate 0.0001 --ext pth 
#ython ABNN_WSI.py --mode TENSOR --model_type RESNET34 --model_pretrained True --model_path '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/GigaPixel-paper/model_epochs' --data_dir '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/train/' --gpu_list 0 --seed 0 --save_dir '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/GigaPixel-paper/HE-WSI' --ext mrxs