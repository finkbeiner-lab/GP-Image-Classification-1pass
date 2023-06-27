"""
Created on July 2021

@author: Nadia Brancati

"""
import torch.utils.data as data
import torch
import os
import glob
from torchvision import transforms
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import pyfiglet
from skimage import measure
from tqdm import tqdm
from PIL import Image
from os.path import exists
import json
import pdb
import numpy as np
from skimage import draw
import argparse
import pandas as pd


class TensorDataset(data.Dataset):

    def __init__(self, root_dir, extension):
        self.root_dir=root_dir
        self.ext=extension
        self.classes, self.class_to_idx = self.find_classes()
        self.file_list = glob.glob(self.root_dir + "**/*."+self.ext)
        print("file list", self.file_list)

    def find_classes(self):
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        print(classes)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        file_name = os.path.join(self.root_dir, self.file_list[index])
        tensor_U = torch.load(file_name,map_location=torch.device("cpu"))
        tensor_U = tensor_U.squeeze()
        tensor_U = tensor_U.unsqueeze(0)
        running_label = os.path.basename(os.path.dirname(self.file_list[index]))
        label = torch.Tensor([self.classes.index(running_label)],device=torch.device("cpu")).to(torch.int64)
        return (tensor_U, label, file_name)

    def __len__(self):
        return len(self.file_list)


class ImageDataset(data.Dataset):

    def __init__(self, root_dir, patch, scale, overlap, device,extension,workers=10,level=16):
        #, segment=False, json_path=None, image_save_dir=None, mask_save_dir=None
        self.root_dir=root_dir
        #self.image_save_dir=image_save_dir
        #self.mask_save_dir=mask_save_dir
        self.patch=patch
        self.scale=scale
        self.overlap=overlap
        self.workers=workers
        self.ext=extension
        self.data_transform = transforms.Compose([
            transforms.Resize(self.scale),
            transforms.CenterCrop(self.scale),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # standard normalization
        ])
        self.device=device
        self.classes, self.class_to_idx = self.find_classes()
        self.file_list = glob.glob(self.root_dir+"**/*."+self.ext)
        self.level=level
        #self.ID_MASK_SHAPE = (patch, patch)
        #self.json_path = json_path
        #self.lablel2id = {'True':'50', 'Pre':'100','False':'150', 'Unknown':'0'}
        #self.segment = segment
        print(self.file_list)
        print(self.classes)



    def find_classes(self):
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def __getitem__(self, index):
        file_name = os.path.join(self.root_dir, self.file_list[index])
        slide = open_slide(os.path.join(self.root_dir,self.file_list[index]))
        tiles = DeepZoomGenerator(slide,tile_size=self.patch,overlap=self.overlap,limit_bounds=False)
        self.level = tiles.level_count-1
        W,H=tiles.level_tiles[self.level]
        print(tiles.level_dimensions)
        #creation of a tensor with size [in_H/patch_size, inW/patch_size] where in_H and in_W are height and width of the original image
        ris = torch.zeros([H, W, 3, self.patch, self.patch], device=self.device)
        #mask_ris = torch.zeros([H, W, 3, self.patch, self.patch], device=self.device)
        #mask_dict = dict()
        #label_df = pd.DataFrame()
        #if self.segment=="True":
        #    mask_dict, label_df = self.get_tile_masks_new(file_name, slide)
        #    print("images and masks created!!")
        #coords_map =dict()
        #arrangement the original images in a set of patch of size
        for w in range(W):
            for h in range(H):
                tile = tiles.get_tile(tiles.level_count-1,(w,h))
                tile_mod = self.data_transform(tile)
                ris[h,w]=tile_mod
                #tile_coords = tiles.get_tile_coordinates(tiles.level_count-1,(w,h))
                #if (self.segment=="True") & (tile_coords[0] in mask_dict.keys()):
                #    msk_img = Image.fromarray(np.uint8(mask_dict[tile_coords[0]])).convert('RGB')
                #    mask_ris[h,w]=self.data_transform(msk_img)
                #    coords_map[tile_coords[0]] = (h,w)
        running_label = os.path.basename(os.path.dirname(self.file_list[index]))
        label = torch.tensor([self.classes.index(running_label)], device=self.device)
        #return (ris, label, file_name, mask_dict, coords_map)
        return (ris, label,file_name)

    def __len__(self):
        return len(self.file_list)


    def tensor_and_info(self,index):
        file_name = os.path.join(self.root_dir, self.file_list[index])
        tensor=torch.load(self.file_list[index])
        running_label = os.path.basename(os.path.dirname(self.file_list[index]))
        label = torch.tensor([self.classes.index(running_label)], device=self.device)
        return (tensor,label,file_name)

    
'''
    def polygon2id(self, image_shape, mask, ids, coords_x, coords_y):
        vertex_row_coords, vertex_col_coords = coords_y, coords_x
        fill_row_coords, fill_col_coords = draw.polygon(
            vertex_row_coords, vertex_col_coords, image_shape)

        # Row and col are flipped
        mask[fill_col_coords, fill_row_coords] = ids
        return mask

    def save_img(self, img, file_name, tileX, tileY, save_dir, label="mask"):
        if label=="mask":
            im = Image.fromarray(img)
        else:
            im=img
        file_name = file_name + "_" + str(tileX)+"x" + "_" + str(tileY) + "y" + "_" + label + ".png"
        save_name = os.path.join(save_dir, file_name)
        im.save(save_name)
        return save_name

    def get_tile_masks_new(self, file_name, slide):
        #file_name = os.path.join(self.root_dir, self.file_list[index])
        #slide = open_slide(os.path.join(self.root_dir,self.file_list[index]))
        json_file_name = os.path.basename(file_name) + ".json"
        json_file_name = os.path.join(self.json_path, json_file_name)
        if not exists(json_file_name):
            print("Json File does not exist")
        with open(json_file_name) as f:
            data = json.load(f)
        
        mask_dict=dict()
        label_df = pd.DataFrame(columns=["file_name","tile_coords","label","Image_path","Mask_path"])
        for ele in tqdm(data):
            # Reset ids for each annotation
            ids = 1
            # Create an Empty mask of size similar to image
            id_mask = np.zeros(self.ID_MASK_SHAPE, dtype=np.uint8)
            region_id = 0
            prev_label = ""
            label = ele['label']
            i = 0
            #plaque_dict[ele['label']] = plaque_dict[ele['label']] + len(ele['region_attributes'])
            for region in ele['region_attributes']:
                # Get tileX and tileY
                tileX = region['tiles'][0]['tileId'][0]
                tileY = region['tiles'][0]['tileId'][1]
                tileWidth = region['tiles'][0]['tileBounds']["WH"][0]
                tileHeight = region['tiles'][0]['tileBounds']["WH"][1]
                # crop the image
                tileX = (tileX * tileWidth)
                tileY = (tileY * tileHeight)
                coords_x, coords_y = zip(*region['points'])
                coords_x = np.array(coords_x)
                coords_y = np.array(coords_y)
                x1 = tileX
                x2 = tileX + tileWidth 
                y1 = tileY 
                y2 = tileY + tileHeight
                # Remove overlap annotations
                if len(coords_x[coords_x > x2]) > 0 or len(coords_y[coords_y > y2]) > 0:
                    print('Overlap')
                    continue
                # Translate the coordinates to fit within the image crop
                coords_x = np.mod(coords_x, tileWidth)
                coords_y = np.mod(coords_y, tileHeight)
                # label
                label = ele['label']
                if i == 0:
                    ids = int(self.lablel2id[label])
                elif label == prev_label:
                    ids = int(self.lablel2id[label])
                # Use polygon2id function to create a mask
                id_mask = self.polygon2id(self.ID_MASK_SHAPE, id_mask, ids, coords_y, coords_x)
                # ids = ids + 5
                prev_label = label
                i+=1
                mask_dict[(tileX,tileY)] = id_mask
                region = slide.read_region((tileX,tileY), 0, (self.patch, self.patch))
                #img_crop_dict[(tileX,tileY)] = vips_img_crop
                save_img_name = self.save_img(region, ele['filename'], tileX, tileY, self.image_save_dir, "image")
                save_mask_name = self.save_img(id_mask, ele['filename'], tileX, tileY, self.mask_save_dir,"mask")
                label_df.loc[len(label_df.index)] = [file_name,(tileX,tileY),label,save_img_name,save_mask_name] 
                id_mask = np.zeros(self.ID_MASK_SHAPE, dtype=np.uint8)   
        return mask_dict, label_df
'''


    