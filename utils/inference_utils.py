import os
import sys
import time
import warnings
from copy import copy, deepcopy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import random
import albumentations as A
import pickle as pkl
import numpy as np
import skimage.io
import itertools
import math
import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_pdf import PdfPages

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# # ROOT = ROOT.relative_to(Path.cwd())  # relative

ROOT = "/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/projects/yolov5_mask_hierarchical_small"
sys.path.append(ROOT)
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.downloads import attempt_download
from models.common import *  
from models.experimental import *
from utils.vis_funcs import _draw_roi, _overlay_masks
from utils.datasets import convert_cell_types
from utils.datasets_combine import class2id#, convert_cell_types
from utils.utils_image import ColorDodge, ColorJitter, pad, pad_if_needed, crop_if_needed, rescale, img_as, \
    get_pad_width
from utils.augmentations import augment_hsv
from utils.general import xyxy2xywhn, xywh2xyxy, masks2xyxy, check_img_size, box_iou, check_version, \
    non_max_suppression, LOGGER
from utils.utils_LUAD import rgba2rgb, split_masks, normal_cell_to_stroma, RGB_TO_LABEL

sys.path.append("/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/github/hierarchical-classification/hierarchical_classification/")
from dataset_utils import DataDict

def attempt_load(weights, map_location=None, inplace=True, fuse=True):#, w_mask=False):
    # if w_mask:
    from models.yolo_mask import Detect, Model, Segment
    # else:
    #     from models.yolo_hier import Detect, Model
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        if fuse:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
        else:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble


#################### Dataset ######################
# CELL_TYPE_CONVERT = {  # old -> new class names (should fit the keys in COLORS (color dictionary))
#     'tumor nuclei': 'non-mitotic tumor',
#     'mitotic figure': 'mitotic tumor'
# }

def get_img_size(dset, scale_ratio, stride, imgs=None, im_paths=None, verbose=False):
    assert imgs or im_paths
    
    if dset.upper() == 'PANNUKE':
        maxh = max([im.shape[0] for im in imgs])
        maxw = max([im.shape[1] for im in imgs])
        minh = min([im.shape[0] for im in imgs])
        minw = min([im.shape[1] for im in imgs])
    else:
        maxh, maxw = 0, 0
        minh, minw = 10000, 10000
        for f_im in im_paths:
            im = skimage.io.imread(f_im)  
            h, w, _ = im.shape
            maxh = max(maxh, h)
            maxw = max(maxw, w)
            minh = min(minh, h)
            minw = min(minw, w)
    max_dim = max([maxh, maxw])
    if verbose:
        print(f'image height range: ({minh}, {maxh}), image width range: ({minw}, {maxw})')
        print(f'maximum dim: {max_dim}')
    img_sz = check_img_size(int(scale_ratio*max_dim), s=stride)
    print(f"scale ratio: {scale_ratio}, image size: {img_sz}")
    return img_sz


class Dataset(torch.utils.data.Dataset):
    __initialized = False

    def __init__(self, img_files_info, classes, imgs=None, img_files=None, class_ids=None, label_files=None, 
                img_size=640, batch_size=16, hyp=None, stride=32, pad=0.0, pad_mode="reflect", keep_res=False, patch_size=None,
                verbose=False, scale_ratio=1.0, mags=None):
        
        self.img_files_info = img_files_info
        self.imgs           = imgs.astype(np.uint8) if imgs is not None else imgs
        self.img_files      = img_files

        self.classes     = classes
        self.class_ids   = class_ids if class_ids else [i for (i, _) in enumerate(classes)]
        self.label_files = label_files
        self.img_size    = img_size
        self.hyp         = hyp
        self.stride = stride
        self.verbose = verbose
        self.scale_ratio = scale_ratio
        self.mags = mags if mags is not None else [40]*len(img_files_info) 
        '''self.augment = hyp['shape_aug'] if augment and ('shape_aug' in hyp) else augment
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.cell_type_convert = cell_type_convert'''

        n = len(img_files_info)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int64)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        self.keep_res = keep_res
        self.patch_size = patch_size if patch_size else self.img_size
        self.pad_mode = pad_mode

        '''self.albumentations = A.Compose([
                                    A.Blur(p=0.01),  # only changes the image
                                    A.MedianBlur(p=0.01),  # only changes the image
                                    A.ToGray(p=0.01),  # only changes the image
                                    A.CLAHE(p=0.01),  # only changes the image
                                    A.RandomBrightnessContrast(p=0.0),  # only change the image
                                    A.RandomGamma(p=0.0),  # only change the image
                                    A.ImageCompression(quality_lower=75, p=0.0)],  # only change the image
                                )'''

    def __len__(self):
        return len(self.img_files_info)

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        scale_ratio = self.scale_ratio*(40/self.mags[index])

        # Read image
        if self.imgs is not None:
            img0 = self.imgs[index]

        else:
            path = self.img_files[index]
            img0 = skimage.io.imread(path)  # RGB or RGBA (for CoNSeP)
            if self.img_files_info[index] == 'CoNSeP':
                img0 = rgba2rgb(img0)
            assert img0 is not None, f'Image Not Found {path}'
        h0, w0 = img0.shape[:2]  # orig hw
        

        # Load label if exists
        if self.label_files is not None:
            path_t = self.label_files[index]

            if self.img_files_info[index] == 'NuCLS':
                labels = pkl.load(open(path_t, 'rb'))  # 'masks', 'class_nms', 'boxes', 'lb_typs'
                # convert cell types
                class_nms = labels['class_nms']
                '''if self.verbose:
                    print(f'before convert: {np.unique(class_nms)}')

                class_nms = convert_cell_types(class_nms, self.cell_type_convert)
                if self.verbose:
                    print(f'after convert: {np.unique(class_nms)}')'''

                objects = {
                    'masks': labels['masks'],
                    'labels': np.array(class2id(class_nms, self.classes, inds=self.class_ids)[0]),
                    'lb_typs': labels['lb_typs'],
                    'class_nms': class_nms,
                }

            elif self.img_files_info[index] == 'CoNSeP' or self.img_files_info[index] == 'PanNuke':
                labels = pkl.load(open(path_t, 'rb'))  # 'masks', 'class_nms'
                class_nms = labels['class_nms']
                objects = {
                    'masks': labels['masks'],
                    'labels': np.array(class2id(class_nms, self.classes, inds=self.class_ids)[0]),
                    'lb_typs': np.array(['polyline' for x in class_nms], dtype='object'),
                    'class_nms': class_nms
                }

            elif self.img_files_info[index] == 'LUAD':
                target  = skimage.io.imread(path_t)
                target  = normal_cell_to_stroma(rgba2rgb(target))
                objects = split_masks(target, val_to_label=RGB_TO_LABEL, dtype='float')
                if len(objects):
                    masks = np.transpose(np.array([x['mask'] for x in objects]), (1, 2, 0))
                    labels = [x['label'] for x in objects]
                else:
                    # Create a mask list [(h, w, 1)] and an empty label list
                    masks = np.zeros((img0.shape[0], img0.shape[1], 1))
                    labels = []
                objects = {
                    'masks': masks,
                    'labels': np.array(labels),
                    'lb_typs': np.array(['polyline' for x in labels], dtype='object')
                }
        else:
            objects = {}

        #         img, objects = rescale(img0, objects, scale=1.0 * self.scale_ratio)
        #         img = rescale(img0, scale=1.0 * self.scale_ratio)

        img = skimage.transform.rescale(img0, scale=1.0 * scale_ratio,
                                        order=3, anti_aliasing=True, 
#                                         preserve_range=True, 
                                        multichannel=True)
        img = img_as(dtype=img0.dtype)(img)

        if self.label_files:
            masks = objects['masks']
            masks = skimage.transform.rescale(masks, 1.0 * scale_ratio,
                                             order=0, anti_aliasing=True, 
                                              preserve_range=True, 
                                              multichannel=True)
            # filter out small objects
            keep_ids = masks.sum(axis=(0, 1)) >= 2 * scale_ratio
            if self.verbose:
                print(f"shape of masks: {masks.shape}")
                print(f"{keep_ids.sum()} out of {len(keep_ids)} objects kept.")
            
            masks = masks[:, :, keep_ids]
            if self.verbose:
                print(f"shape of masks after filtering small objects: {masks.shape}")
            objects['labels'] = objects['labels'][keep_ids]
            if 'lb_typs' in objects:
                objects['lb_typs'] = objects['lb_typs'][keep_ids]

            # convert from masks to labels (nl, 5)
            xyxy, keep_ids = masks2xyxy(masks)
            assert len(keep_ids) == masks.shape[-1]
            objects['labels'] = objects['labels'][keep_ids]
            masks = masks[:, :, keep_ids]
            if 'lb_typs' in objects:
                objects['lb_typs'] = objects['lb_typs'][keep_ids]
            
            nl = xyxy.shape[0]  # number of labels
            labels_out = torch.zeros((nl, 6))
            sample_weights = np.ones(nl)
            if nl:
                # convert from masks to labels (nl, 5)
                labels = np.zeros((nl, 5))
                labels[:, 0] = objects['labels']
                #             targets['masks'] = targets['masks'][:,:,keep_ids]
                #             print(f"values of masks: {np.unique(targets['masks'])}")
                h_, w_ = img.shape[:2]  # current hw
                labels[:, 1:] = xyxy2xywhn(xyxy / 1., w=w_, h=h_, clip=True, eps=1E-3)
                labels_out[:, 1:] = torch.from_numpy(labels)
                if 'lb_typs' in objects:
                    sample_weights[np.where(objects['lb_typs'] == 'rectangle')[0]] = 0
            
        else:
            labels_out = torch.zeros((0, 6))
            sample_weights = np.empty(0)
            masks = np.zeros((img.shape[0], img.shape[1], 1))

        # pad image
        # TODO update pad_if_needed function to return also pad width
        '''img, objects = pad_if_needed(img, objects, (self.img_size,self.img_size), pos='center', mode='constant')'''
        
        pad_widths = get_pad_width(img.shape, (self.img_size, self.img_size), pos='center')
        img = np.pad(img, pad_widths, mode=self.pad_mode)
        masks = np.pad(masks, pad_widths, mode='constant', constant_values=0.0)
        masks = masks.transpose((2, 0, 1))  # .unsqueeze(axis=0)
        
        # Convert
        img = img.transpose((2, 0, 1))  # HWC to CHW,
        img = np.ascontiguousarray(img)
        img = img.copy()
            
        temp_file = self.img_files[index] if self.img_files is not None else self.label_files[index]
        return (torch.from_numpy(img), torch.from_numpy(img0), labels_out, temp_file, pad_widths,
                torch.from_numpy(masks), torch.from_numpy(sample_weights), scale_ratio)

    @staticmethod
    def collate_fn(batch):
        img, img0, label, path, pad_widths, masks, sample_weights, scale_ratio = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), img0, torch.cat(label, 0), path, pad_widths, masks, torch.cat(sample_weights), scale_ratio


def build_dataloader(im_paths, target_paths, img_sz, im_info, classes, class_ids, batch_size, hyp=None, augment=False,
                     stride=32, keep_res=False, patch_size=None, mosaic=False, verbose=False, scale_ratio=1.0,
                     shuffle=False, num_workers=0, cell_type_convert={}):
    dataset = Dataset(img_files=im_paths, img_files_info=im_info, classes=classes, class_ids=class_ids,
                      label_files=target_paths, img_size=img_sz, batch_size=batch_size, hyp=hyp, augment=augment,
                      stride=stride, keep_res=keep_res, patch_size=patch_size, mosaic=mosaic, verbose=verbose,
                      scale_ratio=scale_ratio, cell_type_convert=cell_type_convert)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            collate_fn=dataset.collate_fn)
    return dataset, dataloader


## combine datasets (create data_dicts)
def get_data_dicts(test_data_full, classes, model_stride=64, split="test", batch_size=4, hier_model=False, tree_cls=None, root_only=False,
                  scale_ratio=1.0):
    """ This function combines multiple datasets and outputs a dictionary of data loaders
    args: 
    test_data_full: a list of test datset yaml files
    classes: model classes
    """
    if hier_model:
        assert tree_cls is not None
    data_dicts = {}
    test_sets    = {}
    test_loaders = {}
    test_sets_eval    = {}
    test_loaders_eval = {}
    
    for test_data in test_data_full:
        LOGGER.info("dataset: {}".format(test_data))

        data_yaml = yaml.safe_load(open(test_data, errors='ignore'))
        dset = data_yaml['name']
        
        ## create a "data_dict" class object
        data_dict = DataDict(dataname=data_yaml['name'], classes=data_yaml['names'], 
                             data2model=data_yaml['data2model'], class_ids=None, 
                             ignore=data_yaml['ignore'] if 'ignore' in data_yaml else None
                            )
                            
        LOGGER.info(f"dataset classes: {', '.join(data_dict.classes)}.")  # CLASSES_data
        LOGGER.info(f"dataset class indices: {', '.join(map(str, data_dict.class_ids))}.") # CLASS_IDS_data
        
        data_cls2cid = dict((cls, cid) for cid, cls in enumerate(data_dict.classes))
        setattr(data_dict, "cid2cls", dict((cid, cls) for cls, cid in data_dict.cls2cid.items()))
        LOGGER.info("data to model: {}".format(data_dict.data2model))
        
        if hier_model:
            # if hasattr(model_infer, "tree_cls") and getattr(model_infer, "tree_cls") is not None:
            data_dict.get_data2model_idx(tree_cls)
            
            #if hasattr(train_opt, "root_only") and train_opt.root_only:
            if root_only:
                data_dict.d2m_idx_root = dict((idx_d, tree_cls.iparents[ids_m[0]][0].item()) \
                        for idx_d, ids_m in data_dict.d2m_idx.items())
                data_dict.d2m_root = dict((data_dict.classes[i], tree_cls.classes[j]) \
                                          for (i,j) in data_dict.d2m_idx_root.items())
                #     name_data = data_yaml['name']
                data_dict.get_data2model_root(tree_cls)
                LOGGER.info("Data to model, indices, root level: {}".format(data_dict.d2m_idx_root))
                LOGGER.info("Data to model, indices, root level: {}".format(data_dict.d2m_root))

        else:
            data_dict.d2m_idx = dict((data_dict.classes.index(x), classes.index(y[0])) 
            for (x,y) in data_dict.data2model.items())
            LOGGER.info("Data to model, indices: {}".format(data_dict.d2m_idx))
            
        if dset == 'PanNuke':
            # load all 3 folds of images
            fold_ims = {}
            for fold in [1,2,3]:
                fold_ims[str(fold)] = np.load(os.path.join(data_yaml['dir_data'], data_yaml[f'dir_img_fold{fold}']))

            target_paths_test = pd.read_csv(data_yaml[f'target_{split}'], sep=' ', header=None)[0].tolist()

            folds_test = [x.split('/')[-2].split('_')[-1] for x in target_paths_test]
            ids_test   = [int(x.split('/')[-1].split('.')[0]) for x in target_paths_test]

            test_ims  = [fold_ims[fold][im_id,...] for fold, im_id in zip(folds_test, ids_test)] #all_ims[split]
            info_test = [dset for _ in target_paths_test] #all_info[split]

            LOGGER.info(f"\n{dset} dataset, {len(test_ims)} {split} images, {len(target_paths_test)} {split} targets\n")

        else:

            df_im_paths_test= pd.read_csv(data_yaml['img_test'], sep=' ', header=None)
            im_paths_test, info_test = df_im_paths_test[0].tolist(), df_im_paths_test[1].tolist()
            target_paths_test = pd.read_csv(data_yaml['target_test'], sep=' ', header=None)[0].tolist()
            LOGGER.info(f"\n{dset} dataset, {len(im_paths_test)} test images, {len(target_paths_test)} test targets")
       
        # get the suitable image sizes
        im_paths = None if dset.upper() == 'PANNUKE' else im_paths_test
        imgs     = test_ims if dset.upper() == 'PANNUKE' else None
        '''scale_ratio = opt.img_size/opt.patch_size'''
        img_sz = get_img_size(dset, scale_ratio=scale_ratio, stride=model_stride, 
                              im_paths=im_paths, imgs=imgs, verbose=True)
        
        if dset.upper() == 'PANNUKE':
            test_dataset_eval = Dataset(img_files_info=info_test, classes=data_dict.classes, imgs=np.stack(test_ims), 
                                        img_files=None, class_ids=data_dict.class_ids, label_files=target_paths_test, 
                                        img_size=img_sz, batch_size=batch_size, scale_ratio=scale_ratio)

            test_dataset = Dataset(img_files_info=info_test, classes=data_dict.classes, imgs=np.stack(test_ims), 
                                   img_files=None, class_ids=data_dict.class_ids, label_files=None, img_size=img_sz, 
                                   batch_size=batch_size, scale_ratio=scale_ratio)
            
        else:
            if dset.upper() == "LUAD":
                ids, im_paths, target_paths, test_dataset_eval, test_dataset = {}, {}, {}, {}, {}
                ids[40] = [i for i, x in enumerate(im_paths_test) if not os.path.basename(x).startswith("N")]
                ids[20] = [i for i, x in enumerate(im_paths_test) if os.path.basename(x).startswith("N")]
                
                for mag in [40, 20]:
                    im_paths[mag] = [im_paths_test[i] for i in ids[mag]]
                    target_paths[mag] = [target_paths_test[i] for i in ids[mag]]
                    test_dataset_eval[mag] = Dataset(img_files_info=[dset]*len(im_paths[mag]), classes=data_dict.classes, imgs=None,
                                                      img_files=im_paths[mag], class_ids=data_dict.class_ids, 
                                                      label_files=target_paths[mag], 
                                                      img_size=int(img_sz*(40/mag)), batch_size=batch_size, 
                                                      scale_ratio=scale_ratio, mags=[mag]*len(im_paths[mag]))
                    test_dataset[mag] = Dataset(img_files_info=[dset]*len(im_paths[mag]), classes=data_dict.classes, imgs=None, 
                                                 img_files=im_paths[mag], class_ids=data_dict.class_ids, 
                                                 label_files=None,  img_size=int(img_sz*(40/mag)), batch_size=batch_size, 
                                                 scale_ratio=scale_ratio, mags=[mag]*len(im_paths[mag]))
            else:
                test_dataset_eval = Dataset(img_files_info=info_test, classes=data_dict.classes, imgs=None, img_files=im_paths_test, 
                                            class_ids=data_dict.class_ids, label_files=target_paths_test, img_size=img_sz, 
                                            batch_size=batch_size, scale_ratio=scale_ratio)
                test_dataset = Dataset(img_files_info=info_test, classes=data_dict.classes, imgs=None, img_files=im_paths_test, 
                                       class_ids=data_dict.class_ids, label_files=None, img_size=img_sz, 
                                       batch_size=batch_size, scale_ratio=scale_ratio)

        if isinstance(test_dataset, dict):
    #         test_sets[dset], test_loaders[dset], test_sets_eval[dset], test_loaders_eval[dset] = {}, {}, {}, {}
            for mag, test_set in test_dataset.items():
                test_sets_eval[f"{dset}-{mag}"]    = test_dataset_eval[mag]
                test_loaders_eval[f"{dset}-{mag}"] = DataLoader(test_dataset_eval[mag], batch_size=batch_size, shuffle=False, num_workers=0, 
                                          collate_fn=test_set.collate_fn)
                test_sets[f"{dset}-{mag}"]    = test_set
                test_loaders[f"{dset}-{mag}"] = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, 
                                          collate_fn=test_set.collate_fn)
        else:
            
            test_loader_eval = DataLoader(test_dataset_eval, batch_size=batch_size, shuffle=False, num_workers=0, 
                                          collate_fn=test_dataset_eval.collate_fn)
            test_loader      = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, 
                                          collate_fn=test_dataset_eval.collate_fn)
        
            test_sets[dset]    = test_dataset
            test_loaders[dset] = test_loader
            test_sets_eval[dset]    = test_dataset_eval
            test_loaders_eval[dset] = test_loader_eval
        
        data_dicts[dset] = data_dict
    return data_dicts, test_sets, test_loaders, test_sets_eval, test_loaders_eval

#################### Visualization function ############################
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          pdf_obj=None,
                          savename=None,
                          figsize=(5, 5)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1E-6)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    f = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, horizontalalignment='right')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if pdf_obj:
        pdf_obj.savefig(f, bbox_inches='tight')

    elif savename:
        plt.savefig(savename, bbox_inches='tight')
    #     else:
    plt.show()


def boxes2contours(boxes):
    '''
    args:
        boxes: x1y1x2y2
    '''
    contours = []
    for box in boxes:
        x1, y1, x2, y2 = box
        pt1, pt2, pt3, pt4 = [x1, y1], [x1, y2], [x2, y2], [x2, y1]
        contour = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
        contours.append(contour)
    return contours

def crop_boxes(boxes, crop_left=0, crop_top=0):
    '''
    args:
        boxes: x1y1x2y2
        crop_left: crop for left
        crop_top: crop for top
    '''
    boxes[:, [0, 2]] -= crop_left
    boxes[:, [1, 3]] -= crop_top
    return boxes


def postprocess_boxes(boxes, pads, size_im_m):  # , size_im):

    '''post process detection boxes
    pads   = (pad_l, pad_r, pad_t, pad_b)
    size_im_m = (height_m, width_m)
    size_im   = (height, width)
    1) crop boxes
    2) x1y1x2y2 -> cxywhn
    3) cxywhn -> cxywh (correspond to the original image size)'''
    pad_l, pad_r, pad_t, pad_b = pads
    height_m, width_m = size_im_m
    #     height, width              = size_im

    # normalize output x1y1x2y2 -> cxywhn
    # crop back boxes
    boxes = crop_boxes(boxes, crop_left=pad_l, crop_top=pad_t)
    boxes = xyxy2xywhn(boxes, w=width_m - pad_l - pad_r, h=height_m - pad_t - pad_b)
    return boxes


def vis_boxes_masks(ot, img, LABEL_TO_RGB=None, colors=None):  # , gt=None, ot_no_nms=None):
    '''
    ot: dict
    '''
    assert "boxes" in ot, "Unable to visualize boxes if boxes not provided!"

    img_mask = img.copy() if 'masks' in ot else None
        
    height, width, _ = img.shape

    # cxywhn -> cxywh
    boxes_ot = ot["boxes"] * torch.Tensor([width, height, width, height])  # .to(device)

    # cxywh -> xyxy
    boxes_ot = xywh2xyxy(boxes_ot)

    # xyxy -> contour
    contours_ot = boxes2contours(boxes_ot.numpy().astype(np.int64))


    assert LABEL_TO_RGB or colors
    if colors is None:
        if "labels" in ot:
            colors = [LABEL_TO_RGB[int(i)] for i in ot["labels"]]
        else:
            warnings.warn("labels not provided, colors are not imformative!")
            colors = [LABEL_TO_RGB[0] for i in ot["boxes"]]
    img = _draw_roi(img, contours_ot, colors=colors)
    if 'masks' in ot:
        img_mask = _overlay_masks(img_mask, ot["masks"], colors=colors)
    
    return img, boxes_ot, contours_ot, img_mask


def vis_boxes(ot, img, LABEL_TO_RGB=None, colors=None):  # , gt=None, ot_no_nms=None):
    '''
    ot: dict
    '''
    assert "boxes" in ot, "Unable to visualize boxes if boxes not provided!"

    height, width, _ = img.shape

    # cxywhn -> cxywh
    boxes_ot = ot["boxes"] * torch.Tensor([width, height, width, height])  # .to(device)

    # cxywh -> xyxy
    boxes_ot = xywh2xyxy(boxes_ot)

    # xyxy -> contour
    contours_ot = boxes2contours(boxes_ot.numpy().astype(np.int64))

    #     vis_tles  = ["prediction"]
    #     pred_imgs = [img.numpy().copy()]
    #     img_ot = img.numpy().copy()

    assert LABEL_TO_RGB or colors
    if colors is None:
        if "labels" in ot:
            colors = [LABEL_TO_RGB[int(i)] for i in ot["labels"]]
        else:
            warnings.warn("labels not provided, colors are not imformative!")
            colors = [LABEL_TO_RGB[0] for i in ot["boxes"]]
    img = _draw_roi(img, contours_ot, colors=colors)
    return img, boxes_ot, contours_ot

def postprocess_detections(self, preds, tree_cls=None, #d2m_converts=None,
                           data_dict=None, masks_logits=None):
    """ postprocess model predictions, i.e. model outputs ->
    args:
        self = yolomodel
        preds        = model output
        tree_cls     = TreeCls object correspond to the yolomodel; if not provided, non-hier model(default=None)
        data_dict    = (default=None)
        masks_logits = (default=None)

    """

    if tree_cls is None:
        softmax = nn.Softmax(dim=-1)

    outputs = []
    if masks_logits is not None:
        n_obj_per_image = [len(_) for _ in preds]
        mask_probs = masks_logits.sigmoid().split(n_obj_per_image, dim=0)
    else:
        mask_probs = [None] * len(preds)

    for p, m in zip(preds, mask_probs): # loop over each output
        boxes  = p[:, :4]
        scores = p[:, 4]
        logits = p[:, 5:]

        # convert from raw class logits to conditional & absolute probabilities
        r = {'boxes': boxes.detach().cpu(),
             'scores': scores.detach().cpu(),
             'logits': logits.detach().cpu(),
             'labels': []
             }

        if logits.shape[0] > 0:
            if tree_cls is not None:
                if data_dict is not None:
                    cond_lb, absol_lb, cond_lb_tree, absol_lb_tree, cond_probs, absol_probs = \
                        tree_cls.pred_batch_data(logits.cpu(), data_dict, return_probs=True)
                    r = {**r, **{'labels': cond_lb[:, -1].detach().cpu(),
                                 'absol_labels': absol_lb.detach().cpu(),
                                 'cond_probs': cond_probs,
                                 'absol_probs': absol_probs.detach().cpu(),
                                 'labels_tree': cond_lb_tree.detach().cpu(),
                                 'absol_labels_tree': absol_lb_tree.detach().cpu(),
                                 'labels_hier': cond_lb.detach().cpu()
                                 }}
                else:
                    cond_lb, absol_lb, cond_probs, absol_probs = \
                        tree_cls.pred_batch(logits.cpu(), return_probs=True)

                    r = {**r, **{
                        'labels': cond_lb[:, -1].detach().cpu(),
                        'absol_labels': absol_lb.detach().cpu(),
                        'cond_probs': cond_probs.detach().cpu(),
                        'absol_probs': absol_probs.detach().cpu(),
                    }}
            else:
                cond_probs = softmax(logits)  # softmax on logits
                absol_probs = cond_probs
                labels = p[:, 5:].argmax(axis=1).long() if p.shape[0] else torch.empty(0)
                labels = labels.detach().cpu()
                r = {**r, **{
                    'labels': labels,
                    'cond_probs': cond_probs.detach().cpu(),
                    'absol_probs': absol_probs.detach().cpu()
                }}

            if m is not None:
                if self.model[self.seg_l].nc > 1:  # = self.model[self.det_l].nc:
                    if tree_cls is not None:
                        labels = torch.tensor([x[-1] for x in labels])
                    index = torch.arange(len(m), device=labels.device)
                    m = m[index, labels.long()][:, None]
                else:
                    m = m[index, 0][:, None]
                r['masks'] = m.detach().cpu()
                # r['masks'] = paste_masks_in_image(m, boxes, img_shape, padding=1).squeeze(1)

        outputs.append(r)

    return outputs


def inference_on_loader(model, dataloader,
                        # classes, class_ids,
                        color_dict,
                        nms_params, tree_cls=None, data_dict=None, trial=False,
                        pdf_obj=None, ignore_idx=-100,
                        pdf_name=None, model_type="non-hier", device=0,
                        key_label='labels', convert_gt=False):
    """

    Args:
        model:
        dataloader:
        classes: classes in ground truth (if data_dict is not None, classes should correspond to classes in data_dict)
        color_dict:
        nms_params:
        tree_cls:
        data_dict:
        trial:
        pdf_obj:
        ignore_idx:
        pdf_name:
        model_type:
        device:
        key_label:
        convert_gt:

    Returns:

    """

    results = {}
    gts = {}
    total_time, seen = 0., 0.
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else "cpu")

    if model_type == "hier" and (data_dict is not None):
        cid2cls = deepcopy(data_dict.cid2cls)
        if hasattr(data_dict, "d2m_idx_"):  # "extras" -> in tree but not in dataset
            for cid, mid in data_dict.d2m_idx_.items():
                cid2cls[cid] = model.cid2cls[mid[0]]
    else:
        cid2cls = model.cid2cls
        if model_type == "root":
            convert_gt = True
            assert data_dict is not None and hasattr(data_dict, "d2m_idx_root")
            d2m_root = data_dict.d2m_idx_root  # for gt conversion
            import pdb; pdb.set_trace()
        else:  # model_type == "non_hier"
            print(model_type)
            cid2cls_gt = dict((cid, data_dict.data2model[cls][0] if cls in data_dict.data2model else cls) \
                              for cid, cls in data_dict.cid2cls.items())

    if pdf_obj is None:
        if pdf_name is not None:
            pred_pdf = PdfPages(pdf_name)

    for imgs, img0s, labels, paths, pad_widths, masks_gt, sample_weights, scale_ratios in dataloader:
        imgs = imgs.float()  # uint8 to fp16/32 im.half() if half else
        imgs /= 255  # 0 - 255 to 0.0 - 1.0

        # Inference
        st = time.time()
        if hasattr(model, "seg_l") and model.seg_l is not None:
            (pred_det, train_out), mask_logits = model.forward(imgs.to(device))
        elif hasattr(model, "seg_l") and model.seg_l is None:
            (preds, train_out), _ = model.forward(imgs.to(device))
            pred_det, pred_det_no_nms = non_max_suppression(preds, **nms_params)
        else:
            preds, train_out = model.forward(imgs.to(device))
            ############################ non-max suppression ############################
            # TODO: update the NMS function
            pred_det, pred_det_no_nms = non_max_suppression(preds, **nms_params)

        ############################ end of NMS ############################
        rois = [_[:, :4] for _ in pred_det]

        outputs = postprocess_detections(model, pred_det, tree_cls=tree_cls, data_dict=data_dict)
        total_time += time.time() - st
        seen += imgs.shape[0]

        # process predictions and visualize
        for si, (path_i, img0, img, output, pad_widths_i, masks_gti, scale_rat) in enumerate(zip(
                paths, img0s, imgs, outputs, pad_widths, masks_gt, scale_ratios)):
            LOGGER.info(path_i)

            pad_t, pad_b = pad_widths_i[0]
            pad_l, pad_r = pad_widths_i[1]

            _, height_m, width_m = img.shape
            height, width, _ = img0.shape

            img = img.cpu().detach().numpy().transpose((1, 2, 0))
            pred_boxes = output['boxes']

            # raw predictions (correspond to image size in the model)
            contours_pd_ = boxes2contours(pred_boxes)

            # process boxes in output -> cxywhn (correspond to original image size)
            # output['boxes']
            output['boxes'] = postprocess_boxes(output['boxes'], (pad_l, pad_r, pad_t, pad_b), (height_m, width_m))
            keep_ids = torch.where((output['boxes'] < 0).sum(axis=1) == 0)[0]
            #             import pdb; pdb.set_trace()
            for k, item in output.items():
                if not isinstance(item, dict):
                    output[k] = output[k][keep_ids, ...]

            lb_ot = output[key_label]  # correspond to dataset if data_dict provided, otherwise, correspond to tree

            if len(lb_ot) == 0:
                continue

            # convert from label ids to labels (lb_ot -> cls_ot)
            cls_ot = [cid2cls[cid.item()] for cid in lb_ot]

            """if dset.lower() == 'consep':
                if 'other' in cls_ot:
                    cls_ot[cls_ot.index('other')] = 'consep_other'"""

            ############## Visualization: boxes ###############
            pred_img_ = (img * 255).astype(np.uint8).copy()
            pred_img = img0.numpy().copy()
            gt_img = img0.numpy().copy()

            try:
                pred_img, box_ot, contours_ot = vis_boxes(output, pred_img, colors=[color_dict[cls] for cls in cls_ot])
            except:
                import pdb; pdb.set_trace()

            ucids, counts = torch.unique(lb_ot, return_counts=True)
            ucls = [cid2cls[i.item()] for i in ucids]

            if labels.shape[0] > 0:  # if ground truth available: process gt

                # i-th sample of the batch
                ids = labels[:, 0] == si
                labels_i = labels[ids, :]
                masks_gti = masks_gti.cpu().numpy().transpose((1, 2, 0))

                # crop
                masks_gti = masks_gti[pad_t:height_m - pad_b, pad_l:width_m - pad_r, :]
                # resize
                masks_gti = skimage.transform.rescale(masks_gti, 1.0 / scale_rat,
                                                      order=0, anti_aliasing=True,
                                                      preserve_range=True,
                                                      multichannel=True)
                # dict for gt
                gti = {
                    'boxes': labels_i[:, 2:],  # cxywhn
                    'labels': torch.tensor([ignore_idx if _ == ignore_idx else d2m_root[int(_)] for _ in
                                            labels_i[:, 1]]) if convert_gt else labels_i[:, 1],  # labels_,
                    'masks': masks_gti
                }
                try:
                    cls_gt = [cid2cls_gt[int(i)] for i in gti['labels']]
                except:
                    import pdb; pdb.set_trace()
                    cls_gt = [cid2cls[int(i)] for i in gti['labels']]
                    # cls_gt = [classes[int(i)] if not i == ignore_idx else 'unlabeled' for i in gti['labels']]

                """if dset.lower() == 'consep':
                    if 'other' in cls_gt:
                        cls_gt[cls_gt.index('other')] = 'consep_other'"""

                colors_gt = [color_dict[cls] for cls in cls_gt]
                gt_img, box_gt, contours_gt, gt_img_mask = vis_boxes_masks(gti, gt_img, colors=colors_gt)

                ucids_gt, counts_gt = torch.unique(gti['labels'], return_counts=True)
                try:
                    ucls_gt = [cid2cls_gt[int(i)] for i in ucids_gt]
                    
                except:
                    ucls_gt = [cid2cls[int(i)] for i in ucids_gt]
                    # ucls_gt = ['unlabeled' if i == ignore_idx else classes[i] for i in ucids_gt.long()]

                fig, axs = plt.subplots(1, 4, figsize=(12, 5))
                for j, (cid_gt, cl_gt, count) in enumerate(zip(ucids_gt.long(), ucls_gt, counts_gt)):
                    axs[3].text(0.75, 0.6 - j * 0.05, "{} ({}): {}".format(cl_gt, cid_gt, count),
                                transform=fig.transFigure, size=12)
                axs[3].axis('off')
                axs[2].imshow(gt_img_mask)  # gt_img)
                axs[2].set_title("ground truth")
                gti['box_xyxy'] = box_gt
                gts[path_i] = gti

            else:  # only show and/or save inference visualization
                fig, axs = plt.subplots(1, 2, figsize=(6, 5))

            for j, (cid, cl, count) in enumerate(zip(ucids, ucls, counts)):
                axs[0].text(0.05, 0.6 - j * 0.05, "{} ({}): {}".format(cl, cid, count), transform=fig.transFigure,
                            size=12)
            axs[0].axis('off')
            axs[1].imshow(pred_img)
            axs[1].set_title("prediction")
            plt.suptitle(path_i.split('/')[-1])

            if pred_pdf is not None:
                pred_pdf.savefig(fig, bbox_inches='tight')

            ############## save output (& gt) ##############
            output['box_xyxy'] = box_ot
            results[path_i] = output

        if trial:
            break
        else:
            plt.close('all')
    LOGGER.info(f'Inferenced %f images in %.3fs, average inferencing speed: %.3fs ' % (int(seen),
                                                                                       total_time,
                                                                                       total_time / seen))
    pred_pdf.close()
    return results, gts
