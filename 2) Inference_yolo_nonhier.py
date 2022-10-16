#!/usr/bin/env python
# coding: utf-8
import re
import argparse
import copy
from collections import ChainMap, namedtuple, OrderedDict
import yaml
import skimage
import os
import pandas as pd
import torch
import glob
import pickle as pkl
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import time
import numpy as np
from datetime import datetime
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys
sys.path.append("/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/github/hierarchical-classification/hierarchical_classification/")
from dataset_utils import DataDict
from loss_utils import TreeLoss
from hierarchical_utils import TreeCls



# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# # ROOT = ROOT.relative_to(Path.cwd())  # relative

ROOT = "/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/projects/yolov5_mask_hierarchical_small"
sys.path.append(ROOT)
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.metrics import compute_ap
from models.experimental import Ensemble
from models.common import Conv
from utils.general import check_version, non_max_suppression, LOGGER, xywh2xyxy, xyxy2xywhn, box_iou
from utils.autoanchor import check_anchor_order
from utils.torch_utils import initialize_weights, model_info
# from utils.hierarchical_utils import TreeCls, TreeLoss

# visualization and evaluation functions
from utils.vis_funcs import _draw_roi, _overlay_masks, display_segmentation, display_detection, COLORS


### Inference helper functions
from utils.inference_utils import (Dataset, attempt_load, plot_confusion_matrix, boxes2contours, 
                                   crop_boxes, vis_boxes, postprocess_boxes, get_img_size, get_data_dicts,
                                   inference_on_loader)

### Evaluation Helper Functions
from utils.evaluate_utils import (evaluate_classification, evaluate_detection, plot_confusion_matrix, 
                                _evaluate_detection, _update_cm, evaluate_dset)

# Add colors
# LUAD
COLORS['tumor nuclei'] = COLORS['tumor']
COLORS['non-macrophage immune cell'] = COLORS['lymphocyte nuclei']
# CoNSeP & PanNuke

COLORS['dysplastic/malignant epithelial'] = COLORS['tumor']
COLORS['neoplastic cells'] = COLORS['tumor']
COLORS['endothelial'] = COLORS['vascular endothelium']
COLORS['healthy epithelial'] = COLORS['normal epithelium']
COLORS['inflammatory'] = [4, 51, 255]
COLORS['epithelial'] = COLORS['normal epithelium']
COLORS['connective/soft tissue cells'] = COLORS['stromal cell']
COLORS['dead cell'] = COLORS['dead nuclei']
COLORS['ductal epithelium'] = COLORS['normal epithelium']

COLORS['mitotic figure'] = COLORS['mitotic tumor']
COLORS['dead cells'] = COLORS['dead nuclei']
COLORS['consep_other'] = COLORS['lipid']
    
old2new = {
            'tumor nuclei': "tumor",
            'stroma nuclei': "stromal cell",
            'lymphocyte nuclei': "immune cell",
            'red blood cell': "immune cell",
            'macrophage nuclei': "immune cell",
            'ductal epithelium': "other",
            'dead nuclei':"apoptotic body"
        }
 
#### Inputs
def parse_options():
    parser = argparse.ArgumentParser('Inference YOLO w/o hierarchical classification', add_help=False)
    parser.add_argument('--savedir', type=str,
                        default='/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/projects/yolov5_mask_hierarchical_small/notebooks/results/non_hier/imsize640_batch16_2022-01-17-15-21/',
                        help='directory of data')
    parser.add_argument('--inferdir', type=str, default='', help='inference directory if different from savedir')
    parser.add_argument('--img_size', type=int, default=640, help='size of the image')
    parser.add_argument('--patch_size', type=int, default=320, help='size of the patch')
    parser.add_argument('--best_epoch', type=makelist, default='303, 361, 373, 418', help='best performing model indices')
    parser.add_argument('--batch_size', type=int, default='2',
                        help='validation batch size')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--w_mask', action='store_true')
    parser.add_argument('--hier_model', action='store_true')
    parser.add_argument('--device', default=0, type=int, help='device, -1 if using CPU')
    parser.add_argument('--thres_iou', type=float, default=0.3, help='iou threshold to match with ground truth')
    parser.add_argument('--update_tree_cls', action='store_true', help='a flag indicating whether or not update the tree')

    return parser

def makelist(string):
    return [float(_) for _ in string.split(',')]
    

class options:
    def __init__(self):    
        #### NuCLS ####
        self.savedir = "./results/non_hier/imsize640_batch16_2022-01-17-15-21/" # [303, 361, 373, 418]
  
        self.inferdir         = None


def main(opt):
    for key, val in vars(opt).items():
        print("{}: {}".format(key, val))
    device = torch.device(f'cuda:{opt.device}' if torch.cuda.is_available() else "cpu")

    # load options used in training
    try:
        train_opt_f = os.path.join(opt.savedir, "opt.yaml")
        train_opt = yaml.load(open(train_opt_f), Loader=yaml.FullLoader)
        if "hier_model" not in train_opt:
            train_opt["hier_model"] = False
        # Convert from dictionary to named object
        train_opt = namedtuple("TrainOpt", train_opt.keys())(*train_opt.values())
        # dir(train_opt)
    except:
        train_opt = None
    LOGGER.info("Model directory: {}. \nBest epochs: {}".format(opt.savedir, ', '.join(map(str, opt.best_epoch))))

    if train_opt.hier_model and hasattr(train_opt, "root_only") and train_opt.root_only:
        LOGGER.info("\nTrained w/ root only")

    #### Prepare data
    if train_opt is not None: 
        if hasattr(train_opt, "include"):
            data_train = [train_opt.data[i] for i in train_opt.include]
        else:

            data_train = train_opt.data

        print("trained on {}".format(" + ".join([x.split("/")[-1].split(".")[0] for x in data_train])                              if isinstance(data_train,list)                              else data_train.split("/")[-1].split(".")[0]))

    if opt.update_tree_cls:
        test_data_full = [
            "../datasets/old_tree/NuCLS_data.yaml",
            "../datasets/old_tree/LUAD_data.yaml"
        ]
    elif opt.hier_model or (train_opt.hier_model and hasattr(train_opt, "root_only") and train_opt.root_only):
        test_data_full = [
            "../datasets/combine/LUAD_data.yaml",
            "../datasets/combine/NuCLS_data.yaml",
    #         "../datasets/combine/PanNuke_data.yaml",
    #         "../datasets/combine/CoNSeP_data.yaml",
        ]

    else:
        test_data_full = [
            "../datasets/non_hier/NuCLS_data.yaml",
            "../datasets/non_hier/LUAD_data.yaml"
        ]
    print("validation dataset(s): \n{}".format('\n'.join(test_data_full)))

    
    for iep in range(len(opt.best_epoch)): # for each model
        ### Model
        # load trained ckpt
        epoch = int(opt.best_epoch[iep])
        if len(opt.inferdir) == 0:# is None:
            opt.weights = os.path.join(opt.savedir, 'weights', f'epoch{epoch}.pt')
        else:
            opt.weights = os.path.join(opt.savedir, f'{epoch}.pt')
        # ckpt = torch.load(f_last, map_location=device) 

        LOGGER.info("Loading weights from epoch: {}".format(epoch))

        model_infer = attempt_load(opt.weights, map_location=device, fuse=False)#, w_mask=opt.w_mask)
        stride = int(model_infer.stride.max())  # model stride

        # (optional) define detection &/ segmentation layer(s)
        if hasattr(model_infer, 'seg_l') and model_infer.seg_l:
            det_idx = model_infer.det_l
            seg_idx = model_infer.seg_l
            seg = model_infer.model[seg_idx]
        else: # default, no segmentation layer
            det_idx = -1
        det = model_infer.model[det_idx]
        LOGGER.info("Modeled classes: {}".format(', '.join(model_infer.names)))
        
        # assign model_type
        convert = True
        if hasattr(train_opt, "hier_model") and train_opt.hier_model:
            if hasattr(train_opt, "root_only") and train_opt.root_only:
                print("Overwrite")
                opt.hier_model = False # overwrite
                opt.model_type = "root"
                convert = False
            else:
                opt.hier_model = True # overwrite
                assert(train_opt.hier_model == opt.hier_model)
                opt.model_type = "hier"
        else: # trained non-hier
            if opt.hier_model == True:
                print("Warning: overwrite... Cannot apply an hierarchical model using a model trained w/ non-hierarchical loss!")
            assert opt.hier_model == False
            opt.model_type = "non-hier"
        LOGGER.info("model type: {}, hier model: {}".format(opt.model_type, opt.hier_model))


        # (optional: only for model trained before updating TreeCls) update tree cls and save   
        print(opt.hier_model)
        print(opt.update_tree_cls)
        if opt.hier_model and opt.update_tree_cls:
            LOGGER.info(":) here")
            sys.path.append("/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/github/hierarchical-classification/hierarchical_classification/")
            from hierarchical_utils import TreeCls
            yaml.dump(model_infer.tree_cls.tree, open(os.path.join(opt.savedir, "tree.yaml"), 'w'))
            tree_cls_infer = TreeCls(model_infer.tree_cls.tree)
            model_infer.tree_cls = tree_cls_infer

        # model classes
        if hasattr(model_infer, "tree_cls") and model_infer.tree_cls is not None:
            if hasattr(train_opt, "root_only") and train_opt.root_only:
                CLASSES   = model_infer.tree_cls.classes_root
                CLASS_IDS = [model_infer.tree_cls.cls2cid[x] for x in CLASSES]
            else:
                CLASSES   = model_infer.tree_cls.classes
                CLASS_IDS = model_infer.tree_cls.class_ids
        else:
            CLASSES = model_infer.module.names if hasattr(model_infer, 'module') else model_infer.names  # get class names ['tumor nuclei', 'stroma nuclei', 'lymphocyte nuclei', 'red blood cell', 'macrophage nuclei','dead nuclei']  #
            CLASS_IDS = [i for (i, n) in enumerate(CLASSES)]
        setattr(model_infer, "classes", CLASSES)
        setattr(model_infer, "class_ids", CLASS_IDS)
        setattr(model_infer, "cid2cls", dict((cid, cls) for (cid, cls) in zip(CLASS_IDS, CLASSES)))
            
        LOGGER.info("model classes indices - class")
        for cls, cid in zip(CLASSES, CLASS_IDS):
            LOGGER.info(f"{cid} - {cls}")


        # nms_params = {"conf_thres": 0.15, "iou_thres": 0.45, "max_det": 1000}#getattr(model_infer, "nms_params", )

        nms_params = {"conf_thres": 0.25, "iou_thres": 0.45, "max_det": 1000}#getattr(model_infer, "nms_params", )
        # nms_params["max_det"] = int(np.ceil(400*(img_sz/512)*(img_sz/512)))
        # nms_params["max_det"] = int(np.ceil(1000*(img_sz/512)*(img_sz/512)))
        # LOGGER.info("image size: {}x{}, non-max suppression parameters: {}.".format(img_sz, img_sz, nms_params))
        LOGGER.info("non-max suppression parameters: {}.".format(nms_params))
        if hasattr(model_infer, "nms_params"):
            LOGGER.info("non-max suppression parameters used in training: {}.".format(model_infer.nms_params))
            model_infer.nms_params = nms_params

        if iep == 0:
            data_dicts, test_sets, test_loaders, test_sets_eval, test_loaders_eval = \
            get_data_dicts(test_data_full, classes=CLASSES, model_stride=stride, split="test",  
            batch_size=opt.batch_size, hier_model=opt.hier_model, tree_cls=None, root_only=False, 
            scale_ratio = opt.img_size/opt.patch_size)

            set_params = "conf_{}_iou_{}_maxdet_{}_covg_iou_{}".format(nms_params['conf_thres'], 
                                                             nms_params['iou_thres'], 
                                                             nms_params['max_det'],
                                                             opt.thres_iou
                                                            )
            LOGGER.info("Inference set parameters: {}".format(set_params))
            
            if opt.model_type == "root": #opt.hier_model and hasattr(train_opt, "root_only") and train_opt.root_only:
                for dnm, data_dict in data_dicts.items():
                    data_dicts[dnm].d2m_idx_root = dict((idx_d, model_infer.tree_cls.iparents[ids_m[0]][0].item()) \
                    for idx_d, ids_m in data_dict.d2m_idx.items())

        ### Inference

        # update inference directories depended on model
        if opt.w_mask:
            model_infer.seg_l = None
        infer_dirs = {}
        for dset, data_dict in data_dicts.items():
            LOGGER.info("dataset: {}".format(dset))
            infer_dir = os.path.join(opt.savedir, f'infer/model{epoch}/{dset}')
                
            # inference saving directory
            infer_dir = "{}_{}".format(infer_dir, set_params)
            if not os.path.exists(infer_dir):
                os.makedirs(infer_dir)
            LOGGER.info(f"saving inference results to {infer_dir}")
            
            infer_dirs[dset] = infer_dir
            
        # if opt.hier_model and (not (hasattr(train_opt, "root_only") and train_opt.root_only)) and hasattr(model_infer, "tree_cls"):
        #     tree_cls_infer = model_infer.tree_cls
        # else:
        #     tree_cls_infer = None
        if opt.model_type == 'non-hier':
            tree_cls_infer = None
        else:
            tree_cls_infer = model_infer.tree_cls
            
        CLASSES_, CLASS_IDS_  = CLASSES, CLASS_IDS


        '''#### Test inference

        for key, test_loader_eval in test_loaders_eval.items():
            infer_dir = infer_dirs['LUAD'] if 'LUAD' in key else infer_dirs[key] 
            data_dict = data_dicts['LUAD'] if 'LUAD' in key else data_dicts[key]
            pdf_name = os.path.join(infer_dir, "test_wgt.pdf")
            pdf_name = pdf_name.replace(".pdf", "_{}x.pdf".format(key.split("LUAD-")[1])) if 'LUAD' in key else pdf_name
            if opt.hier_model:
                CLASSES_  = data_dict.classes
                CLASS_IDS_ = data_dict.class_ids
                
            results, gts = inference_on_loader(model_infer, test_loader_eval, 
        #                                    classes=CLASSES_, class_ids=CLASS_IDS_, 
                                           color_dict=COLORS, 
                                           nms_params=nms_params,#model_infer.nms_params,
                                           tree_cls=tree_cls_infer,
                                           data_dict=data_dict, trial=True, ignore_idx=-100, 
        #                                    data_prob=data_prob,
                                           pdf_name=pdf_name,
                                           model_type=opt.model_type, device=opt.device, key_label='labels', 
                                               convert_gt=False)'''


        #### Inference and save outputs (to pdf files)
        # print(data_dicts.keys(), test_loaders_eval.keys())
        all_results, all_gts = {}, {}

        for dset, test_loader_eval in test_loaders_eval.items():
            infer_dir = infer_dirs['LUAD'] if 'LUAD' in dset else infer_dirs[dset] 
            data_dict = data_dicts['LUAD'] if 'LUAD' in dset else data_dicts[dset]
            pdf_name  = os.path.join(infer_dir, "pred_wgt.pdf")
            pdf_name = pdf_name.replace(".pdf", "_{}x.pdf".format(dset.split("LUAD-")[1])) if 'LUAD' in dset else pdf_name
            
            if opt.hier_model:
                CLASSES_   = data_dict.classes
                CLASS_IDS_ = data_dict.class_ids
                print("\n{}\nclasses: {}\nclass_ids: {}".format(dset, ', '.join(CLASSES_), ', '.join(map(str, CLASS_IDS_))))

            results, gts = inference_on_loader(model_infer, test_loader_eval, 
                                           color_dict=COLORS, 
                                           nms_params=nms_params,#model_infer.nms_params,
                                           tree_cls=tree_cls_infer,
                                           data_dict=data_dict, trial=False, ignore_idx=-100, 
                                           pdf_name=pdf_name,
                                           model_type=opt.model_type, device=opt.device, key_label='labels', 
                                               convert_gt=False)
            # update gts if root only
            if opt.model_type == "root":
                for key, gt_ in gts.items():
                    gt_ids = gt_['labels'] # label ids in dataset 
                    gt_ids_root = torch.tensor([x if x in data_dict.ignore_idx else data_dict.d2m_idx_root[int(x)] for x in gt_ids])
                    gts[key]['labels_'] = gt_ids_root # root label ids in tree
            else:
                for key, gt_ in gts.items():
                    gt_ids  = gt_['labels'] # label ids in dataset 
                    gt_ids_ = [int(x) if x in data_dict.ignore_idx else data_dict.d2m_idx[int(x)] for x in gt_ids]
                    gts[key]['labels_'] = gt_ids_ # label ids in tree
            
            suffix_save = "_{}".format(dset.split("LUAD-")[1]) if 'LUAD' in dset else ""
            if tree_cls_infer is not None:
                data_prob = "cond"
                suffix_save += "_{}".format(data_prob)

            # # save outputs and gts ??
            pkl.dump(results, open(os.path.join(infer_dir, "results{}.pkl".format(suffix_save)), 'wb'))
            try:
                pkl.dump(gts, open(os.path.join(infer_dir, "gts{}.pkl".format(suffix_save)), 'wb'))
            except:
                pkl.dump(gts, open(os.path.join(infer_dir, "gts{}.pkl".format(suffix_save)), 'wb'), protocol=4)
                
            all_results[dset] = results
            all_gts[dset]     = gts
            
            
        # combine 20x 40x for LUAD
        if any([re.search("LUAD", x) for x in all_gts.keys()]):
            _gts_ = [t for k, t in all_gts.items() if "LUAD" in k]
            _ots_ = [t for k, t in all_results.items() if "LUAD" in k]
            all_gts_ = copy.deepcopy(all_gts)
            all_results_ = copy.deepcopy(all_results)
            for k in all_gts_.keys():
                if "LUAD" in k:
                    all_gts.pop(k)
                    all_results.pop(k)
            all_gts["LUAD"]     = dict(ChainMap(*_gts_))
            all_results["LUAD"] = dict(ChainMap(*_ots_))

        ### Evaluate
        #### Evaluate mAP
        # mAP
        mAPs = dict()
        thres = np.arange(0.5,1.05,0.05)
        verbose = False

        for dset, all_result in all_results.items():
            all_gt = all_gts[dset]
                
            mAPs[dset] = {}

            perfs = np.zeros((len(thres), 3)) # tp, fp, fn
            prs = np.zeros((len(thres), 2)) # tp, fp, fn
            boxes_gt = [x['box_xyxy'] for x in all_gt.values()]
            boxes_ot = [x['box_xyxy'] for x in all_result.values()]
            for i, thres_iou in enumerate(thres): # for each threshold
                perfs[i,:], prs[i,:], _ = evaluate_detection(boxes_ot, boxes_gt, thres_iou=thres_iou, verbose=verbose)
            ap, mpre, mrec = compute_ap(prs[:,1], prs[:,0])
            mAPs[dset] = {
                "recalls": prs[:,1],
                "precisions": prs[:,0],
                "ap": ap, 
                "mpre": mpre, 
                "mrec": mrec,
                "perf": perfs
            }

            LOGGER.info("mAP for {}: {}".format(dset, ap))
            pkl.dump(mAPs[dset], open(os.path.join(infer_dirs[dset], "mAPs.pkl"),"wb"))

        all_eval  = {}
        key_ot_lb = 'labels' 

        # get the list of new classes, class indices
        classes_old = model_infer.classes
        classes_new, class_ids_new, ct = [], [], 0
        for cls in classes_old:
            if cls not in old2new:
                continue 
            if old2new[cls] not in classes_new:
                classes_new.append(old2new[cls])
                class_ids_new.append(ct)
                ct += 1
        # get the conversion dictionaries from new to old
        new2old, new2old_idx = {}, {}
        for clso, clsn in old2new.items():
            if clsn not in new2old:
                new2old[clsn] = []
                new2old_idx[classes_new.index(clsn)] = []
            new2old[clsn].append(clso)
            new2old_idx[classes_new.index(clsn)].append(classes_old.index(clso))


        all_cms  = {}
        all_eval = {}
        suffix_convert = "root"
        title_convert = "Root level"
        all_new_cms = {}

        for dset, results in all_results.items():    
            LOGGER.info("\ndataset: {}".format(dset))
            gts       = all_gts[dset]
            data_dict = data_dicts[dset]
            infer_dir = infer_dirs[dset] 
        #     data_dict = data_dicts['LUAD'] if 'LUAD' in dset else data_dicts[dset]
            if opt.model_type == "non-hier":
                classes_confu, class_ids_confu = model_infer.classes, model_infer.class_ids
                if len(data_dict.ignore_idx) > 0:
                    for _idx, _idn in zip(data_dict.ignore_idx, data_dict.ignore):
                        if (_idx not in class_ids_confu) and (_idn not in classes_confu):
                            classes_confu.append(_idn)
                            class_ids_confu.append(_idx)

                key_gt_lb = "labels_"
            else:
                classes_confu, class_ids_confu  = CLASSES, CLASS_IDS
                key_gt_lb = "labels"
                
            suffix_save = "" 
            if tree_cls_infer is not None:
                data_prob = "cond"
                suffix_save += "_{}".format(data_prob)
            
            if opt.model_type == "hier":
                class_ids_confu = copy(data_dict.class_ids)
                classes_confu   = copy(data_dict.classes)
                class_ids_confu += list(data_dict.d2m_idx_.keys())
                classes_confu   += [model_infer.tree_cls.classes[x[0]] for x in data_dict.d2m_idx_.values()]
            
            all_eval[dset], all_cms[dset] = evaluate_dset(results, gts, classes_confu, class_ids_confu, 
                                                          key_ot_lb, key_gt_lb, thres_iou=opt.thres_iou, 
                                                          infer_dir=infer_dir, suffix_save=suffix_save, verbose=False, 
                                                          ignore=data_dict.ignore, ignore_idx=data_dict.ignore_idx)
            
            if convert:
                confu_pdf_conv = PdfPages(os.path.join(infer_dir, 'confusion_{}.pdf'.format(suffix_convert)))

                cm_old = all_cms[dset]['all'].loc[classes_old, classes_old].to_numpy()

                cm_new = _update_cm(cm_old, new2old_idx, classes_new)

                plot_confusion_matrix(cm_new, classes=classes_new, figsize=(8,8),
                            normalize=False,
                            title=title_convert,
                            pdf_obj=confu_pdf_conv
                            )
                plot_confusion_matrix(cm_new, classes=classes_new, figsize=(8,8),
                            normalize=True,
                            title= "Overall accuracy: {:.3f} ({}/{})".format(np.diag(cm_new).sum()/cm_new.sum(), 
                                                                       np.diag(cm_new).sum(), cm_new.sum()),
                            pdf_obj=confu_pdf_conv
                            )
                all_new_cms[dset] = cm_new
                confu_pdf_conv.close()




    # ## Combine
    dir_out = os.path.join(opt.savedir, "infer")
    LOGGER.info("Saving result summary comparison to {}".format(dir_out))
    df_all_sum = pd.DataFrame(columns=["dataset", "coverage", "overall accuracy", "correct", "total", "f_score"])

    dfs_sum = {}
    for dir_model in glob.glob(os.path.join(dir_out, "model*/")): # loop over each model 
        epoch = dir_model.split('/')[-2].split('model')[-1]
        for dir_dset in glob.glob(os.path.join(dir_model, "*{}*".format(set_params))): # loop over each dataset with this parameter setting
            dset_ = dir_dset.split("/")[-1].split(set_params)[0]
            
            f_sum = glob.glob(os.path.join(dir_dset, "summary*.csv"))
            if len(f_sum) == 0:
                continue
            if dset_ not in dfs_sum:
                # dfs_sum[dset_] = pd.DataFrame(columns=["epoch", "coverage", "overall accuracy", "correct", "total", "f_score"])
                dfs_sum[dset_] = pd.DataFrame(columns=["epoch", "coverage", "overall accuracy", "f_score", "precision", "recall"])

            df_sum = pd.read_csv(f_sum[0], header=1)
            df_sum = df_sum.transpose()
            df_sum.reset_index(inplace=True)
            df_sum.columns = df_sum.loc[0,:]
            df_sum.drop(0, axis=0, inplace=True)
    #         try:
            dfs_sum[dset_] = dfs_sum[dset_].append({"epoch": epoch, 
    #                                                 "coverage": pd.to_numeric(df_sum['coverage'].values[0]), 
                                                    "coverage": pd.to_numeric(df_sum['recall'].values[0]), 
                                                    "overall accuracy": pd.to_numeric(df_sum['accuracy'].values[0]), 
    #                                         "correct": pd.to_numeric(df_sum['correct'].values[0]), 
    #                                         "total": pd.to_numeric(df_sum['total'].values[0]), 
                                                    "f_score": pd.to_numeric(df_sum['f_score'].values[0]),
                                                    "precision": pd.to_numeric(df_sum['precision'].values[0]),
                                                    "recall": pd.to_numeric(df_sum['recall'].values[0]),
                                                   }, ignore_index=True)
    #         except:
    #             pass
        
    xl_writer = pd.ExcelWriter(os.path.join(dir_out, 'infer_comparison_{}.xlsx'.format(set_params)), engine='xlsxwriter')
    for dset_, df in dfs_sum.items():
        df.to_excel(xl_writer, sheet_name=dset_)
    xl_writer.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference for YOLO", parents=[parse_options()])
    args = parser.parse_args()
    main(args)

