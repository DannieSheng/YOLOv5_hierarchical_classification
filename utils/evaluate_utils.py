import os
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from utils.general import box_iou
from matplotlib.backends.backend_pdf import PdfPages
from copy import copy
import sys
ROOT = "/archive/DPDS/Xiao_lab/shared/hudanyun_sheng/projects/yolov5_mask_hierarchical_small"
sys.path.append(ROOT)
from utils.general import LOGGER

def evaluate_classification(labels_det_ot, labels_det_gt, class_ids, ignore_idx=-100):
    """ A function to evaluate the performance of a classifier
    Args:
        labels_det_ot: ground truth class labels (detected instances) 
        labels_det_gt: output class labels (detected instances) 
        class_ids: class indices
        ignore_idx: class index to be ignored (if any), default -100
    Returns:
        eval_cls: classification evaluation output dictionary
            - accuracy: overall accuracy
            - per_cls_accuracy: per class accuracy
            - cm: confusion matrix
            - correct: # of correctly classified objects
            - total: total number of objects (ignored excluded if any)
    """

    '''labels_det_ot    = np.array(labels_ot)[matched_id_ot]
    labels_det_gt = np.array(labels_gt)[matched_id_gt] '''
    
    cm = confusion_matrix(labels_det_gt, labels_det_ot, labels=class_ids)

    if ignore_idx in class_ids:
        ind = class_ids.index(ignore_idx)
        num_corr, num_obj = np.diagonal(cm[0:ind, 0:ind]).sum()+np.diagonal(cm[ind+1:, ind+1:]).sum(), \
        cm[0:ind, 0:ind].sum()+cm[ind+1:, ind+1:].sum()
        accu_per_cls = np.diagonal(cm)/(cm.sum(axis=1)+ 1E-6)
        accu_per_cls = np.concatenate((accu_per_cls[0:ind], accu_per_cls[ind+1:]))
    else:
        num_corr, num_obj = np.diagonal(cm).sum(), cm.sum()
        accu_per_cls      = np.diagonal(cm)/cm.sum(axis=1)
        
    eval_cls = {
        "accuracy": num_corr / num_obj, # accu
        "per_cls_accuracy": accu_per_cls, # per_cls accuracy
        "cm": cm,
        "correct": num_corr,
        "total": num_obj
    }
    
    return eval_cls


def _evaluate_detection(boxes_ot, boxes_gt, thres_iou=0.5, verbose=False):
    """A function to evaluate the goodness of object detection (for a single image)
    Args:
        boxes_ot: output detection boxes (in xyxy)
        boxes_gt: ground truth detection boxes (in xyxy)
        thres_iou: iou threshold (default: 0.5)
        verbose: a flag indicating whether or not print results (default: False)
    Returns:
        eval_det: detection evaluation result in a dictionary
            - coverage (nTP/ngt)
            - f_score
            - idx_gt: indices for TP in ground truth
            - idx_ot: indices for TP in output
            - FP: indices for FP in output
    """

    n_gt, n_ot = boxes_gt.shape[0], boxes_ot.shape[0]
    
    iou_scores = box_iou(boxes_gt, boxes_ot).numpy()
    max_iou    = np.max(iou_scores, axis=1)
    '''max_id_ot  = np.argmax(iou_scores, axis=1)'''
    
    TP   = np.where(max_iou > thres_iou)[0]
    nTP = len(TP)
    
    # For later usage of calculating confusion matrix & accuracy
    matched_id_ot = np.argmax(iou_scores, axis=1)[TP] # max_id_ot[TP]
    matched_id_gt = TP

    FP  = list(set(range(n_ot)) - set(matched_id_ot)) # detected - TP
    nFP = len(FP)

    FN  = list(set(range(n_gt)) - set(TP)) # gt - TP
    nFN = len(FN)
    
    eval_det = {
#         'coverage': nTP/n_gt,
        'recall': nTP/ (nTP+nFN),
        'precision': nTP/ (nTP+nFP),
        'f_score': 2*nTP/(2*nTP+nFP+nFN),
        'idx_gt': TP,
        'idx_ot':matched_id_ot,
        'FP': FP,
        'nFP': n_gt-nTP
    }
    
    if verbose:
        print("iou threshold: {:.2f} # TP: {}, # FP: {}, # FN: {}, \
        recall (coverage): {:.3f}, precision: {:.3f}, f-score: {:.3f}".format(thres_iou, nTP, nFP, nFN, 
                                          eval_det['recall'], eval_det['precision'], eval_det['f_score']
                                         ))              

    return eval_det 

def evaluate_detection(boxes_ot, boxes_gt, thres_iou=0.5, verbose=False, save_all=False):
    """A function to evaluate the goodness of object detection (for a list of boxes)
    Args:
        boxes_ot: a list of output detection boxes (in xyxy)
        boxes_gt: a list of ground truth detection boxes (in xyxy)
        thres_iou: iou threshold (default: 0.5)
        verbose: a flag indicating whether or not print results (default: False)
    Returns:
        perf: a matrix indicating performance: tp, fp, fn
        pr: a matrix indicating precision and recall
    """
    perf = np.zeros((3)) # tp, fp, fn
    pr = np.zeros((2)) # precision, recall
    evals = []
    for box_ot, box_gt in zip(boxes_ot, boxes_gt):
        _eval = _evaluate_detection(box_ot, box_gt, thres_iou=thres_iou, verbose=False)
        perf += np.array([len(_eval['idx_gt']), len(_eval['FP']), _eval['nFP']])
        evals.append(_eval)
    recall    = perf[0]/(perf[0]+perf[2])
    precision = perf[0]/(perf[0]+perf[1])
    if verbose:
        print("iou threshold: {:.2f}, recall (coverage): {:.3f}, precision: {:.3f}".format(
            thres_iou, recall, precision))
    pr = np.array([precision, recall])
    return perf, pr, evals
    

def plot_confusion_matrix(cm, classes,
                          classes_p=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          pdf_obj=None,
                          savename=None,
                          figsize=(5, 5),
                          ax=None,
                          verbose=False
                         ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes_p = classes if classes_p is None else classes_p
    cm = cm.astype(np.int32)
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1E-6)
        toshow = "Normalized confusion matrix"
    else:
        toshow = 'Confusion matrix, without normalization'
    if verbose:
        print(toshow)
        print(cm)

    if ax is None:
        f, ax = plt.subplots(1,1, figsize=figsize)
#     f = plt.figure(figsize=figsize)
    temp = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    plt.colorbar(temp, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes, rotation=45, horizontalalignment='right')
    ax.set_yticks(tick_marks, classes_p)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    if pdf_obj:
        pdf_obj.savefig(f, bbox_inches='tight')

    elif savename:
        plt.savefig(savename, bbox_inches='tight')


def evaluate_dset(results, gts, classes_confu, class_ids_confu, key_ot_lb, key_gt_lb, thres_iou=0.5, infer_dir="./", suffix_save="", 
verbose=False, ignore=[], ignore_idx=[]):
    """
    Evaluate object detection and classification for the whole dataset (detection coverage, confusion matrices)
    Plot and save to PDF
    Args:
        results: a dictionary of inference output (key: filename)
        gts: a dictionary of inference output (key: filename)
    """
    confu_pdf = PdfPages(os.path.join(infer_dir, 'confusion{}.pdf'.format(suffix_save)))

    boxes_gt = [x['box_xyxy'] for x in gts.values()]
    boxes_ot = [x['box_xyxy'] for x in results.values()]
        
    perf, pr, evals_det = evaluate_detection(boxes_ot, boxes_gt, thres_iou=thres_iou, verbose=verbose)
    dfs_cm = {}
    seen = 0
    for f_im, ot in results.items():
        fig, axs = plt.subplots(1,2, figsize=(12, 6))
        gt = gts[f_im]
        n_gt, n_ot = gt['box_xyxy'].shape[0], ot['box_xyxy'].shape[0]
        eval_det = evals_det[seen] # keys: 'recall', 'precision', 'f_score', 'idx_gt', 'idx_ot', 'FP', 'nFP'
        idx_ot, idx_gt, FP, recall = eval_det['idx_ot'], eval_det['idx_gt'], eval_det['FP'], eval_det['recall']
        FN  = np.array(list(set(list(range(gt['box_xyxy'].shape[0]))) - set(idx_gt)))
        assert len(idx_gt) / (len(FN) + len(idx_gt)) == eval_det['recall']
        assert set(idx_ot)|set(FP) == set(list(range(ot['box_xyxy'].shape[0])))
        if eval_det['recall'] == 0:
            LOGGER.info("\t 0 coverage for {}".format(f_im))
            results[f_im]['eval'] = {'coverage': 0, 'accuracy': np.nan, 'confusion': np.nan}
        
        else:
            axs[1].text(0.6, 0.5, "coverage: {:.3f} ({}/{})".format(recall, len(idx_gt), n_gt), 
                        transform=fig.transFigure, size=12)
            LOGGER.info("\t{} coverage: {:.3f}".format(f_im.split("/")[-1], eval_det['recall']))
            
            labels_ot = ot[key_ot_lb]
            labels_gt = gt[key_gt_lb]
            labels_det_ot = np.array(labels_ot)[idx_ot]
            labels_det_gt = np.array(labels_gt)[idx_gt]
            eval_cls = evaluate_classification(labels_det_ot, labels_det_gt, class_ids=class_ids_confu)
            
            # attach missing and falsely detected
            cm_ = np.append(eval_cls['cm'], np.zeros((len(class_ids_confu),1)), axis=1) # miss detected
            cm_ = np.append(cm_, np.zeros((1, cm_.shape[1])), axis=0) # falsely detected
        
            if len(FN):  # miss detected
                idx, counts = np.unique(np.array(labels_gt)[FN], return_counts=True)
                for ign in ignore_idx:
                    idx[idx==ign] = class_ids_confu.index(ign)
                cm_[idx.astype(np.int32), len(class_ids_confu)] = counts
                
            if len(FP):# attach falsely detected
                idx, counts = np.unique(np.array(labels_ot)[FP], return_counts=True)
                cm_[len(class_ids_confu), idx.astype(np.int32)] = counts
                
            _nTP = eval_cls['cm'].sum() 
            _nFN = cm_[:len(classes_confu),:].sum()-_nTP
            try: 
                assert _nTP / (_nTP+ _nFN) == eval_det['recall'] and cm_[:len(classes_confu),:].sum() == n_gt 
            except:
                import pdb; pdb.set_trace()    
            df_cm = pd.DataFrame(columns=classes_confu+['missing'], data=cm_, index=classes_confu+['falsely detected'])
            dfs_cm[f_im] = df_cm
            accu, correct, total, cm = eval_cls['accuracy'], eval_cls['correct'], eval_cls['total'], eval_cls['cm']    
            axs[1].text(0.6, 0.4, "accuracy: {:.3f} ({}/{})".format(accu, correct, total), transform=fig.transFigure, size=12)
            axs[1].axis('off')
            plot_confusion_matrix(df_cm.to_numpy(), classes=df_cm.columns, classes_p=df_cm.index,
                                  title="{}".format(f_im.split("/")[-1]), ax=axs[0])
            
            confu_pdf.savefig(fig, bbox_inches='tight')

            results[f_im]['eval'] = {'coverage': eval_det['recall'], 'accuracy': accu, 'confusion': cm}
        seen += 1    
    plt.close("all")    
    ## overall detection performance
    perf_det = {
        "recall": pr[1],
        "precision": pr[0],
        "f_score": 2/sum(1/pr)
    }
    LOGGER.info("Overall coverage at iou {:.2f}: {:.3f}, precision: {:.3f}, F score: {:.3f}"\
                .format(thres_iou, perf_det["recall"], perf_det["precision"], perf_det["f_score"]))

    
    ## overall classification performance
    dfs_all     = sum(dfs_cm.values())
    cm_all_full = dfs_all.to_numpy()
    cm_all      = cm_all_full[:len(classes_confu),:len(classes_confu)] # no miss detection / falsely detection information
    # get rid of ignored classes
    classes_confu_ = copy(classes_confu)
    cm_all_ = copy(cm_all)
    #if len(ignore):
    for x in ignore:
        idx = classes_confu_.index(x)
        cm_all_ = np.delete(cm_all_, idx, axis=0)
        cm_all_ = np.delete(cm_all_, idx, axis=1)
        classes_confu_.pop(idx)
    perf_cls = {
        "accuracy": np.diag(cm_all_).sum()/cm_all_.sum(),
        "per_cls_accuracy": np.diag(cm_all_)/(cm_all_.sum(axis=1) + 1E-6)
    }
    
    LOGGER.info("\nOverall classification accuracy {:.3f}".format(perf_cls['accuracy']))
    temp = "\n".join([" - ".join([x, str(round(y, 3))]) for (x,y) in zip(classes_confu, perf_cls['per_cls_accuracy'])])
    LOGGER.info("Per-class accuracy")
    LOGGER.info(temp)
    
    # overall confusion matrix
    nTP = cm_all.sum()
    nFN = cm_all_full[:len(classes_confu),:].sum()-nTP
    nFP = cm_all_full[len(classes_confu),:].sum()
    assert nTP / (nTP+nFN) == perf_det["recall"]
    plot_confusion_matrix(cm_all_full, classes=dfs_all.columns, classes_p=dfs_all.index, figsize=(8,8),
                          title= "Overall coverage: {:.3f} ({}/{})".format(perf_det["recall"], nTP, nTP+nFN),
                          pdf_obj=confu_pdf)

            
    plot_confusion_matrix(cm_all_, classes=classes_confu_, figsize=(8,8),
                          normalize=True,
                          title= "Overall accuracy: {:.3f} ({}/{})".format(perf_cls['accuracy'], 
                                                                           np.diag(cm_all_).sum(), 
                                                                           cm_all_.sum()),
                          pdf_obj=confu_pdf)
                          
    confu_pdf.close()
    eval_out = {**perf_det, **perf_cls}
    dfs_cm["all"] = dfs_all
    pkl.dump(dfs_cm, open(os.path.join(infer_dir, "confusion_matrices_{}.pkl".format(suffix_save)), "wb"))
    pkl.dump(eval_out, open(os.path.join(infer_dir, "summary{}.pkl".format(suffix_save)), "wb"))
    eval_out_ = pd.DataFrame.from_dict(eval_out, orient='index')
    eval_out_.to_csv(os.path.join(infer_dir, 'summary{}.csv'.format(suffix_save)))
    plt.close("all")
    return eval_out, dfs_cm
    
    
old2new = {
            'tumor nuclei': "tumor",
            'stroma nuclei': "stromal cell",
            'lymphocyte nuclei': "immune cell",
            'red blood cell': "immune cell",
            'macrophage nuclei': "immune cell",
            'ductal epithelium': "other",
            'dead nuclei':"apoptotic body"
        }
        
        
def _update_cm(cm_old, new2old_idx, classes_new):
    cm_new_ = copy(cm_old)
    
    for nid, oids in new2old_idx.items():
        # row
        cm_new_[nid,:] = cm_old[oids, :].sum(axis=0)
        
    cm_new_ = cm_new_[:len(classes_new), :]
    cm_new  = copy(cm_new_)# np.zeros(cm_all_.shape)
    for nid, oids in new2old_idx.items():
        # column
        cm_new[:, nid] = cm_new_[:, oids].sum(axis=1)
    cm_new = cm_new[:, :len(classes_new)]
    return cm_new
        
def evaluate_dset_convert(evals, cms, classes, class_ids, old2new, infer_dir="./", suffix_save="", title="root level"):
    """Convert evaluation output given convert dictionary, i.e. convert classification indices if necessary
    Args:
        evals
        cms
        classes (list): class names correspond to original outputs (in evals and cms)
        class_ids (list): class indices correspond to original outputs (in evals and cms)
        old2new (dict): convert from classes to another set of classes
    """
    confu_pdf = PdfPages(os.path.join(infer_dir, 'confusion{}.pdf'.format(suffix_save)))
    
    # get the list of new classes, class indices
    classes_new, class_ids_new, ct = [], [], 0
    for cls in classes:
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
        new2old_idx[classes_new.index(clsn)].append(classes.index(clso))
        
    # TODO: update confusion matrices for each individual image???
    
    # update overall confusion matrix
    cm_new = _update_cm(cms['all'], new2old_idx, classes_new)
    
    plot_confusion_matrix(cm_new, classes=classes_new, figsize=(8,8),
                        normalize=False,
                        title=title,
                        pdf_obj=confu_pdf
                        )
    plot_confusion_matrix(cm_new, classes=classes_new, figsize=(8,8),
                        normalize=True,
                        title= "Overall accuracy: {:.3f} ({}/{})".format(np.diag(cm_new).sum()/cm_new.sum(), 
                                                                   np.diag(cm_new).sum(), cm_new.sum()),
                        pdf_obj=confu_pdf
                        )
    
    


    