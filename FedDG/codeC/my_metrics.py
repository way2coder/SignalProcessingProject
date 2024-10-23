import torch
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,generate_binary_structure
import cv2
threshold1 = 0.5
threshold2 = 0
def jc(result, reference):

    result = np.atleast_1d((result > threshold1).astype(bool))
    reference = np.atleast_1d((reference > threshold2).astype(bool))   
    # result = np.atleast_1d(result.astype(bool))
    # reference = np.atleast_1d(reference.astype(bool))
    
    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)
    
    jc = float(intersection) / float(union)
    
    return jc

'''
def dc(target,predictive,ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

'''
def dc(result, reference):
    #Dice coefficient
    


    # binary_array1 = (array1 > threshold).astype(np.bool)
    # print(f"threshold1 = {threshold1},threshold2 = {threshold2}\n")
    result = np.atleast_1d((result > threshold1).astype(bool))
    reference = np.atleast_1d((reference > threshold2).astype(bool)) 

    intersection = np.count_nonzero(result & reference)
    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    # print(f"intersection = {intersection}, size_i1 = {size_i1}, size_i2 = {size_i2}")
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    maxitem = np.max(reference)
    # print(F"maxitem = {maxitem}")
    result = np.atleast_1d((result > threshold1).astype(bool))
    reference = np.atleast_1d((reference > threshold2).astype(bool)) 
    # result = np.atleast_1d(result.astype(bool))
    # reference = np.atleast_1d(reference.astype(bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        # print(F"threshold2 = {threshold2}")
        
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95

def assd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average symmetric surface distance.
    """
    assd = np.mean( (asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)) )
    return assd

def asd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average surface distance metric.
    
    """
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd

def recall(result, reference):
    """
    Recall.
    """

    result = np.atleast_1d((result > threshold1).astype(bool))
    reference = np.atleast_1d((reference > threshold2).astype(bool)) 
    # result = np.atleast_1d(result.astype(bool))
    # reference = np.atleast_1d(reference.astype(bool))
        
    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0
    
    return recall

def specificity(result, reference):
    """
    Specificity.
    """

    
    # result = np.atleast_1d(result.astype(bool))
    # reference = np.atleast_1d(reference.astype(bool))
    result = np.atleast_1d((result > threshold1).astype(bool))
    reference = np.atleast_1d((reference > threshold2).astype(bool)) 
       
    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0
    
    return specificity

def precision(result, reference):
    """
    Precison.
    """


    # result = np.atleast_1d(result.astype(bool))
    # reference = np.atleast_1d(reference.astype(bool))
    result = np.atleast_1d((result > threshold1).astype(bool))
    reference = np.atleast_1d((reference > threshold2).astype(bool)) 
        
    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)
    
    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0
    
    return precision


def calculate_metrics(pred, target):
    '''
        a function designed to calculate a group of metrics
    '''
    
    if torch.is_tensor(pred):
        pred = pred.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    dice = dc(pred,target)
    arr_2d = (pred.squeeze() * 255).astype(np.uint8)
    temp1, _ = cv2.threshold(arr_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    arr_2d1 = (target.squeeze() * 255).astype(np.uint8)
    temp2, _ = cv2.threshold(arr_2d1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    global threshold1
    threshold1 = temp1/255
    global threshold2
    threshold2 = temp2/255
    jaccard = jc(pred,target)
    hd95_score = hd95(pred,target)
    assd_score = assd(pred,target)
    sp_score = specificity(pred,target)
    recall_score = recall(pred,target)
    pre_score = precision(pred,target)
    return dice,jaccard,hd95_score,assd_score,sp_score,recall_score,pre_score
