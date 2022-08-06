import logging
import urllib.request
import json 
import setproctitle
import subprocess
import boto3

import time
import datetime

from os.path import join,basename,isfile,isdir,splitext,exists,abspath,expanduser
from os import listdir,remove,makedirs,environ

import cv2
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import lagrange

from libxmp import XMPError,consts
from libxmp.utils import object_to_dict


NAME_EXTRA_FILES_DIR='extraFiles'
NAME_PLOT_FACE_DETECTION_DIR='2DBBplot'

FEATHER=20

X_ratio_level_prc=[.25,1,5,10,20]
N_ellipse_level=[2,3,5,10,15]


Coeff_Lagrange=Polynomial(lagrange(X_ratio_level_prc,N_ellipse_level)).coef
d_lagrange=len(Coeff_Lagrange)-1

Ratio_level_set_sharpness=[0.08,0.1,0.4]

def get_list_img(input_path):
    """
    Return a list of image path ready to be read given a root path. 

    Args:
        input_path ([str]): Root path, can be either a path to a complete folder or even a file. Both are handled here. 

    Returns:
        [list]: list of paths of image ready to be read. 
    """
    #les_img can either contain a single path (the input one) or a list of image path if a folder path has been provided
    les_img=[input_path] if isfile(input_path) else [join(input_path,f) for f in listdir(input_path) if isfile(join(input_path,f))] if isdir(input_path) else logging.critical(f'[Provided argument  : {input_path} is neither a file or a directory !]')
    logging.info(f'Input provided path {input_path} has been sucessfully opened. It contains {len(les_img)} images.')
    return les_img

def f(x):
    """
    This function computes the number of ellipse to write on the XMP metadata based on the occupied area of a face.
    The function is based on Lagrange interpolation. 
    Please note the Lagrangian polynom is entirely defined through the set of point : (X_ratio_level_prc , N_ellipse_level)

    Args:
        x (float): Input ratio, based on the size of the bounding box for a given face. 

    Returns:
        nb_ellipse (int): The number of ellipse to write on the XMP metadata. Output value are clip within {1..15}
    """
    #Round the input ratio to the closest int.
    x=np.round(x)
   
    #If the ratio is higher/lower than a fixed threshold, clip the output number of ellipse to write.
    if x>X_ratio_level_prc[-1]:
        return N_ellipse_level[-1]
    elif x<X_ratio_level_prc[0]:
        return 1

    #Compute the number of ellipse to write on XMP metadata.
    nb_ellipse=0
    for idx,c in enumerate(Coeff_Lagrange): 
        nb_ellipse+=c*np.power(x,d_lagrange-idx)
    nb_ellipse=np.round(nb_ellipse)

    return nb_ellipse
        
def init_S3():
    """
    Allow to init the right Bucket on S3 to further download weight file used in RetinaFace for face detection. 
    Returns:
        [s3bucket]: Return the models Bucket from S3.
    """
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("models.rd.meero.com")
    return bucket

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def set_environment(use_gpu=False):
    """
    Function directly taken from Victor Busa code on the Eagle project.
    Allow to set visible a GPU from CUDA.
    Args:
        use_gpu (bool, optional): Allow to specify if GPU mode must be activate or not. Defaults to False.
    """
    if use_gpu:
        available_gpus = get_available_gpu_indexes()
        environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        environ["CUDA_VISIBLE_DEVICES"] = select_gpu(available_gpus)
    

def get_available_gpu_indexes():
    """
    Returns the indexes of all the available GPUs sorted by the one having the most memory first. 
    Function directly taken from Victor Busa code on the Eagle project.

    Raises:
        Exception: If no GPU are available.

    Returns:
        [list]: A sorted list with all the GPUs which are available.
    """

    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used,memory.total',
            '--format=csv,nounits,noheader'
        ])
    result = result.decode('utf-8')
    available_gpus = []
    for i, x in enumerate(result.strip().split('\n')):
        used_memory, available_memory = [int(e) for e in x.split(',')]

        # GPU is considered available if the used memory is below 20 percent
        # of the available memory
        if used_memory <= 0.2 * available_memory:
            available_gpus.append((i, available_memory))

    if not available_gpus:
        raise Exception("All the GPUs are in use")
    indexes =[x[0] for x in sorted(available_gpus, key=lambda x: x[1])[::-1]]
    return indexes


def select_gpu(available_gpus, random=False):
    """
    Allow to select the most suitable GPU from the available ones.

    Args:
        available_gpus ([list]): The list of the available GPUs. 
        random (bool, optional): If the most suitable GPU (with the largest memory) doesnt matter. Defaults to False.

    Raises:
        Exception: If available_gpus list is empty, meaning no GPUs are available.

    Returns:
        [str]: The index string of the GPU which is going to be targeted.
    """
    if len(available_gpus) == 0:
        raise Exception('No GPUs are available right now')
    if random:
        return str(np.random.choice(available_gpus))
    return str(available_gpus[0]) 