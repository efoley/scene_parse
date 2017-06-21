import numpy as np

import os
import csv

from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
import itertools
import colorcet as cc
from scipy import misc, ndimage

import hashlib
import tqdm
import urllib3
#import requests
import zipfile


def maybe_mkdir(d):
    try: os.mkdir(d)
    except(FileExistsError): pass

def _md5(path):
    hash_md5 = hashlib.md5()
    
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()

def _download_with_progress(url, dest_path, check_md5=None):
    http = urllib3.PoolManager()
    r = http.request('HEAD', url)

    content_length = int(r.headers['Content-Length'])

    r = http.request('GET', url, preload_content=False)
    
    #filename = url.split('/')[-1]
    #dest_path = os.path.join(dest_dir, filename)
        
    if os.path.exists(dest_path):
        existing_md5 = _md5(dest_path)
        if existing_md5==check_md5:
            print("File already exists at {}".format(dest_path))
            return
        else:
            raise RuntimeError("File already exists at {} but md5 doesn't match expected".format(dest_path))
    
    hash_md5 = hashlib.md5()
    
    with tqdm.tqdm_notebook(total=content_length, unit='B', unit_scale=True) as pbar:
        with open(dest_path, "wb") as f:
            for chunk in r.stream(1<<20):
                f.write(chunk)
                if not check_md5 is None:
                    hash_md5.update(chunk)
                pbar.update(len(chunk))
    
    
    if not check_md5 is None:
        download_md5 = hash_md5.hexdigest()
        if download_md5!=check_md5:
            raise RuntimeError("md5 sum does not match expected")
        
def _unzip(zip_path, dest_dir):
    z = zipfile.ZipFile(zip_path)
    z.extractall(dest_dir)
    
def download_and_extract(url, dest_dir, check_md5=None):
    archive_filename = url.split('/')[-1]
    archive_path = os.path.join(dest_dir, archive_filename)
    
    marker_name = '.download_and_extract_{}_{}'.format(archive_filename, check_md5)
    marker_path = os.path.join(dest_dir, marker_name)
    
    if os.path.exists(marker_path):
        print("File from {} has already been downloaded and extracted.".format(url))
        return
    
    _download_with_progress(url, archive_path, check_md5=check_md5)
    
    print('Unzipping {} into {}.'.format(archive_path, dest_dir))
    _unzip(archive_path, dest_dir)
    
    os.remove(archive_path)
    
    with open(marker_path, 'wb') as f:
        f.write(b"1")
    
        
##### viz stuff
def _to_bokeh_image(img_in):
    if len(img_in.shape)>2:
        # multi-channel to use with plot.image_rgba()
        img = np.zeros(img_in.shape[:2], dtype=np.uint32)    
        img_v = img.view(dtype=np.uint8).reshape(list(img.shape) + [4])
        img_v[:,:,:3] = img_in
        img_v[:,:,3] = 255
        return img[::-1,:]
    else:
        # single channel image to be used with plot.image()
        return img_in[::-1,:]

def _remap_annotations(ann):
    ann_objects = np.unique(ann)
    ann_remap = np.r_[:len(np.unique(ann))]*256//len(np.unique(ann))
    ann_remap_dict = {o:r for o,r in zip(ann_objects, ann_remap)}
    
    ann_remap = np.copy(ann)
    ann_remap_v = np.reshape(ann_remap, [-1])
    for i, v in enumerate(ann_remap_v):
        ann_remap_v[i] = ann_remap_dict[v]
        
    return ann_remap, ann_remap_dict

def _find_a_blob(arr, v):
    a = np.zeros_like(arr, dtype=np.int)    
    a[arr==v] = True
    ssize = 32
    while ssize>0:
        da = ndimage.binary_erosion(a, structure=np.ones((ssize, ssize), dtype=np.int))
        nz = np.nonzero(da)
        if len(nz[0])!=0:
            return tuple([nze[0] for nze in nz])
        ssize//=2
    return None

def _determine_good_label_locations(ann):
    uniques = np.unique(ann)
    locs = []
    for u in uniques:
        l = _find_a_blob(ann, u) # index [x,y] of blob location
        assert l is not None
        locs.append((u, l))
    return locs

def _plot_annotations(ann, object_names, x_range, y_range):
    r, rd = _remap_annotations(ann)
    p = figure(x_range=x_range, y_range=y_range)
    p.image([_to_bokeh_image(r)], x=0, y=0, dw=r.shape[1], dh=r.shape[0], palette=cc.rainbow)
    
    label_locs = _determine_good_label_locations(ann)
    for obj,loc in label_locs:
        if obj==0:
            continue
        y = r.shape[0]-loc[0]-1
        x = loc[1]
        text = object_names[obj]
        # TODO EDF should make single text call for all annotations
        p.text([x], [y], text=[text])
    
    return p

def show_image_and_annotations(img, ann, object_names):    
    p_img = figure(x_range=(0, img.shape[1]), y_range=(0, img.shape[0]))
    p_img.image_rgba([_to_bokeh_image(img)], x=0, y=0, dw=img.shape[1], dh=img.shape[0])
    # TODO EDF put in some sort of legend for annotations
    p_ann = _plot_annotations(ann, object_names, x_range=p_img.x_range, y_range=p_img.y_range)
    show(column([p_img, p_ann]))