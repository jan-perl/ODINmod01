# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
#utilities voor raster zaken
# -

import glob

import pandas as pd
import numpy as np
import os as os
import re as re
import seaborn as sns
import matplotlib.pyplot as plt

# +
#os.system("pip install geopandas contextily rasterio rasterstats")
# -

import geopandas
import contextily as cx
plt.rcParams['figure.figsize'] = [10, 6]

import rasterio
import rasterio.plot
from rasterio.transform import Affine  
from rasterio import features

import rasterstats

calcgdir="../intermediate/calcgrids"


# +
#https://stackoverflow.com/questions/72452203/how-to-make-moving-average-using-geopandas-nearest-neighbors
#import libpysal
#from https://gis.stackexchange.com/questions/421556/performance-problem-with-getting-average-pixel-values-within-buffered-circles
#import qgis
# -

def createNLgrid(res,fn,countin,deel):
#resulution
#coords    "[{300000, 0}, {625000, 280000})"; 
    
    if deel == 'Ut':
        transformx = Affine.translation(113000 - res / 2, 480000 - res / 2) * Affine.scale(res, -res)    
        hv=(480000 - 430000)/res
        wv=(180000- 113000)/res  
    else:
        transformx = Affine.translation(0 - res / 2, 625000  - res / 2) * Affine.scale(res, -res)    
        hv=(625000 - 300000)/res
        wv=(280000- 0)/res

    new_dataset = rasterio.open(
         fn,          'w',         driver='GTiff',
        height= hv,        width=wv,       count=countin,
         dtype=np.float32(1).dtype,
         crs='EPSG:28992',
         transform=transformx,
         nodata = np.nan
#          nodata = 0
     )
    return new_dataset
#example lasttifname=calcgdir+'/land.tif'
#example gridNL100 = createNLgrid(100,lasttifname,1,'')


#all_touched = True zorgt dat bij 100 meter grid ca 1.300 of  1.484 van waardes te
# hoog terug komt  in check met if fillrela
# en dat 49 van de 971 waarden in Utrecht bij uitlezen waarde van 
#all_touched = False zorgt dat bij 100 meter grid ca 1.0025467901140264, 1.0006930004604095 van waardes te
# hoog terug komt  in check met if fillrela
# en dat 0 van de 971 waarden in Utrecht bij uitlezen waarde anders worden
def makerasterdef (df,col,grid):
    shapesusd = ((geom,value) for geom, value in zip( df.geometry, df[col]))
    imageout = rasterio.features.rasterize(
            shapes=shapesusd,
            merge_alg=rasterio.features.MergeAlg.replace,
#            all_touched=True,
            default_value=np.nan,
            out_shape=grid.shape,
            transform=grid.transform)
    return imageout
#example imageMXIo= makerasterdef(itotUtr,'MXI_22',gridNL100b)


def makerasterpts (df,col,grid):
    shapesusd = ((geom,value) for geom, value in zip( df["center"], df[col]))
    imageout = rasterio.features.rasterize(
            shapes=shapesusd,            
            merge_alg=rasterio.features.MergeAlg.add,
            default_value=0,
            out_shape=grid.shape,
            transform=grid.transform)
    return imageout
#example imageMXIp= makerasterpts(itotUtr,'MXI_22',gridNL100b)


# +
def setaxhtn(ax):
    ax.set_xlim(left=137000, right=143000)
    ax.set_ylim(bottom=444000, top=452000)
    
def setaxutr(ax):
    ax.set_xlim(left=113000, right=180000)
    ax.set_ylim(bottom=480000, top=430000)


# -

def misstats (c1,c2,itotUtri):
    itotUtri["griddif"]= abs(itotUtri[c1] - itotUtri[c2])
    mis=itotUtri[np.isnan(itotUtri[c1])]
    print (( len(itotUtri["griddif"]),len(mis),
    len(itotUtri[itotUtri["griddif"]>1e-6]) ))
    return (itotUtri[itotUtri["griddif"]>1e-6])
#example misstats("mxifromgrida","MXI_22")            


# +
#example misstats("mxifromgridp","MXI_22") 
# -

#now some kernel and transformations
#note: for gaussian: ony can do 2 1d convolutions
def kernelfietspara():
    maxfietsdist =5000
    maxkernrng = 1* maxfietsdist
    gridstep =100
    FWHMcnv  = 2.0 * np.sqrt (2.0 * np.log (2.0));
    #print(FWHMcnv)
    gaussdenom   = (maxfietsdist/FWHMcnv) ;
    #print(gaussdenom)
    x = np.linspace(-maxkernrng,+maxkernrng, int(2*maxkernrng/gridstep+1))
    y = np.linspace(-maxkernrng,+maxkernrng, int(2*maxkernrng/gridstep+1))
    X, Y = np.meshgrid(x, y)
    d1 = np.sqrt(X*X+Y*Y)
    Z1 = np.exp(-d1*d1 / (2*gaussdenom* gaussdenom))
    Z1= Z1/ np.sum(Z1)
    K1D= np.exp(-x*x / (2*gaussdenom* gaussdenom))
    K1D= K1D/ np.sum(K1D)
    Z2D= np.outer( K1D, (K1D))
#    print(np.sum(Z1))
#    print('sum')
    Z = Z1
    return {'X':X,'Y':Y,'Z':Z,'K1D':K1D,'Z2D':Z2D }
#example fietskern1= kernelfietspara()
#example dcalc=(fietskern1['Z2D']-fietskern1['Z'])
#example print(np.max(np.abs(dcalc)) )


# +
from numba import cuda
import numpy as np

@cuda.jit
def convolve2d(result, mask, image):
    # expects a 2D grid and 2D blocks,
    # a mask with odd numbers of rows and columns, (-1-) 
    # a grayscale image
    
    # (-2-) 2D coordinates of the current thread:
    i, j = cuda.grid(2) 
    
    # (-3-) if the thread coordinates are outside of the image, we ignore the thread:
    image_rows, image_cols = image.shape
    if (i >= image_rows) or (j >= image_cols): 
        return
    
    # To compute the result at coordinates (i, j), we need to use delta_rows rows of the image 
    # before and after the i_th row, 
    # as well as delta_cols columns of the image before and after the j_th column:
    delta_rows = mask.shape[0] // 2 
    delta_cols = mask.shape[1] // 2
    
    # The result at coordinates (i, j) is equal to 
    # sum_{k, l} mask[k, l] * image[i - k + delta_rows, j - l + delta_cols]
    # with k and l going through the whole mask array:
    s = 0
    for k in range(mask.shape[0]):
        for l in range(mask.shape[1]):
            i_k = i - k + delta_rows
            j_l = j - l + delta_cols
            # (-4-) Check if (i_k, j_k) coordinates are inside the image: 
            if (i_k >= 0) and (i_k < image_rows) and (j_l >= 0) and (j_l < image_cols):  
                s += mask[k, l] * image[i_k, j_l]
    result[i, j] = s


# -

def convfietsimg(img,kern1):
    return np.apply_along_axis(lambda sl: np.convolve(sl,kern1['K1D'],mode='same'),1, 
           np.apply_along_axis(lambda sl: np.convolve(sl,kern1['K1D'],mode='same'),0, img) )


def convfiets2d(image,kern1,bdim=32):
    # We preallocate the result array:
    result = np.empty_like(image)
    # We use blocks of 32x32 pixels:
    blockdim = (bdim, bdim)
#    print('Blocks dimensions:', blockdim)

    # We compute grid dimensions big enough to cover the whole image:
    griddim = (image.shape[0] // blockdim[0] + 1, image.shape[1] // blockdim[1] + 1)
#    print('Grid dimensions:', griddim)
    # We apply our convolution to our image:
    convolve2d[griddim, blockdim](result, kern1, image)
    return result
#example image4g= convfiets2d(image1 ,fietskern1['Z'] )     


# +
#routines that plot arrays but no not keep x,y coordinate transformations
def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def plotrb(dset,c1,c2):
    # Normalize band DN
    red_norm = np.sqrt(normalize(dset.read(c1)))/2+.5
    blue_norm = np.sqrt(normalize(dset.read(c2)))/2+.5
    # Stack bands
    nrg = np.dstack((red_norm, (red_norm+blue_norm)/2, blue_norm))
    # View the color composite
    plt.imshow(nrg,origin='lower') 
#heel NL niet geschaald
#plotrb(dataset3,4,5)    


# -

class GridArgError(Exception):
    pass


def makegridcorr (df,grid):
    shapesusd = ((geom,value) for geom, value in zip( df.geometry, df.index+1))
    imageout = rasterio.features.rasterize(
            shapes=shapesusd,
            merge_alg=rasterio.features.MergeAlg.replace,
#            all_touched=True,
            default_value=0,
            out_shape=grid.shape,
            transform=grid.transform)
    if(np.max(imageout) != len(df)):
        print (('info: high proj, len df ',np.max(imageout) , len(df) ))
        raise(GridArgError("makegridcorr: Last element not projected"))
    return imageout
#example dfrefs= makegridcorr (itotUtr,gridNL100c)


#zet er nieuwe punten bij
def addmakegridcorr (df,grid):
    shapesusd = ((geom,value) for geom, value in zip( df.geometry, df.index+1))
    imageout = rasterio.features.rasterize(
            shapes=shapesusd,
            merge_alg=rasterio.features.MergeAlg.add,
#            all_touched=True,
            default_value=0,
            out_shape=grid.shape,
            transform=grid.transform)
    return imageout
#example dfrefs= makegridcorr (itotUtr,gridNL100c)


# +
import numba
#from numba.utils import IS_PY3
from numba.decorators import jit

@jit
def _docountpixarea(img,hival,retval):
    for hh in range(img.shape[0]):
        for ww in range(img.shape[1]):
            if (img[hh,ww] >0) and (img[hh,ww] <=hival):
                retval[int(img[hh,ww])-1] +=1

def countpixarea(img):  
    ncats=int(np.max(img))
    retval=np.zeros(ncats,dtype=int)
    _docountpixarea(img,ncats,retval)
    return retval

def findmiss(indf,img):
    oppix= countpixarea(img)
    if len(indf)!=len(oppix):
        raise(GridArgError("findmiss: Incompatible length passed"))        
    oppix[oppix==0]
    print(len(indf),len(oppix))
    indf['area_pix']=oppix*100*100
    indf['area_diff']=indf['area_pix']-indf['area_geo']
    rv=indf[oppix==0]
    return rv
    
#example missptdf= findmiss(itotUtr,dfrefs)
#example print(missptdf.index)
#example missptdf[['area_pix','area_diff','area_geo']]    


# +
import numba
#from numba.utils import IS_PY3
from numba.decorators import jit

@jit
def _dosumpixarea(img,hival,valarr,retval):
    for hh in range(img.shape[0]):
        for ww in range(img.shape[1]):
            if (img[hh,ww] >0) and (img[hh,ww] <=hival):
                retval[int(img[hh,ww])-1] += valarr[hh,ww]

def sumpixarea(img,valarr):    
    ncats=int(np.max(img))
    retval=np.zeros(ncats,dtype=np.float32)
    _dosumpixarea(img,ncats,valarr,retval)
    return retval


# -

import seaborn
#example seaborn.scatterplot(data=itotUtr,x='area_geo',y='area_pix')

# +

@jit
def fillmiss(imgori,imgmiss):
    for hh in range(imgmiss.shape[0]):
        for ww in range(imgmiss.shape[1]):
            if imgmiss[hh,ww] >0:
                imgori[hh,ww] = imgmiss[hh,ww]

def testmissfill(dfrefsi,missptdfi,gridNL100ci,itotUtri):
    missptimg= addmakegridcorr (missptdfi,gridNL100ci)
    print(np.sum(missptimg))
    dfrefs2i= dfrefsi.copy()               
    fillmiss(dfrefs2i,missptimg)
    print(np.sum(dfrefs2i-dfrefsi))
    missptdf2= findmiss(itotUtri,dfrefs2i)
    print(missptdf2.index)
    missptdf2[['area_pix','area_diff','area_geo']] 
    return dfrefs2i
    
#example dfrefs2=testmissfill(drefsi,missptdfi,gridNL100ci,itotUtri)    


# +
#from numba.utils import IS_PY3
from numba.decorators import jit
@jit
def fillvalues(image,idximg,indat):
    for hh in range(idximg.shape[0]):
        for ww in range(idximg.shape[1]):
            if idximg[hh,ww] >0:
                image[hh,ww] = indat[int(idximg[hh,ww])-1] 

def mkimgpixavgs(grid,idximg,addfietsf,kern1,indf,donorm=True):
    ilst=[idximg]
    ncats=int(np.max(idximg))
    if len(indf)!=ncats:
        raise(GridArgError("mkimgpixavgs: Incompatible length passed"))        
    
    oppix= countpixarea(idximg)
#    print("normarr ",normarr.dtype,len(normarr))
    image=np.zeros(idximg.shape,dtype=np.float32)
    fillvalues(image,idximg,np.float32(oppix))
    putidx=2
    ilst.append(image.copy())
    normarr=oppix.copy()
    normarr[np.isnan(normarr) | (normarr==0)] = 1
    for col in indf.columns:
        putidx+=1
        image=np.zeros(idximg.shape,dtype=np.float32)
        if donorm:
            normval = np.float32(indf[col]) /normarr
        else:
            normval = np.float32(indf[col]) 
#        print(col,normval.dtype,len(normval))
        fillvalues(image,idximg,normval)
        ilst.append(image.copy())
        if addfietsf:
            putidx+=1
            imagef= convfietsimg(image ,kern1) 
            ilst.append(imagef.copy())
            print(np.sum(indf[col]), np.sum(image),np.sum(imagef) )
#en voeg, ALLEEN VOOR LAATSTE PAAR, ratios en ratios smooth toe            
    if addfietsf:
        for col in range(2):
            putidx+=1
            imager= ilst[putidx-1-2*len(indf.columns)]/ \
                    np.where(ilst[putidx+1-2*len(indf.columns)]>100,
                             ilst[putidx+1-2*len(indf.columns)],np.nan)
            ilst.append(imager.copy())
    grid.write(np.array(ilst),list(i+1 for i in range(putidx))  )  
    return ilst
#example imagelst=mkimgpixavgs(gridNL100c,dfrefs,True,itotUtr[['O_MXI22T','O_MXI22N']])    
# +
#read back points, using rasterio directly

import numba
#from numba.utils import IS_PY3
from numba.decorators import jit

#@jit
def addobjvals(img,coords,values):
    for obj in range(coords.shape[0]): 
        img[coords[obj,1],coords[obj,0]] += values[obj]
        
#@jit
def getobjvals(img,coords,values):
    for obj in range(coords.shape[0]): 
        values[obj]= img[coords[obj,1],coords[obj,0]]         

def gridoncenters (grid,r1):
    corrgrid = np.zeros([grid.width, grid.height],dtype=np.int32)    
    addobjvals(corrgrid,r1,range(len(r1)))
    valtst = np.zeros (len(r1),dtype=np.int32)
    getobjvals(corrgrid,r1,valtst)
    print(valtst[abs(valtst - range(len(r1)) ) > 1e-6])
    return corrgrid

def centergridcoords (grid,ctrser):
    r1= np.array(rasterio.transform.rowcol(grid.transform,xs=ctrser.x,ys=ctrser.y)).T
    return r1
#usage of centergridcoords:     
#ctrxform = centergridcoords (smftg1,buurtendata[ "center"]) 
#gridoncenters (smftg1,ctrxform)


# +
#now some kernel and transformations
#note: for gaussian: ony can do 2 1d convolutions
def roundfilt(gridstep,dist):
    maxkernrng= int(dist/gridstep+2)*gridstep
    x = np.linspace(-maxkernrng,+maxkernrng, int(2*maxkernrng/gridstep+1))
    y = np.linspace(-maxkernrng,+maxkernrng, int(2*maxkernrng/gridstep+1))
    X, Y = np.meshgrid(x, y)
    Z = np.int8(np.sqrt(X*X+Y*Y) <dist)
    return Z

#print(roundfilt(100,660) )


# -

#sla images uit bestand op
def getcachedgrids(src):
    clst={}
    for i in src.indexes:
        clst[i] = src.read(i) 
    return clst


# +
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)  

#from https://stackoverflow.com/questions/53699012/performant-cartesian-product-cross-join-with-pandas
def cartesian_product_multi(*dfs):
#todo set columns
    idx = cartesian_product(*[np.ogrid[:len(df)] for df in dfs])
    rv= pd.DataFrame(
        np.column_stack([df.values[idx[:,i]] for i,df in enumerate(dfs)]))
#    collst = ()
#    rv.columns= list(mygeoschpc4.columns) + list(dfgrps.columns)
    return rv



# -
def getcachedgrids(src):
    clst={}
    for i in src.indexes:
        clst[i] = src.read(i) 
    return clst



