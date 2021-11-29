# Miguel Ibanez, Roma October 2017

import os
import numpy as np
import copy
import random
import math
from math import pi
from scipy.stats import chi2
######################################################
###########################################
###########################################
# GRAPHICAL FUNCTIONS
import matplotlib.pyplot as plt
import matplotlib as mpl
def compare2matrices(matrixtoplot1,matrixtoplot2,mymin=None,mymax=None,filename=None):

	dim=np.shape(matrixtoplot1)[0]

	fig, axes = plt.subplots(nrows=1, ncols=2)

	if mymin==None:
		mymin1=np.min(matrixtoplot1)
		mymin2=np.min(matrixtoplot2)
		mymin=np.min([mymin1,mymin2])
	if mymax==None:
		mymax1=np.max(matrixtoplot1)
		mymax2=np.max(matrixtoplot2)
		mymax=np.max([mymax1,mymax2])
	for ax,mat in zip(axes.flat,[matrixtoplot1,matrixtoplot2]):
		ax.matshow(mat,vmin=mymin,vmax=mymax )
		ax.xaxis.set_ticks(range(dim))
		ax.yaxis.set_ticks(range(dim))
    
	cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
	norm = mpl.colors.Normalize(vmin =mymin, vmax=mymax)
	mpl.colorbar.ColorbarBase(ax=cbar_ax,norm=norm)
	if filename is not None:
    		plt.savefig(filename)
	plt.show()


def plot2matrices(mat1,mat2,ticksspacing=1,filename=None,removediagonal=True,xlabel=None,ylabel=None,title=None):
    D=len(mat1)
    mat3=np.ones((D,D))
    for i in range(D):
        for j in range(D):
            if i<j:
                mat3[i,j]=mat1[i,j]
            else:
                mat3[i,j]=mat2[i,j]
    if removediagonal:
        mat3-=np.eye(D)*np.diag(mat3)
    plt.matshow(mat3)
    plt.colorbar()
    plt.xticks(range(0,D,ticksspacing))
    plt.yticks(range(0,D,ticksspacing))
    if title is not None:
        plt.title(title,fontsize=20)
    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=20)
    if ylabel is not None:
        plt.ylabel(ylabel,fontsize=20)
    if filename is not None:
        plt.savefig(filename)
    plt.show()
###########################################
###########################################
######################################################


