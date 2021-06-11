#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing as mp
import os
import sys

from EstimateAOIBoundingBoxes_Standalone import performAutomaticAOIEstimationBenchmark, performAutomaticAOIEstimationDetailed, convertToBorders
from DirectoryGenerator import DirectoryGenerator

if __name__ == '__main__':
    rootFolder = sys.argv[1]
    sample_data = [rootFolder + DirectoryGenerator().getDelimiter() + f for f in os.listdir(rootFolder) if f.endswith('.png')]
    
    #performAutomaticAOIEstimationBenchmark(sample_data)
    #performAutomaticAOIEstimationDetailed(sample_data)
    convertToBorders(sample_data)