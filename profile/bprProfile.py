import numpy
import logging
import sys
from sandbox.util.ProfileUtils import ProfileUtils
from bpr import BPRArgs, BPR, UniformPairWithoutReplacement, Sampler 
from bprCython import UniformUserUniformItem
from sandbox.util.SparseUtils import SparseUtils


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class bprProfile(object):
    def __init__(self):
        numpy.random.seed(21)        
        
        #Create a low rank matrix  
        m = 500
        n = 200
        self.k = 8 
        self.X = SparseUtils.generateSparseBinaryMatrix((m, n), self.k, csarray=True)
        print(self.X)
        
    def profileLearnModel(self):
        args = BPRArgs()   
        args.learning_rate = 0.1
        k = 5
        
        learner = BPR(k, args)    
        
        maxIterations = 5
        sample_negative_items_empirically = True
        sampler = UniformUserUniformItem(sample_negative_items_empirically)
        
                
        ProfileUtils.profile('learner.train(self.X, sampler, maxIterations)', globals(), locals())
        
profiler = bprProfile()
profiler.profileLearnModel()  