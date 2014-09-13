import logging 
import sys 
import unittest
import numpy
import sppy 
import numpy.testing as nptst 
from bprCython import BPRArgs, BPR, UniformUserUniformItem, AllUserUniformItem, AllUserAllItem 
from bpr import BPR as BPRPy 
from bpr import BPRArgs as BPRArgsPy 
from sandbox.util.SparseUtils import SparseUtils
from sandbox.util.MCEvaluator import MCEvaluator

class  bprCythonTest(unittest.TestCase):
    def setUp(self):
        numpy.set_printoptions(precision=3, suppress=True)
        numpy.random.seed(21)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        
        self.m = 30 
        self.n = 20 
        k = 5
        u = 0.1 
        w = 1-u
        self.X = SparseUtils.generateSparseBinaryMatrix((self.m, self.n), k, w, csarray=True)
        self.X.prune()
        
    def testInit(self): 
        args = BPRArgs()   
        k = 5
        
        learner = BPR(k, args)
        
    def testTrain(self): 
        args = BPRArgs()   
        args.learning_rate = 0.1
        k = 5
        
        learner = BPR(k, args)    
        
        maxIterations = 10
        sample_negative_items_empirically = True
        sampler = UniformUserUniformItem()

        user_factors, item_factors = learner.train(self.X, sampler, maxIterations)
        print(MCEvaluator.averageAuc(self.X, user_factors, item_factors))
        
        #Let's try regularisation 
        args.user_regularization = 1
        learner.train(self.X, sampler, maxIterations)
        
        #Let's try regularisation 
        args.positive_item_regularization = 1
        

    def testLoss(self): 
        args = BPRArgsPy()   
        #args.bias_regularization = 0
        #args.positive_item_regularization = 0
        #args.negative_item_regularization = 0 
        #args.user_regularization = 0 
        k = 5
        
        for i in range(10): 
            learner = BPR(k, args)   
            sampler = UniformUserUniformItem()        
            user_factors, item_factors, loss_samples = learner.init(self.X, sampler)
            
            ell = learner.loss(loss_samples, user_factors, item_factors)
            #print(ell) 
            
            #Now compare versus python version 
            #args = BPRArgsPy()
            learner2 = BPRPy(k, args)
            learner2.init(self.X)
            learner2.user_factors = user_factors
            learner2.item_factors = item_factors
            learner2.loss_samples = loss_samples
            ell2 = learner2.loss()
            
            self.assertEquals(ell, ell2)
        

    def testDerivatives(self): 
        args = BPRArgs()   
        args.user_regularization = 0.0 
        args.negative_item_regularization = 0 
        args.positive_item_regularization = 0 
        args.bias_regularization = 0
        args.learning_rate = 1.0 
        k = 5
        
        sampler = AllUserAllItem()   
        sampler.numAucSample= 10
        
        learner = BPR(k, args)   
        user_factors, item_factors, loss_samples = learner.init(self.X, sampler)
        
        eps = 10**-8
        
        
        #Get user, pos item and negative item 
        u = 0 
                   
        
        #Compute user derivate via perturbations 
        du1 = numpy.zeros(k)   
        du2 = numpy.zeros(k)         
        
        inds = self.X.rowInds(u)
        indsBar = numpy.setdiff1d(numpy.arange(self.X.shape[1]), inds)
        
        for i in inds: 
            for j in indsBar: 
                for ell in range(k): 
                    user_factors2 = user_factors.copy()                
                    
                    user_factors2[u, ell] = user_factors[u, ell]+eps
                    loss1 = learner.loss(loss_samples, user_factors2, item_factors)
                    user_factors2[u, ell] = user_factors[u, ell]-eps
                    loss2 = learner.loss(loss_samples, user_factors2, item_factors) 
                    
                    du1[ell] += (loss1-loss2)/(2*eps)
                    
                du3, di3, dj3 = learner.update_factors(user_factors.copy(), item_factors.copy(), u, i, j)
                du2 += du3    
                
        du1 = du1/numpy.linalg.norm(du1)      
        du2 = du2/numpy.linalg.norm(du2)        
        
        print(du1, du2)        
        
    def testSampler(self): 
        sampler = UniformUserUniformItem() 
        
        for u, i, j in sampler.generate_samples(self.X): 
            self.assertEquals(self.X[u, i], 1)
            self.assertEquals(self.X[u, j], 0)
        
        
if __name__ == '__main__':
    unittest.main()