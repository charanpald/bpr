#cython: profile=True 
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

import random 

class Sampler(object):

    def __init__(self,sample_negative_items_empirically):
        self.sample_negative_items_empirically = sample_negative_items_empirically

    def init(self,data,max_samples=None):
        self.data = data
        self.num_users, self.num_items = data.shape
        self.max_samples = max_samples

    def sample_user(self):
        cdef unsigned int u
        u = self.uniform_user()
        #num_items = self.data[u].getnnz()
        #assert(num_items > 0 and num_items != self.num_items)
        return u

    def sample_negative_item(self,user_items):
        cdef unsigned int j
        
        j = self.random_item()
        while j in user_items:
            j = self.random_item()
        return j

    def uniform_user(self):
        return random.randint(0,self.num_users-1)

    def random_item(self):
        """sample an item uniformly or from the empirical distribution
           observed in the training data
        """
        cdef unsigned int u, i
        if self.sample_negative_items_empirically:
            # just pick something someone rated!
            u = self.uniform_user()
            i = random.choice(self.data.rowInds(u))
        else:
            i = random.randint(0,self.num_items-1)
        return i

    def num_samples(self, unsigned int n):

        if self.max_samples is None:
            return n
        return min(n,self.max_samples)

class UniformUserUniformItem(Sampler):

    def generate_samples(self,data,max_samples=None):
        cdef unsigned int u 
        cdef unsigned int i, j 
            
        
        self.init(data,max_samples)
        for _ in xrange(self.num_samples(self.data.nnz)):
            u = self.uniform_user()
            # sample positive item
            i = random.choice(self.data.rowInds(u))
            j = self.sample_negative_item(self.data.rowInds(u))
            yield u,i,j