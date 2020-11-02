import numpy as np

class halfNormalSampler:
    def __init__(self, mu = 0.0, scale = 1.0, prob = 1.0):

        self.mu = mu
        self.scale = scale
        self.prob = prob
        self.setNameString()

    def setNameString(self):
        self.name = 'halfNormalSampler(' + str(self.mu) + ', ' + str(self.scale) + ', ' + str(self.prob) + ')'

    def setMu(self, mu: float):
        self.mu = mu
        self.setNameString()

    def setScale(self, scale: float):
        self.scale = scale
        self.setNameString()

    def setProb(self, prob: float):
        self.prob = prob

        self.setNameString()

    def sample(self):
        time_offset = np.abs(np.random.normal(loc=self.mu, scale=self.scale))
        use_val = (np.random.uniform(0,1) < self.prob)        
    
        return time_offset if use_val else 0

    
class constantSampler:
    def __init__(self, offsetMinutes = 5):
        self.offsetMinutes = offsetMinutes
        self.setNameString()

    def setNameString(self):
        self.name = 'constantSampler(' + str(self.offsetMinutes) + ')'

    def setOffsetMinutes(self, offsetMinutes: float):
        self.offsetMinutes = offsetMinutes

    def sample(self): 
        return self.offsetMinutes