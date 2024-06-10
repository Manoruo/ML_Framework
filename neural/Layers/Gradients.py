import numpy as np 


class GradientCollection():
    def __init__(self):
        self.layers_dx = []
        self.layers_dw = []
        self.layers_db = []

        self.num_layers = 0
        self.amount_accum = 0 


    def append(self, gradient):
        dx, dw, db = gradient
        
        self.layers_dx.append(dx)
        self.layers_dw.append(dw)
        self.layers_db.append(db)
        
        self.num_layers += 1

    def __add__(self, collection2): 
        assert isinstance(collection2, GradientCollection), "Cannot add GradientCollection type with non-GradientCollection type"

        if self.num_layers < 1:
            return collection2
        elif collection2.num_layers < 1:
            return self

       
        # go through each consecutive layer
        for i in range(self.num_layers):
            
            # get the corresponding values in both self and other collection 
            self.layers_dx[i] = self.layers_dx[i] + collection2.layers_dx[i] # get dx gradient at layer i 
            self.layers_dw[i] = self.layers_dw[i] + collection2.layers_dw[i] # get dw gradient at layer i 
            self.layers_db[i] = self.layers_db[i] + collection2.layers_db[i] # get db gradient at layer i 
            
        self.amount_accum += 1 
        return self



        