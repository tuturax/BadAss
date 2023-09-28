#####################
# Library
#####################
import pandas as pd
import numpy as np

from layer_1.layer_2.sub_elasticity    import Sub_Elasticity_class




#####################
# Class Elasticities
#####################
class Elasticity_class:

    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_model_instance):

        # Private attribute for the instance of the Main class
        self.__class_model_instance = class_model_instance

        self.s = Sub_Elasticity_class(class_model_instance)
        self.__p = pd.DataFrame()

    #################################################################################
    #########           Return the Dataframe of the elasticity p           ##########
    def __repr__(self) -> str:
        return str(self.p)

    #################################################################################
    #########        Fonction to return the shape of the matrix            ##########
    @property
    def len(self) :
        return len(self.p.shape)

    #################################################################################
    #########        Setter to change the elasticities matrix              ##########
    """    
    # For the E_s matrix
    @property
    def s(self) :
        return self.__s    """


    # For the E_p matrix
    @property
    def p(self) :
        return self.__p
    
    @p.setter
    def p(self, matrix) :
        if type(matrix)  == type(np.ndarray([])) :
            if matrix.shape != self.p.shape :
                raise IndexError("The shape of your matrix isn't matching with the elasticity matrix")
            else : 
                self.__p.values[:] = matrix
                
        
        if type(matrix) == type(pd.DataFrame()) :
            if matrix.shape != self.p.shape :
                raise IndexError("The shape of your matrix isn't matching with the elasticity matrix")
            else : 
                self.__p = matrix

        else :
            raise TypeError("Please enter a numpy matrix  or Pandas datafrmae to fill the E_s matrix")