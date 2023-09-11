#####################
# Library
#####################
import pandas as pd
import numpy as np






#####################
# Class Elasticities
#####################
class Elasticity_class:

    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_model_instance):

        # Private attribute for the instance of the Main class
        self.__class_model_instance = class_model_instance

        self._s = pd.DataFrame()
        self._p = pd.DataFrame()

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
    # For the E_s matrix
    @property
    def s(self) :
        return self._s
    
    @s.setter
    def s(self, matrix) :
        if type(matrix)  == type(np.ndarray([])) :
            if matrix.shape != self.s.shape :
                raise IndexError("The shape of your matrix isn't matching with the elasticity matrix")
            else : 
                self._s.values[:] = matrix
                
        
        elif type(matrix) == type(pd.DataFrame()) :

            if matrix.shape != self.s.shape :
                raise IndexError("The shape of your matrix isn't matching with the elasticity matrix")
            else : 
                self._s = matrix

        else :
            raise TypeError("Please enter a numpy matrix or Pandas dataframe to fill the E_s matrix")
        

    # For the E_p matrix
    @property
    def p(self) :
        return self._p
    
    @p.setter
    def p(self, matrix) :
        if type(matrix)  == type(np.ndarray([])) :
            if matrix.shape != self.p.shape :
                raise IndexError("The shape of your matrix isn't matching with the elasticity matrix")
            else : 
                self._p.values[:] = matrix
                
        
        if type(matrix) == type(pd.DataFrame()) :
            if matrix.shape != self.p.shape :
                raise IndexError("The shape of your matrix isn't matching with the elasticity matrix")
            else : 
                self._p = matrix

        else :
            raise TypeError("Please enter a numpy matrix  or Pandas datafrmae to fill the E_s matrix")
        
    #################################################################################
    #########     Fonction to update the elasticities matrix               ##########
    def half_satured(self) :
        ### Description of the fonction
        """
        Method to attribute to the E_s matrix the value of a half-satured enzyme
        """
        self.s = -0.5*self.__class_model_instance.Stoichio_matrix.transpose()
