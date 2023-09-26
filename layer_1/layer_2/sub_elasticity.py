#####################
# Library
#####################
import pandas as pd
import numpy as np






#####################
# Class Sub_Elasticities
#####################
class Sub_Elasticity_class:

    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_model_instance):

        # Private attribute for the instance of the Main class
        self.__class_model_instance = class_model_instance

        self.__df = pd.DataFrame()
        self.__thermo     = pd.DataFrame()
        self.__enzyme     = pd.DataFrame()
        self.__regulation = pd.DataFrame()

    #################################################################################
    #########           Return the Dataframe of the elasticity p           ##########
    def __repr__(self) -> str:
        return str(self.__df)

    #################################################################################
    #########        Fonction to return the shape of the matrix            ##########
    @property
    def len(self) :
        return (self.__df.shape)

    #################################################################################
    #########        Setter to change the elasticities matrix              ##########
    # For the E_s matrix
    @property
    def df(self) :
        if False :#self.__thermo.eq(0).all().all() & self.__enzyme.eq(0).all().all() & self.__regulation.eq(0).all().all() :
            self.__df = self.__thermo - self.__enzyme + self.__regulation
            return self.__df
        else :
            return(self.__df)
    
    @df.setter
    def df(self, matrix) :

        if type(matrix)  == type(np.ndarray([])) :
            if matrix.shape != self.df.shape :
                raise IndexError("The shape of your matrix isn't matching with the elasticity matrix")
            else : 
                self.__df.values[:] = matrix

        elif type(matrix) == type(pd.DataFrame()) :
            if matrix.shape != self.__df.shape :
                raise IndexError("The shape of your matrix isn't matching with the elasticity matrix")
            else : 
                self.__df = matrix

        else :
            raise TypeError("Please enter a numpy matrix or Pandas dataframe to fill the E_s matrix")
    

    #################################################################################
    #########     Fonction to update the elasticities matrix               ##########
    def half_satured(self) :
        ### Description of the fonction
        """
        Method to attribute to the E_s matrix the value of a half-satured enzyme
        """
        self.__df = -0.5*self.__class_model_instance.Stoichio_matrix.transpose()
