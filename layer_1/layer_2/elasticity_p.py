#####################
# Library
#####################
import pandas as pd
import numpy as np


#####################
# Class Sub_Elasticities
#####################
class Elasticity_p_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self._df = pd.DataFrame(columns=["Temperature"])

    #################################################################################
    #########           Return the Dataframe of the elasticity p           ##########
    def __repr__(self) -> str:
        return str(self._df)

    #################################################################################
    #########        Fonction to return the shape of the matrix            ##########
    @property
    def len(self):
        return self._df.shape

    # For the E_p matrix
    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, matrix):
        if isinstance(matrix, np.ndarray):
            if matrix.shape != self.df.shape:
                raise IndexError(
                    "The shape of your matrix isn't matching with the elasticity matrix"
                )
            else:
                self._df.values[:] = matrix
                self.__class_MODEL_instance._reset_value(session="E_p")

        elif type(matrix) == type(pd.DataFrame()):
            if matrix.shape != self.df.shape:
                raise IndexError(
                    "The shape of your matrix isn't matching with the elasticity matrix"
                )
            else:
                self._df = matrix
                self.__class_MODEL_instance._reset_value(session="E_p")

        else:
            raise TypeError(
                "Please enter a numpy matrix  or Pandas datafrmae to fill the E_s matrix"
            )

    #################################################################################
    #########        Fonction to change a coefficient of the matrix        ##########
    def change(self, flux_name: str, parameter_name: str, value: float):
        if flux_name not in self.df.index:
            raise NameError(f"The flux name '{flux_name}' is not in the model")
        elif parameter_name not in self.df.columns:
            raise NameError(
                f"The parameter name '{parameter_name}' is not in the model"
            )
        else:
            self.df.at[flux_name, parameter_name] = value
            self.__class_MODEL_instance._reset_value(session="E_p")
