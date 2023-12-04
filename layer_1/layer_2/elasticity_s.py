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
    def __init__(self, class_MODEL_instance):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self._df = pd.DataFrame()
        self.thermo = pd.DataFrame()
        self.enzyme = pd.DataFrame()
        self.regulation = pd.DataFrame()

    #################################################################################
    #########           Return the Dataframe of the elasticity p           ##########
    def __repr__(self) -> str:
        return str(self._df)

    #################################################################################
    #########        Fonction to return the shape of the matrix            ##########
    @property
    def len(self):
        return self._df.shape

    #################################################################################
    #########        Setter to change the elasticities matrix              ##########
    # For the E_s matrix
    @property
    def df(self):
        if (
            self.thermo.eq(0).all().all()
            and self.enzyme.eq(0).all().all()
            and self.regulation.eq(0).all().all()
        ):
            return self._df
        else:
            self._df = self.thermo - self.enzyme + self.regulation
            return self._df

    @df.setter
    def df(self, matrix):
        if type(matrix) == type(np.ndarray([])):
            if matrix.shape != self.df.shape:
                raise IndexError(
                    "The shape of your input matrix isn't matching with the elasticity matrix"
                )
            else:
                self._df.values[:] = matrix
                self.__class_MODEL_instance._reset_value(session="E_s")

        elif type(matrix) == type(pd.DataFrame()):
            self._df = matrix
            self.__class_MODEL_instance._reset_value(session="E_s")

        else:
            raise TypeError(
                "Please enter a numpy matrix or Pandas dataframe to fill the E_s matrix"
            )

    #################################################################################
    #########     Fonction to update the elasticities matrix               ##########
    def reset(self):
        ### Description of the fonction
        """
        Method to reset the value of the elasticity E_s and sub_elasticities
        """
        # Reset of the sub_elasticity dataframe
        self.thermo.fillna(0, inplace=True)
        self.enzyme.fillna(0, inplace=True)
        self.regulation.fillna(0, inplace=True)

        # Reset of the main elasticity dataframe
        self.df.fillna(0, inplace=True)
        # Reset of the value of the system
        self.__class_MODEL_instance._reset_value(session="E_s")

    #################################################################################
    #########     Fonction to update the elasticities matrix               ##########
    def half_satured(self):
        ### Description of the fonction
        """
        Method to attribute to the E_s matrix the value of a half-satured enzyme
        """
        self.reset()
        self._df = -0.5 * self.__class_MODEL_instance.N_without_ext.transpose()

    #################################################################################
    #########        Fonction to change a coefficient of the matrix        ##########
    def change(self, flux_name: str, metabolite_name: str, value: float):
        if flux_name not in self.df.index:
            raise NameError(f"The flux name '{flux_name}' is not in the model")
        elif metabolite_name not in self.df.columns:
            raise NameError(
                f"The parameter name '{metabolite_name}' is not in the model"
            )
        else:
            self.df.at[flux_name, metabolite_name] = value
            self.__class_MODEL_instance._reset_value(session="E_s")

    #################################################################################
    #########     Fonction to update the elasticities matrix               ##########
    def fill_sub_elasticity(self, a=1, b=1):
        ### Description of the fonction
        """
        Method to fill the sub elasticities dataframes
        """
        self.reset()

        N = self.__class_MODEL_instance.Stoichio_matrix.to_numpy()

        # Definition of the sub_elasticity of the thermodynamic effects
        M_plus = np.zeros(N.shape)
        M_moins = np.zeros(N.shape)

        for i in range(N.shape[0]):
            for j in range(N.shape[1]):
                if N[i][j] > 0.0:
                    M_moins[i][j] = np.abs(N[i][j])

                if N[i][j] < 0.0:
                    M_plus[i][j] = np.abs(N[i][j])

        M_moins = np.transpose(M_moins)
        M_plus = np.transpose(M_plus)

        L, N_red = self.__class_MODEL_instance.Link_matrix

        c_int_df = self.__class_MODEL_instance.metabolites.df.loc[
            self.__class_MODEL_instance.metabolites.df.index.isin(N_red.index)
        ]
        c_int = c_int_df["Concentration (mmol/gDW)"].to_numpy()

        k_eq = self.__class_MODEL_instance.reactions.df[
            "Equilibrium constant"
        ].to_numpy()

        zeta = np.exp(
            np.log(k_eq) - np.dot(np.transpose(N_red.to_numpy()), np.log(c_int))
        )

        premier_terme = np.linalg.pinv(
            np.diag(np.transpose(zeta - np.ones((k_eq.shape[0], k_eq.shape[0])))[0])
        )
        second_terme = np.dot(np.diag(zeta), M_plus) - M_moins

        ela_thermo = np.dot(premier_terme, second_terme)

        for i, index in enumerate(self.thermo.index):
            for j, column in enumerate(self.thermo.columns):
                self.thermo.at[index, column] = ela_thermo[i][j]

        # Definition of the sub_elasticity of the enzymes effects
        beta = np.random.beta(a=a, b=b, size=np.shape(np.transpose(N)))
        ela_enzyme = np.multiply(beta, np.transpose(N))

        for i, index in enumerate(self.enzyme.index):
            for j, column in enumerate(self.enzyme.columns):
                self.enzyme.at[index, column] = ela_enzyme[i][j]

        # self.enzyme.values[:,:] = ela_enzyme

        # Definition of the sub_elasticity of the regulations effects
        beta = np.random.beta(a=a, b=b, size=np.shape(np.transpose(N)))
        alpha = np.random.beta(a=a, b=b, size=np.shape(np.transpose(N)))

        W_acti = W_inib = np.ones(np.shape(np.transpose(N)))

        ela_regu = np.multiply(alpha, W_acti) - np.multiply(beta, W_inib)

        # self.regulation.values[:,:] = ela_regu

        for i, index in enumerate(self.regulation.index):
            for j, column in enumerate(self.regulation.columns):
                self.regulation.at[index, column] = ela_regu[i][j]
