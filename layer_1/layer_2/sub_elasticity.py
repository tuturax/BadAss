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
        self.thermo     = pd.DataFrame()
        self.enzyme     = pd.DataFrame()
        self.regulation = pd.DataFrame()

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
            self.__df = self._thermo - self.__enzyme + self.__regulation
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
    def reset(self) :
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

    #################################################################################
    #########     Fonction to update the elasticities matrix               ##########
    def half_satured(self) :
        ### Description of the fonction
        """
        Method to attribute to the E_s matrix the value of a half-satured enzyme
        """
        self.reset()
        self.__df = -0.5*self.__class_model_instance.Stoichio_matrix.transpose()

    #################################################################################
    #########     Fonction to update the elasticities matrix               ##########
    def fill_sub_elasticity(self, a=1, b=1 ) :
        ### Description of the fonction
        """
        Method to fill the sub elasticities dataframes
        """
        self.reset()

        N = self.__class_model_instance.Stoichio_matrix.to_numpy()

        # Definition of the sub_elasticity of the thermodynamic effects
        M_plus  = np.zeros( N.shape )
        M_moins = np.zeros( N.shape )

        for i in range(N.shape[0]) :
            for j in range(N.shape[1]) :
            
                if N[i][j] > 0.0 :
                    M_moins[i][j] = np.abs(N[i][j])
                
                if N[i][j] < 0.0 : 
                    M_plus[i][j] = np.abs(N[i][j])

        M_moins = np.transpose(M_moins)
        M_plus = np.transpose(M_plus)

        L, N_red = self.__class_model_instance.Link_matrix

        c_int = self.__class_model_instance.metabolites.df.loc[self.__class_model_instance.metabolites.df.index.isin(N_red.index)] 
        
        k_eq = self.__class_model_instance.reactions.df["Equilibrium constant"].to_numpy()

        zeta = np.exp(     np.log(k_eq)    -    np.dot( np.transpose(N_red) , np.log(c_int)  )      )

        print(type(np.log(k_eq)))

        premier_terme = np.linalg.pinv(np.diag(np.transpose(zeta - np.ones((k_eq.shape[0],k_eq.shape[0])))[0]))
        second_terme = np.dot(np.diag(zeta.T[0]) , M_plus)  - M_moins 

        ela_thermo = np.dot( premier_terme , second_terme  )
        
        self.thermo.values[:] = ela_thermo

        # Definition of the sub_elasticity of the enzymes effects
        beta = np.random.beta(a=a, b=b, size=np.shape(np.transpose(N)))
        ela_enzyme = np.multiply(beta, np.transpose(N))

        self.enzyme.values[:] = ela_enzyme

        # Definition of the sub_elasticity of the regulations effects
        beta  = np.random.beta(a=a, b=b, size=np.shape(np.transpose(N)))
        alpha = np.random.beta(a=a, b=b, size=np.shape(np.transpose(N)))

        W_acti = W_inib = np.ones(np.shape(np.transpose(N)))
        
        ela_regu = np.multiply(alpha, W_acti) - np.multiply(beta, W_inib)

        self.regulation.values[:] = ela_regu



    