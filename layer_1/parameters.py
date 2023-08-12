#####################
# Library
#####################
import pandas as pd




#####################
# Class Parameters
#####################
class Parameter_class():

    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_model_instance):

        # Private attribute for the instance of the Main class
        self.__class_model_instance = class_model_instance

        self.df = pd.DataFrame(index = ['Temperature'], columns= ['Mean values', 'Standard deviation'])
        self.df.loc['Temperature', 'Mean values'] = 273.15
        self.df.loc['Temperature', 'Standard deviation'] = 1.0

    #################################################################################
    #########           Return the Dataframe of the reactions                         ##########
    def __repr__(self) -> str :
        return str(self.df)

    #################################################################################
    #########        Fonction to return the number of reaction                         ##########
    @property
    def len(self) :
        return len(self.df)

    #################################################################################
    #########           Fonction to add a reaction                         ##########
    def add(self, name = None, mean = 1, Standard_deviation = 1.0 ) -> None :
        ### Description of the fonction
        """
        Fonction to add a parameter to the model
            
        name                : Name of the parameter
        mean                : Mean value of the parameter
        Standard_deviation  : Standard deviation of the parameter

        """

        # Look if the parameter is already in the model
        if name in self.df.index.to_list() :
            raise TypeError("The parameter \""+ name +"\" is already in the model !")

        # Else, the parameter is add to the model by an add to the DataFrame
        else :
            self.df.loc[name] = [mean, Standard_deviation]
            
    #################################################################################
    #########           Fonction to remove a parameter                      ##########
    
    def remove(self, name : str) -> None :
        ### Description of the fonction
        """
        Fonction to remove a parameter to the model
            
        name        : Name of the reaction to remove a parameter
        """

        # Look if the reaction is in the model
        if name not in self.df.index.to_list() :
            raise TypeError("Please enter a valide name \n")

        else :
            # Else, the parameter is remove from the dataframe
            self.df.drop(name, inplace=True)
            
            # Removing this parameter from the elasticity matrix E_p
            self.__class_model_instance.elasticity.p.drop(name, axis = 1, inplace = True)

            