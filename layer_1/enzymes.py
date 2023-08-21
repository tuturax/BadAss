#####################
# Library
#####################
import pandas as pd




#####################
# Class Enzymes
#####################
class Enzymes_class():

    #############################################################################
    ###############             Initialisation              #####################
    def __init__(self, class_model_instance):

        # Private attribute for the instance of the Main class
        self.__class_model_instance = class_model_instance

        self.df = pd.DataFrame(columns= ['Concentration / Activity', 'Reactions linked'])

    #################################################################################
    ###########           Return the Dataframe of the enzymes           #############
    def __repr__(self) -> str :
        return str(self.df)

    #################################################################################
    #########        Fonction to return the number of enzymes              ##########
    @property
    def len(self) :
        return len(self.df)

    #################################################################################
    #########           Fonction to add a enzyme                         ##########
    def add(self, name = "", mean = 1, reaction_linked = []) -> None :
        ### Description of the fonction
        """
        Fonction to add an enzyme to the model
            
        name                : Name of the enzyme
        mean                : Mean value of the enzyme
        reaction_linked     : List of string of the reaction linked to this enzyme

        """

        # Look if the enzyme is already in the model
        if name in self.df.index.to_list() :
            raise TypeError("The enzyme \""+ name +"\" is already in the model !")

        # Else, the enzyme is add to the model by an add to the DataFrame
        else :
            self.df.loc[name] = [mean, reaction_linked]
            
    #################################################################################
    #########           Fonction to remove a enzyme                      ##########
    
    def remove(self, name : str) -> None :
        ### Description of the fonction
        """
        Fonction to remove an enzyme to the model
            
        name        : Name of the enzyme to remove a enzyme
        """

        # Look if the enzyme is in the model
        if name not in self.df.index.to_list() :
            raise TypeError("Please enter a valide name \n")

        else :
            # Else, the enzyme is remove from the dataframe
            self.df.drop(name, inplace=True)
            
            # Removing this enzyme from the elasticity matrix E_p
            if name in self.__class_model_instance.elasticity.p.index :
                self.__class_model_instance.elasticity.p.drop(name, axis = 1, inplace = True)



    #################################################################################
    #########      Fonction to add an enzyme link to every reaction        ##########
    def add_to_all_reaction(self) -> None :
        ### Description of the fonction
        """
        Fonction to add an enzyme to every reaction of the model
        """
        for reaction in self.__class_model_instance.reactions.df.index :
            name_enzyme = "enzyme_" + reaction
            # Look if the enzyme is already in the model
            if name_enzyme not in self.df.index.to_list() :
                self.add(name_enzyme, 1, [reaction])