#####################
# Library
#####################
import pandas as pd
import numpy as np


#####################
# Class Proteins
#####################
class Proteins_class:
    #############################################################################
    ###############             Initialisation              #####################
    def __init__(self, class_MODEL_instance):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        # creation of the protein dataframe
        self.__df = pd.DataFrame(columns=["Concentration", "Type", "Reactions influenced", "Influencers", "Intrinsic noise SD"])

        # Private list to deal with the fact that a dataframe cannot be filled if there is no collumn in the dataframe
        self.__cache_prot = []

        # Private value of matrix
        self.__A = pd.DataFrame()
        self.__B = pd.DataFrame()

    #################################################################################
    ###########          Return the Dataframe of the proteins           #############
    def __repr__(self) -> str:
        return str(self.df)

    #################################################################################
    #########          Fonction to return the number of proteins           ##########
    @property
    def len(self):
        return len(self.df)

    #################################################################################
    ########   Function to have access to the private df of the proteins   ##########
    @property
    def df(self):
        # We sort the value of the portein dataframe in order to display the TF first and then the enzymes
        order = {'TF': 0, 'enzyme': 1}

        self.__df.sort_values(by='Type', key=lambda x: x.map(order), inplace=True)

        return(self.__df)
    
    #################################################################################
    #########    Fonction to return the A matrix (prot-prot influence)     ##########
    @property
    def A(self):
        self.__A = self.__A.reindex(index=self.df.index, columns=self.df.index)
        return self.__A
    
    @A.setter
    def A(self, matrix):
        # If the new matrix don't have the same shape as the previous one
        if isinstance(matrix, np.ndarray):
                # If the new matrix don't have the same shape as the previous one
                if matrix.shape != self.df.shape:
                    # Then we report an error
                    raise IndexError("The shape of your input matrix isn't matching with the A matrix")
                
                # Else, we atribute the value of the np matrix to the elasticity dataframe
                else:
                    self.__df.values[:] = matrix


        # If the new matrix is a dataframe
        elif isinstance(matrix, pd.DataFrame):
            
            # We attribute this dataframe as the new elasticity matrix
            self.__df = matrix

        # If the new matrix is neither a np or pd one, we report an error
        else:
            raise TypeError(
                "Please enter a numpy matrix or Pandas dataframe to fill the A matrix"
            )

    #################################################################################
    #########    Fonction to return the B matrix (prot-metabolite influence)     ##########
    @property
    def B(self):
        list_meta_model = self.__class_MODEL_instance.N_without_ext.index
        for meta in list_meta_model :
            if meta not in self.__B.columns :
                self.__B[meta] = 0

        return self.__B
    
    @B.setter
    def B(self, matrix):
        # If the new matrix don't have the same shape as the previous one
        if isinstance(matrix, np.ndarray):
                # If the new matrix don't have the same shape as the previous one
                if matrix.shape != self.df.shape:
                    # Then we report an error
                    raise IndexError("The shape of your input matrix isn't matching with the B matrix")
                
                # Else, we atribute the value of the np matrix to the elasticity dataframe
                else:
                    self.__df.values[:] = matrix


        # If the new matrix is a dataframe
        elif isinstance(matrix, pd.DataFrame):
            
            # We attribute this dataframe as the new elasticity matrix
            self.__df = matrix

        # If the new matrix is neither a np or pd one, we report an error
        else:
            raise TypeError(
                "Please enter a numpy matrix or Pandas dataframe to fill the B matrix"
            )


    #################################################################################
    #########                Fonction to add a protein                     ##########
    def add(self, name="", mean=1, type = "TF", reactions_influenced_dict={}, influencer_dict={}, int_noise = 0) -> None:
        ### Description of the fonction
        """
        Fonction to add a protein to the model

        Parameters
        ----------

        name                : str
            Name of the protein\n

        mean                : float
            Mean value of the protein\n

        type                : str
            type of the protein ("TF" or "enzyme")\n

        reactions_influenced_dict    : dict
            Dictionnary of reaction names and the coefficient of how they are influenced by this enzyme\n

        influencer_dict  : dict
            dict of metabolites names that influence this TF\n
        
        int_noise           : float
            Standard deviation of intrinsic the noise\n
        """

        # Look if the protein is already in the model
        if name in self.df.index.to_list():
            raise NameError(f"The protein name {name} is already in the model !\n")

        # Look if the protein is already in the model
        if name in self.__class_MODEL_instance.metabolites.df.index.copy() :
            raise NameError(f"The protein name {name} can't have the same name than a metabolite !\n")

        # The concentration must be over 0
        if mean < 0 :
            raise ValueError(f"The mean concentration of the protein must be a float superior at 0 !\n")
        
        # Same with the intrinsic noise 
        if int_noise < 0 :
            raise ValueError(f"The input value for ext_noise must be a float over 0 !\n")

        # We check the type of the input of the type of protein
        if type.lower()[0] == "t" :
            type = "TF"
            
        elif type.lower()[0] == "e" :
            type = "enzyme"
        else :
            raise NameError(f"The input type '{type}' is invalide, must be TF or enzyme !\n")
        


        # Then we check if the reaction that the protein have an influence on is in the model
        list_reaction_model = self.__class_MODEL_instance.reactions.df.index
        for reaction in reactions_influenced_dict.keys() :
            # If a reaction of the dict isn't in the model, we raise an error
            if reaction not in list_reaction_model :
                raise NameError(f"The reaction linked '{reaction}' is not in the reaction dataframe ! \n")
        
        # Same with the internal metabolites or proteins that have linear effects on this protein
        list_meta_model = self.__class_MODEL_instance.N_without_ext.index
        list_prot_model = self.df.index

        influencer_dict
        for influ in influencer_dict.keys() :
            if influ not in list_meta_model and influ not in list_prot_model :
                raise NameError(f"The metabolite linked '{influ}' is not in the metabolite or protein dataframe ! \n")
        

        # At this state, everythings is alright
        # So we add (or change) the protein to the dataframe
        if name in self.df.index :
            self.change(name, mean, type, reactions_influenced_dict, influencer_dict, int_noise)
        else : 
            self.df.loc[name] = [mean, type, reactions_influenced_dict, influencer_dict, int_noise]

        # And we add a reaction of consumption of the protein
        alpha = 1.0
        self.__class_MODEL_instance.reactions.add(name="consumption_" + name, metabolites={name: -1*alpha})


        # Then, we change the A matrix that represent the interaction between each protein
        # We add a line and a column at A matrix
        self.A[name] = 0
        self.A.loc[name] = [0 for _ in self.A.columns]
        # Than we fill this new line
        for influ, coeff in influencer_dict.items() :
            if influ in list_prot_model :
                self.__A.at[name, influ] = coeff

        # Same with the B matrix that represent the influence of metaboite on protein.
        self.B.loc[name] = [0 for _ in self.B.columns]
        # Than we fill this new line
        for influ, coeff in influencer_dict.items() :
            if influ in list_meta_model :
                self.__B.at[name, influ] = coeff


        # If there is no reaction in the model, we keep in memory the protein
        if self.__class_MODEL_instance.Stoichio_matrix_pd.columns.size == 0:
            self.__cache_prot.append(name)
            print("Don't worry, the metabolite will be add after the add of the 1st reaction")
        

        #Else, we add every protein that we keeped into memory to the stoichiometrix matrix
        else:
            self.__cache_prot.append(name)
            for prot in self.__cache_prot:
                if prot not in self.__class_MODEL_instance.Stoichio_matrix_pd.index:
                    self.__class_MODEL_instance.Stoichio_matrix_pd.loc[prot] = 0.0

            self.__cache_prot = []


    #################################################################################
    #########                Fonction to change a protein                     ##########
    def change(self, name="", mean=True, type = True, reactions_influenced_dict=True, influencer_dict=True, int_noise = True) -> None:
        ### Description of the fonction
        """
        Fonction to change a protein to the model

        Parameters
        ----------

        name                : str
            Name of the protein\n

        mean                : float
            Mean value of the protein\n

        type                : str
            type of the protein ("TF" or "enzyme")\n

        reactions_influenced_dict    : dict
            Dictionnary of reaction names and the coefficient of how they are influenced by this enzyme\n

        influencer_dict  : dict
            dict of metabolites names that influence this TF\n
        
        int_noise           : float
            Standard deviation of intrinsic the noise\n
        """

        # Look if the protein is already in the model
        if name not in self.df.index.to_list():
            raise NameError(f"The protein name {name} is not in the model !\n")


        # If the user have enter a float for the mean concentration of the protein
        if isinstance(mean, float) :
            # The concentration must be over 0
            if mean < 0 :
                raise ValueError(f"The mean concentration of the protein must be a float superior at 0 !\n")
            # Then we replace it
            else :
                self.df.at[name, "Concentration"]
    
        # Same with the intrinsic noise 
        if isinstance(int_noise, float) :
            if int_noise < 0 :
                raise ValueError(f"The input value for int_noise must be a float over 0 !\n")
            else :
                self.df.at[name, "Intrinsic noise SD"]


        # We check the type of the input of the type of protein
        if isinstance(type, str) :
            if type.lower()[0] != self.df.at[name, "Type"] :
                if type.lower()[0] == "t" :
                    type = "TF"
                elif type.lower()[0] == "e" :
                    type = "enzyme"
                else :
                    raise NameError(f"The input type '{type}' is invalide, must be TF or enzyme !\n")
                
                self.df.at[name, "Type"]


        # Then we check if the reaction that the protein have an influence on is in the model
        if isinstance(reactions_influenced_dict, dict) :
            if reactions_influenced_dict != self.df.at[name, "Reactions influenced"] :
                # First we keep in memory the reaction of the model
                list_reaction_model = self.__class_MODEL_instance.reactions.df.index
                # If a reaction of the dict isn't in the model, we raise an error
                for reaction in reactions_influenced_dict.keys() :
                    if reaction not in list_reaction_model :
                        raise NameError(f"The reaction linked '{reaction}' is not in the reaction dataframe ! \n")
                
                self.df.at[name, "Reactions influenced"]

        # Same with the internal metabolites or proteins that have linear effects on this protein
        list_meta_model = self.__class_MODEL_instance.N_without_ext.index
        list_prot_model = self.df.index

        if isinstance(influencer_dict, dict) :
            if influencer_dict != self.df.at[name, "Influencers"] :
                # If an influencer of the dict isn't in the metabolite or protein dataframe, we raise an error
                for influ in influencer_dict.keys() :
                    if influ not in list_meta_model and influ not in list_prot_model :
                        raise NameError(f"The metabolite linked '{influ}' is not in the metabolite or protein dataframe ! \n")
                
                self.df.at[name, "Influencers"]


        # At this state, everythings is alright
        

        # Then, we change the A matrix that represent the interaction between each protein
        if isinstance(influencer_dict, dict) :
            # First we reset the value of the line => no proteins have an influence on the changed protein
            self.A.loc[name,:] = 0
            self.B.loc[name,:] = 0
            # Then we add the new coeff to the matrix
            for influ, coeff in influencer_dict.items() :
                if influ in list_prot_model :
                    self.__A.at[name, influ] = coeff
                elif influ in list_meta_model :
                    self.__B.at[name, influ] = coeff





    #################################################################################
    #########           Fonction to remove an enzyme                       ##########

    def remove(self, name: str) -> None:
        ### Description of the fonction
        """
        Fonction to remove a protein\n

        Parameters
        ----------

        name        : str
            Name of the transcription factor to remove
        """

        # Look if the enzyme is in the model
        if name not in self.df.index:
            raise NameError(
                f"The protein '{name}' is not in the protein dataframe, please enter a valide name \n"
            )

        else:
            # Else, the enzyme is remove from the dataframe
            self.df.drop(name, inplace=True)
            # And also remove from the A matrix
            self.__A.drop(name, axis=0, inplace=True)
            self.__A.drop(name, axis=1, inplace=True)
            # And B
            self.__B.drop(name, axis=0, inplace=True)

            self.__class_MODEL_instance.reactions.remove(name="consumption_" + name)

