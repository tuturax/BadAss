#####################
# Library
#####################
import numpy as np
import pandas as pd

from layer_1.reactions   import Reaction_class
from layer_1.metabolites import Metabolite_class
from layer_1.parameters  import Parameter_class
from layer_1.elasticities import Elasticity_class
            
#####################
# Class model
#####################
class model:

    #############################################################################
    ########   Class method to creat a model from stochi matrix    ##############
    @classmethod
    def from_matrix(cls, matrix) :
        class_instance = cls()
        return(class_instance)

    #############################################################################
    #############             Initialisation                #####################
    def __init__(self):

        # Call of reaction Class
        self._reactions   = Reaction_class(self)
        # Call of metabolite Class
        self._metabolites = Metabolite_class(self)
        # Call of elasticity Class
        self._elasticities = Elasticity_class(self)
        # Call of parameter Class
        self._parameters  = Parameter_class(self)



        # Initialisation of the Matrix_Stoechio attribute
        self._Stoichio_matrix = pd.DataFrame()

        self._enzyme = pd.DataFrame(columns=["Reactions", "Value"])
        
        # Initialisation of the dynamic attributes
        self._Jacobien_reversed = np.array([])

        print("Model created \n \nTo add metabolite, use .add_meta \nTo add reaction,   use .add_reaction")

    #############################################################################
    ##################              Getter                  #####################
    @property
    def Stoichio_matrix(self) :
        return self._Stoichio_matrix

    @property
    def reactions(self) :
        return self._reactions
    
    @property
    def metabolites(self) :
        return self._metabolites
    
    @property
    def parameters(self) :
        return self._parameters

    @property
    def elasticity(self) :
        return self._elasticities
    

    # Dynamic property
    @property
    def Jacobian(self) :
        J = self.Stoichio_matrix @ self.elasticity.s
        return J
    
    @property
    def Jacobian_reversed(self) :
        J_inv = np.linalg.pinv(self.Jacobian.to_numpy())
        return J_inv
    
    @property
    def R_s_p(self) :
        return -np.dot(self.Jacobian_reversed , np.dot(self.Stoichio_matrix.to_numpy() , self.elasticity.p.to_numpy() ))
    
    @property
    def R_v_p(self) :
        return np.dot(self.elasticity.s.to_numpy() , self.R_s_p) + self.elasticity.p.to_numpy()
    
    @property
    def R_s_c(self) :
        return -np.dot(self.Jacobian_reversed , np.dot(self.Stoichio_matrix.to_numpy() , self.elasticity.s.to_numpy() ) ) + np.identity(len(self.Stoichio_matrix.to_numpy()))
    
    @property
    def R_v_c(self) :
        return np.dot(self.elasticity.s.to_numpy() , self.R_s_c)
    
    @property
    def R(self) :
        return( np.block([[self.R_s_p ],

                          [self.R_v_p ]   ])       )
    
    @property
    def enzyme(self) :
        return self._enzyme

    #################################################################################
    ######    Representation = the Dataframe of the Stoichiometric matrix     #######
    def __repr__(self) -> str:
        return str(self._Stoichio_matrix)


    

    #############################################################################
    #############   Function to update after a modification of N  ###############

    # Call the update function when the matrix_Stoichio is modified
    @Stoichio_matrix.setter
    def Stoichio_matrix(self, new_df) :
        self._Stoichio_matrix = new_df
        self._update_network()

    

    def _update_network(self, session = "Matrix") -> None :
        ### Description of the fonction
        """
        Fonction to update the dataframes after attibuated a new values to the stoichiomatrix
        """
        
        if session == "Matrix" :
            # Put the dataframee to 0
            self.metabolites.df.drop(self.metabolites.df.index, inplace=True)
            self.reactions.df.drop(self.reactions._df.index, inplace=True)

        # Deal with the reactions
        # Loop on every reaction of the stoichiometry matrix
        for reaction in self.Stoichio_matrix.columns :
            # Creation of a dictionnary that will contain every metabolite (as keys) and their stoichiometries coeff (as values)
            dict_stochio = {}

            # We also add the stochiometric coefficent to the dataframe of reaction
            for meta in self.Stoichio_matrix.index :
                if self.Stoichio_matrix.loc[meta][reaction] != 0 :
                    dict_stochio[meta] = self.Stoichio_matrix.loc[meta, reaction]
            
            # Then we add the reaction to the reactions Dataframe
            self.reactions._update(name=reaction, metabolites=dict_stochio)

        # Deal with the metabolites

        for meta in self.Stoichio_matrix.index :
            self.metabolites._update(meta)

        # We update the elasticities matrix based on the new stoichiometric matrix
        self._update_elasticity


    #################################################################################
    ############     Function to the elaticities matrix of the model     ############
    @property
    def _update_elasticity(self) :
        ### Description of the fonction
        """
        Fonction to update the dynamic matrix of the model after a direct modification of the stoichiometric matrix
        """
        for meta in self.Stoichio_matrix.index :
            if meta not in self.elasticity.s.columns :
                self.elasticity.s[meta] = [0 for i in self.elasticity.s.index]
        
        for para in self.parameters.df.index :
            if para not in self.elasticity.p.columns :
                self.elasticity.p[para] = [0 for i in self.elasticity.p.index]

        # Pandas doesn't allow to add line before at least 1 column is add
        # So we don't add the reaction part until then
        if self.elasticity.s.columns.size != 0 :
            for reaction in self.Stoichio_matrix.columns :
                if reaction not in self.elasticity.s.index :
                    self.elasticity.s.loc[reaction] = [0 for i in self.elasticity.s.columns]
        
        for reaction in self.Stoichio_matrix.columns :
            if reaction not in self.elasticity.p.index :
                self.elasticity.p.loc[reaction] = [0 for i in self.elasticity.p.columns]




       
        
    #############################################################################
    ###############  Function to creat a simple linear network ##################
    def creat_linear(self, n : int) :
        ### Description of the fonction
        """
        Fonction to create a linear system of n metabolite
            
        n         : Number of metabolite in the linear network

        """
        if n <= 1 :
            raise TypeError("Please enter an integer >= 2")
        
        else :
            # reinitialisation of the data
            self.__init__()

            matrix = np.array([[0 for i in range(n-1)] for k in range(n)])

            for i in range(n) :
                for j in range(n-1) :
                    if i==j :
                        matrix[i][j] = -1
                    elif i-1==j :
                        matrix[i][j] = 1
            
            noms_lignes   = [f'meta_{i}'     for i in range(n)]
            noms_colonnes = [f'reaction_{i}' for i in range(n-1)]

            # Attribution of the new stoichiometic matrix
            self.Stoichio_matrix = pd.DataFrame(matrix, index=noms_lignes, columns=noms_colonnes)

            self.metabolites.df.loc[f"meta_{0}"  , "External"] = True
            self.metabolites.df.loc[f"meta_{n-1}", "External"] = True








    #############################################################################
    ##################   Function to read a CSV/XLS file  #######################
    def read_CSV(self, file = "./Exemples/XLS/ecoli_core_model.xls") :
        ### Description of the fonction
        """
        Fonction read an Excel file
            
        file     : string the specify the directory of the Excel file

        """

        df = pd.read_excel(file)
        N = df.drop(df.columns[0], axis=1)
        N = N.drop(N.index)

        for ligne in df.to_numpy() :
            N.loc[ligne[0]] = ligne[1:]
        
        
        self._Stoichio_matrix = N

        self._update_network

        for meta in self.metabolites.df.index :
            if meta[-3:] == '(e)' :
                self.metabolites.df.loc[meta]['External'] = True

    #############################################################################
    ###################   Function to read a SBML file  #########################
    def read_SBML(self, file = "./Exemples/SBML/E_coli_CCM.xml") :
        ### Description of the fonction
        """
        Fonction read a SBML file
            
        file     : string the specify the directory of the SBML file

        """
        import libsbml
        
        reader = libsbml.SBMLReader()

        document = reader.readSBML(file)
        
        n_error = document.getNumErrors()
        if n_error != 0 :
            print(f"There is {n_error} in your SBML file, please fix it before to use this function")
        
        else :
            print(f"0 error detected in your SBML file")
            model = document.getModel()

            N = pd.DataFrame(dtype=float)

            for reaction in model.reactions :
                
                N[reaction.getName()] = pd.Series([0] * len(N.index), dtype = 'float64')


                reactants = reaction.getListOfReactants()
                for reactant in reactants :
                    specie = model.getSpecies(reactant.getSpecies())
                    stoichio = reactant.getStoichiometry()
                    
                    if specie.getName() not in N.index :
                        N.loc[specie.getName()] = pd.Series([0] * len(N.columns), index=N.columns, dtype = 'float64')

                    N.loc[specie.getName(), reaction.getName()] = stoichio
                    


                products = reaction.getListOfProducts()
                for product in products :
                    specie = model.getSpecies(product.getSpecies())
                    stoichio = product.getStoichiometry()

                    if specie.getName() not in N.index :
                        N.loc[specie.getName()] = pd.Series([0] * len(N.columns), index=N.columns, dtype = 'float64')

                    N.loc[specie.getName(), reaction.getName()] = stoichio

                list_species = []
                for specie in model.species :
                    list_species.append(specie.getName())
                for specie in list_species :
                    if specie not in N.index :
                        N.loc[specie] = pd.Series([0] * len(N.columns), index=N.columns, dtype = 'float64')
                
                N.fillna(0, inplace=True)

            self.Stoichio_matrix = N

    #############################################################################
    ###################   Function to check the model   #########################
    @property
    def check(self) :

        # Check the reaction
        unused_reactions = []
        for react in self._Stoichio_matrix.columns.to_list() :
            counter = 0
            for meta in self._Stoichio_matrix.index.to_list() :
                counter += np.abs(self._Stoichio_matrix.loc[meta, react])
            if counter == 0 :
                unused_reactions.append(react)
        
        # Check the metabolite
        unused_metabolites = []
        for meta in self._Stoichio_matrix.index.to_list() :
            counter = 0
            for react in self._Stoichio_matrix.columns.to_list() :
                counter += np.abs(self._Stoichio_matrix.loc[meta, react])
            if counter == 0 :
                unused_metabolites.append(meta)

        print("The following reactions are unused : \n")
        for unused_react in unused_reactions :
            print(f"-{unused_react} \n")

        print("\n \n")
        print("The following metabolites are unused : \n")
        for unused_meta in unused_metabolites :
            print(f"-{unused_meta} \n")

        return(unused_reactions, unused_metabolites)

