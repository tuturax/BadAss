#####################
# Library
#####################
import numpy as np
import pandas as pd

from layer_1.reactions    import Reaction_class
from layer_1.metabolites  import Metabolite_class
from layer_1.parameters   import Parameter_class
from layer_1.elasticities import Elasticity_class
from layer_1.enzymes      import Enzymes_class
            
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
        self.__reactions   = Reaction_class(self)
        # Call of metabolite Class
        self.__metabolites = Metabolite_class(self)
        # Call of elasticity Class
        self.__elasticities = Elasticity_class(self)
        # Call of parameter Class
        self.__parameters  = Parameter_class(self)
        # Call of enzyme Class
        self.__enzymes  = Enzymes_class(self)




        # Initialisation of the Matrix_Stoechio attribute
        self._Stoichio_matrix = pd.DataFrame()

        self._enzyme = pd.DataFrame(columns=["Reactions", "Value"])
        
        # Initialisation of the dynamic attributes
        self._Jacobien_reversed = np.array([])

        # Sampling file of the model
        self.data_sampling = pd.DataFrame(columns=["Name","Type", "Standard deviation", "Distribution"])

        print("Model created \n \nTo add metabolite, use .metabolites.add_meta \nTo add reaction,   use .reactions.add_reaction")
    #################################################################################
    ######    Representation = the Dataframe of the Stoichiometric matrix     #######
    def __repr__(self) -> str:
        return str(self._Stoichio_matrix)


    #############################################################################
    ##################              Getter                  #####################
    @property
    def Stoichio_matrix(self) :
        return self._Stoichio_matrix
    
    @property
    def __Stoichio_matrix(self) :
        return self.Stoichio_matrix.to_numpy()
    @property
    def reactions(self) :
        return self.__reactions
    
    @property
    def metabolites(self) :
        return self.__metabolites
    
    @property
    def enzymes(self) :
        return self.__enzymes

    @property
    def parameters(self) :
        return self.__parameters

    @property
    def elasticity(self) :
        return self.__elasticities
    
   
    # The attibute with __ are the one compute with numpy and aim to be call for other compuation
    # The attribute without it are only the representation of the them on dataframe
    
    # Dynamic property
    @property
    def __Jacobian(self) :
        return np.dot(self.Stoichio_matrix.to_numpy() , self.elasticity.s.to_numpy() )
    @property
    def Jacobian(self) :
        return pd.DataFrame(self.__Jacobian, index = self.metabolites.df.index, columns = self.elasticity.s.columns)


    @property
    def __Jacobian_reversed(self) :
        return  np.linalg.pinv(self.__Jacobian)  
    @property
    def Jacobian_reversed(self) :
        return pd.DataFrame(self.__Jacobian_reversed, index=self.Jacobian.columns, columns=self.Jacobian.index)


    @property
    def __R_s_p(self) :
        return -np.dot(self.__Jacobian_reversed , np.dot(self.__Stoichio_matrix , self.elasticity.p.to_numpy() ))
    @property
    def R_s_p(self) :
        return pd.DataFrame(self.__R_s_p, index = self.metabolites.df.index, columns = self.parameters.df.index)
    
    @property
    def __R_v_p(self) :
        return np.dot(self.elasticity.s.to_numpy() , self.__R_s_p) + self.elasticity.p.to_numpy()
    @property
    def R_v_p(self) :
        return pd.DataFrame(self.__R_v_p, index = self.reactions.df.index, columns = self.parameters.df.index)
    
    @property
    def __R_s_c(self) :
        return -np.dot(self.Jacobian_reversed , np.dot(self.__Stoichio_matrix , self.elasticity.s.to_numpy() ) ) + np.identity(len(self.__Stoichio_matrix))
    @property
    def R_s_c(self) :
        return pd.DataFrame(self.__R_s_c, index = self.metabolites.df.index, columns = self.metabolites.df.index)
    
    @property
    def __R_v_c(self) :
        return np.dot(self.elasticity.s.to_numpy() , self.__R_s_c)
    @property
    def R_v_c(self) :
        return pd.DataFrame(self.__R_v_c, index = self.reactions.df.index, columns = self.metabolites.df.index)

    @property
    def __R(self) :
        return( np.block([[self.R_s_p ],

                          [self.R_v_p ]   ])       )
    @property
    def R(self) :
        return pd.DataFrame(self.__R, index = self.metabolites.df.index.to_list() + self.reactions.df.index.to_list() , columns=self.parameters.df.index)


    @property
    def __Standard_deviations(self) :
        return self.parameters.df['Standard deviation']
    
    @property
    def __covariance(self) :
        matrix_covariance_dx = np.identity(len(self.__Standard_deviations))
        for i in range(len(matrix_covariance_dx)) :
            matrix_covariance_dx[i][i] = self.__Standard_deviations[i]**2
        
        matrix_RC = np.dot(self.R , matrix_covariance_dx)

        return(       np.block([[   matrix_covariance_dx              ,                     np.transpose(matrix_RC)         ],

                                [   matrix_RC                         ,         np.dot(matrix_RC,np.transpose(self.R))      ]      ])  ) 
    @property
    def covariance(self) :
        return pd.DataFrame(self.__covariance, index = (self.parameters.df.index.to_list() + self.R.index.to_list()) , columns = ( self.parameters.df.index.to_list() + self.R.index.to_list() ) )

    @property
    def __MI(self) :
        I = Cov = self.__covariance
        for i in range(Cov.shape[0]) :
            for j in range(Cov.shape[1]) :
                I[i][j] = (1/(2*np.log(2)))*Cov[i][i]*Cov[j][j]/Cov[i][j]
        return(I)
    @property
    def MI(self) :
        return pd.DataFrame(self.__MI, index = (self.parameters.df.index.to_list() + self.R.index.to_list()) , columns = ( self.parameters.df.index.to_list() + self.R.index.to_list() ) )




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
            self.reactions.df.drop(self.reactions.df.index, inplace=True)

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

    

    #################################################################################
    ############     Function that return the correlation coefficient    ############
    def rho(self, Absolute = False, deleted = [], dtype = float) :
        ### Description of the fonction
        """
        Fonction to compute the correlation coefficient
        """
        Cov_df = self.covariance
        Cov = Cov_df.to_numpy()

        rho = np.zeros((Cov.shape[0], Cov.shape[1]), dtype=dtype)
        
        if Absolute == True :
            for i in range(Cov.shape[0]) :
                for j in range(Cov.shape[1]) :
                    rho[i][j] = (Cov[i][j])/((np.abs(Cov[i][i])*np.abs(Cov[j][j]))**0.5)
        
        else :
            for i in range(Cov.shape[0]) :
                for j in range(Cov.shape[1]) :
                    rho[i][j] = (Cov[i][j])/((np.real(Cov[i][i])*np.real(Cov[j][j]))**0.5)

        return (pd.DataFrame(rho, index = Cov_df.index , columns = Cov_df.columns), rho)

    
        
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
        self.parameters.df.reset_index(inplace=True)
        self.enzymes.df.reset_index(inplace=True)
        self.elasticity.s.reset_index(inplace=True)
        self.elasticity.p.reset_index(inplace=True)
        

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

        self.reset
        
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

    #############################################################################
    ###################   Function plot the rho matrix   #########################
    def plot_rho(self,title = "Correlation", label = True, value_in_cell = True, index_to_keep = []) :
        import matplotlib
        import matplotlib.pyplot as plt

        # Get the rho matrix
        rho_df, rho = self.rho()

        # Look the index to keep for the plot of the matrix 
        index_to_keep_bis = []
        # If nothing is specified, we keep everything
        #else
        if len(index_to_keep) != 0 :
            # We take a look at every index that the user enter
            for index in index_to_keep :
                # If one of them is not in the model, we told him
                if index not in rho_df.index :
                    print(f"- {index} is not in the correlation matrix")
                # else, we keep in memory the index that are in the model
                else :
                    index_to_keep_bis.append(index)

        else :
            index_to_keep_bis = rho_df.index

        # Then we create a new matrix with only the index specified
        rho_df = rho_df.loc[index_to_keep_bis, index_to_keep_bis]
        rho = rho_df.to_numpy()


        fig, ax = plt.subplots()
        custom_map = matplotlib.colors.LinearSegmentedColormap.from_list( "custom", ["red", "white", "blue"])

        im = plt.imshow(rho, cmap=custom_map, vmin= -1, vmax= 1 )

        # Display the label next to the axis
        if label == True :
            ax.set_xticks(np.arange(len(rho_df.index)), labels=rho_df.index)
            ax.set_yticks(np.arange(len(rho_df.index)), labels=rho_df.index)

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Display the value of each cell 
        if value_in_cell == True :
            for i in range(rho.shape[0]):
                for j in range(rho.shape[1]):
                    text = ax.text(j, i, round(rho[i, j],2),
                            ha="center", va="center", color="black")

        # Title of the plot
        ax.set_title(title)
        fig.tight_layout()

        # Plot of the black line to separate the parameters from the variables
        # Width of the line
        line_width = 1
        # Number of parameters
        N_para = self.parameters.df.shape[0]
        # Position of the line
        x_p_e = [-0.5, N_para -.5]
        y_p_e = [N_para -.5, N_para -.5]
        plt.plot(x_p_e,y_p_e, 'black', linewidth=line_width)
        plt.plot(y_p_e,x_p_e, 'black', linewidth=line_width)

        x_p = [-0.5, N_para -.5]
        y_p = [N_para -.5, N_para -.5]
        plt.plot(x_p,y_p, 'black', linewidth=line_width)
        plt.plot(y_p,x_p, 'black', linewidth=line_width)

        plt.colorbar()
        plt.show()

    #############################################################################
    ###################   Function sampled the model    #########################
    def sampling(self, N : int) :

        # If the number of sample asked if < 1 = bad
        if N < 1 :
            raise ValueError("The number of sample must be greater or egual to 1 !")
        
        # We check every name of in the sampling dataframe
        for index in self.data_sampling.index :
            type_samp = self.data_sampling.loc[index, "Type"].lower()  # .lower() = put the string in lowercase
            name = self.data_sampling.loc[index, "Name"]


            # Case where the elasticity p is sampled
            if type_samp == "elasticity_p" :
                # If the name is not a list in the case of the elasticity, it's bad
                if type(name) != list :
                    raise TypeError("For the elasticity, be sure to use a list of 2 string, the first for the flux name and the second for the differtial of the elasticity")

                # If the list have more or less than 2 elements, it's not valide
                elif len(name) != 2 :
                    raise TypeError("For the elasticity, be sure to use a list of 2 string, the first for the flux name and the second for the differtial of the elasticity")

                else :
                    # We attribute both elements of the list, the first must be the flux name and second the differential name's
                    flux, differential = name
                    # We check if the flux is in the model
                    if flux not in self.elasticity.p.index :
                        raise NameError(f"The flux name \n{flux}\n is nnot in the elasticity matrix")

                    elif differential not in self.elasticity.p.columns : 
                        raise NameError(f"The differential name \n{differential}\n is not in the elasticity matrices E_p")
            

            # Case where the elasticity s is sampled
            elif type_samp == "elasticity_s" :
                # If the name is not a list in the case of the elasticity, it's bad
                if type(name) != list :
                    raise TypeError("For the elasticity, be sure to use a list of 2 string, the first for the flux name and the second for the differtial of the elasticity")

                # If the list have more or less than 2 elements, it's not valide
                elif len(name) != 2 :
                    raise TypeError("For the elasticity, be sure to use a list of 2 string, the first for the flux name and the second for the differtial of the elasticity")

                else :
                    # We attribute both elements of the list, the first must be the flux name and second the differential name's
                    flux, differential = name
                    # We check if the flux is in the model
                    if flux not in self.elasticity.s.index :
                        raise NameError(f"The flux name \n{flux}\n is nnot in the elasticity matrix")

                    elif differential not in self.elasticity.s.columns : 
                        raise NameError(f"The differential name \n{differential}\n is not in the elasticity matrices E_s")
            

            # Case where a parameter is sampled
            elif type_samp == "parameter" :
                if name not in self.parameters.df.index :
                        raise NameError(f"The parameter name \n{name}\n is not in the parameters dataframe")


            # Case where the metabolite concentration is sampled                  
            elif type_samp == "metabolite" :
                if name not in self.metabolites.df.index :
                        raise NameError(f"The metabolite name \n{name}\n is not in the metabolites dataframe")


            # Case where the flux s is sampled
            elif type_samp == "flux" :
                if name not in self.reactions.df.index :
                        raise NameError(f"The flux name \n{name}\n is not in the reactions dataframe")


            # Case where a enzyme concentration/activity is sampled
            elif type_samp == "enzyme" :
                if name not in self.enzymes.df.index :
                        raise NameError(f"The enzyme name \n{name}\n is not in the enzymes dataframe")


            else : 
                raise NameError(f"The type \n{type_samp}\n is not available")
        

        # Let's check if the name of the distribution are correct
        distribution_allowed = ["uniform" , "normal", "beta"]
        for index in self.data_sampling.index :
            type_Distribution = self.data_sampling.loc[index, "Distribution"]
            if type_Distribution.lower() not in distribution_allowed :
                raise NameError(f"The name of the distribution {type_Distribution} is not handle by the programme !")

        

        ## At this state, the datframe of the sampling data is well build


        def value_rand(type_samp : str , SD : str , mean : float) :
            if type_samp.lower() == "uniform" :
                deviation = (9*SD)**0.25
                np.random.uniform(mean-deviation , mean + deviation)
            
            elif type_samp.lower == "normal" :
                return np.random.normal(mean, SD)
            
            elif type_samp.lower == "beta" :
                alpha = (( ( 1-mean )/( (np.sqrt(SD))*(2-mean)**2) )-1)/(2-mean)
                beta  = alpha * (1-mean)
                return np.random.beta(alpha, beta)


        # We call of a dataframe in order to initialise the variable with the good shape and get the name of the indexs and columns
        rho_df , rho_sampled = self.rho()

        # We save the original value of the model
        self.__save_state()

        for i in  range(N) :

            for index in self.data_sampling.index :
                type_machin        = self.data_sampling.loc[index, "Type"]
                type_samp          = self.data_sampling.loc[index, "Distribution"]
                name               = self.data_sampling.loc[index, "Name"]
                standard_deviation = self.data_sampling.loc[index, "Standard deviation"]

                if type_machin.lower() == "elasticity_p" :
                    flux, differential = name
                    mean = self.__original_atributes["elasticities_p"].loc[flux, differential]
                    self.elasticity.p.loc[flux, differential] = value_rand(type_samp, standard_deviation, mean)

                elif type_machin.lower() == "elasticity_s" :
                    flux, differential = name
                    mean = self.__original_atributes["elasticities_s"].loc[flux, differential]
                    self.elasticity.s.loc[flux, differential] = value_rand(type_samp, standard_deviation, mean)

                elif type_machin.lower() == "parameter" :
                    mean = self.__original_atributes["parameters"].loc[name, 'Mean values']
                    self.parameters.loc[name, 'Mean values'] = value_rand(type_samp, standard_deviation, mean)

                elif type_machin.lower() == "metabolite" :
                    mean = self.__original_atributes["metabolite"].loc[name, 'Concentration (mmol/gDW)']
                    self.metabolites.df.loc[name, 'Concentration (mmol/gDW)'] = value_rand(type_samp, standard_deviation, mean)
                
                elif type_machin.lower() == "flux" :
                    mean = self.__original_atributes["reactions"].loc[name, 'Flux (mmol/gDW/h)']
                    self.metabolites.df.loc[name, 'Flux (mmol/gDW/h)'] = value_rand(type_samp, standard_deviation, mean)
                
                elif type_machin.lower() == "enzyme" :
                    mean = self.__original_atributes["enzymes"].loc[name, 'Concentration / Activity']
                    self.metabolites.df.loc[name, 'Concentration / Activity'] = value_rand(type_samp, standard_deviation, mean)
                
            rho_sampled += self.rho()[1]

        rho_sampled /= (N+1)
    
        self.__upload_state()
        
        return(rho_sampled)





    #############################################################################
    ###################   function to remember a state   ########################
    def __save_state(self) :
        # Use of copy.deepcopy because dataframe are mutable = change also there renferencements
        import copy

        self.__original_atributes = {}

        self.__original_atributes["stoichiometry"]    = copy.deepcopy(self.Stoichio_matrix)
        self.__original_atributes["metabolites"]      = copy.deepcopy(self.metabolites.df)
        self.__original_atributes["reactions"]        = copy.deepcopy(self.reactions.df)
        self.__original_atributes["parameters"]       = copy.deepcopy(self.parameters.df)
        self.__original_atributes["elasticities_s"]   = copy.deepcopy(self.elasticity.s)
        self.__original_atributes["elasticities_p"]   = copy.deepcopy(self.elasticity.p)
        self.__original_atributes["enzymes"]          = copy.deepcopy(self.enzymes.df)

        print(self.__original_atributes["stoichiometry"].loc["D_Glucose", "PTS_RPTSsy"])

        

    #############################################################################
    ###################   function to remember a state   ########################
    def __upload_state(self) :
        self.metabolites.df  =   self.__original_atributes["metabolites"]
        self.reactions.df    =   self.__original_atributes["reactions"]
        self.parameters.df   =   self.__original_atributes["parameters"]
        self.elasticity.s    =   self.__original_atributes["elasticities_s"]
        self.elasticity.p    =   self.__original_atributes["elasticities_p"]
        self.enzymes.df      =   self.__original_atributes["enzymes"]
        self.Stoichio_matrix =   self.__original_atributes["stoichiometry"]

        print(self.__original_atributes["stoichiometry"].loc["D_Glucose", "PTS_RPTSsy"])



    #############################################################################
    ###################  test   #########################

    def changement_puis_rest(self) :
        self.__save_state()
        #print(len(self.Stoichio_matrix))
        print("\n \n")


        self.Stoichio_matrix.loc["D_Glucose", "PTS_RPTSsy"] = 123.0
        print(self.__original_atributes["stoichiometry"].loc["D_Glucose", "PTS_RPTSsy"])

        #print(len(self.Stoichio_matrix))
        print("\n \n")

        self.__upload_state()
        #print(len(self.Stoichio_matrix))


    #############################################################################
    ###################   Function to reset the model   #########################
    @property
    def reset(self) :
        self.__init__()


