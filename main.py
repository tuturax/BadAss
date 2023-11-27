#####################
# Library
#####################
import numpy as np
import pandas as pd
import sympy
import scipy

from layer_1.reactions import Reaction_class
from layer_1.metabolites import Metabolite_class
from layer_1.parameters import Parameter_class
from layer_1.elasticities import Elasticity_class
from layer_1.enzymes import Enzymes_class
from layer_1.regulation import Regulation_class


#####################
# Class MODEL
#####################
class MODEL:
    #############################################################################
    ########   Class method to creat a model from stochi matrix    ##############
    @classmethod
    def from_matrix(cls, matrix):
        class_instance = cls()
        return class_instance

    #############################################################################
    #############             Initialisation                #####################
    def __init__(self):
        # Call of reaction Class
        self.__reactions = Reaction_class(self)
        # Call of metabolite Class
        self.__metabolites = Metabolite_class(self)
        # Call of elasticity Class
        self.__elasticities = Elasticity_class(self)
        # Call of parameter Class
        self.__parameters = Parameter_class(self)
        # Call of enzyme Class
        self.__enzymes = Enzymes_class(self)
        # Call of regulation Class
        self.__regulations = Regulation_class(self)

        # Initialisation of the Matrix_Stoechio attribute
        self._Stoichio_matrix = pd.DataFrame()

        # Sampling file of the model
        self.data_sampling = pd.DataFrame(
            columns=["Name", "Type", "Mean", "Standard deviation", "Distribution"]
        )

        print(
            "Model created \n \nTo add metabolite, use .metabolites.add_meta \nTo add reaction,   use .reactions.add_reaction"
        )
        # Cache of the Network
        self.__cache_Link_matrix = None
        self.__cache_Reduced_Stoichio_matrix = None
        # Cache of the dynamic proprites
        self.__cache_Jacobian = None
        self.__cache_Reversed_Jacobian = None

        # Cache of MCA coefficients
        self.__cache_R_s_p = None
        self.__cache_R_v_p = None
        self.__cache_R_s_c = None
        self.__cache_R_v_p = None

    #################################################################################
    ######    Representation = the Dataframe of the Stoichiometric matrix     #######
    def __repr__(self) -> str:
        return str(self._Stoichio_matrix)

    #############################################################################
    ##################              Getter                  #####################
    @property
    def Stoichio_matrix(self):
        return self._Stoichio_matrix

    @property
    def __Stoichio_matrix(self):
        return self.Stoichio_matrix.to_numpy()

    @property
    def reactions(self):
        return self.__reactions

    @property
    def metabolites(self):
        return self.__metabolites

    @property
    def enzymes(self):
        return self.__enzymes

    @property
    def parameters(self):
        return self.__parameters

    @property
    def regulations(self):
        return self.__regulations

    @property
    def elasticity(self):
        return self.__elasticities

    @property
    def Link_matrix(self):
        # If the value isn't None, return the cache value
        if self.__cache_Link_matrix is not None and (
            self.__cache_Link_matrix.shape[0] != self.__Stoichio_matrix.shape[0]
            or self.__cache_Reduced_Stoichio_matrix.shape[1]
            != self.__Stoichio_matrix.shape[1]
        ):
            return (self.__cache_Link_matrix, self.__cache_Reduced_Stoichio_matrix)

        # Definition of a local stoichio matrix that will change gradually in the property
        N = self.Stoichio_matrix
        # .copy() ?

        # Else we take a look to the dependent row of the stoichio matrix
        dependent_rows = []
        independent_rows = []

        ext_meta_are_dependent = False

        # Case where we remove we considere the external metabolite has dependent !
        if ext_meta_are_dependent == True:
            # First, we can remove the external metabolite
            for meta in self.Stoichio_matrix.index:
                if self.metabolites.df.at[meta, "External"] == True:
                    # We add the external metabolite in the list of dependent row
                    dependent_rows.append(meta)
                    # And remove this metabolite from the local stoichio matrix N
                    N = N.drop(meta)

        _, independent = sympy.Matrix(N.to_numpy()).T.rref()

        for i, meta in enumerate(N.index):
            if i in independent:
                independent_rows.append(meta)
            else:
                dependent_rows.append(meta)

        # Build of the reduced stoichio matrix
        Nr = N.loc[independent_rows]

        # Then we deduce the link matrix L from Nr and N
        L = np.dot(self.Stoichio_matrix.to_numpy(), np.linalg.inv(Nr.to_numpy()))

        self.__cache_Link_matrix = L
        self.__cache_Reduced_Stoichio_matrix = Nr

        return (self.__cache_Link_matrix, self.__cache_Reduced_Stoichio_matrix)

    # The attibute with __ are the one compute with numpy and aims to be call for other compuation
    # The attribute without it are only the representation of the them on dataframe

    # MCA properties

    ###################
    # Jacobian
    @property  # Core
    def __Jacobian(self):
        if self.__cache_Jacobian is None:
            # Reset of the cache value of the inversed matrix of J
            self.__cache_Reversed_Jacobian = None
            # Compute the J matrix
            L, Nr = self.Link_matrix
            self.__cache_Jacobian = np.dot(
                Nr.to_numpy(), np.dot(self.elasticity.s.df.to_numpy(), L)
            )

        return self.__cache_Jacobian

    @property  # Displayed
    def Jacobian(self):
        Nr = self.Link_matrix[1]
        return pd.DataFrame(self.__Jacobian, index=Nr.index, columns=Nr.index)

    #########################
    # Inverse of the Jacobian
    @property  # Core
    def __Jacobian_reversed(self):
        if self.__cache_Reversed_Jacobian is None:
            # Reset of the cache value of the MCA coeff
            self.__cache_R_s_p = None
            self.__cache_R_v_p = None
            self.__cache_R_s_c = None
            self.__cache_R_v_p = None

            # Compute the J-1 matrix
            J_inv = np.linalg.pinv(self.__Jacobian)

            # Then we attribute to the cache value of the link matrix the new value
            self.__cache_Reversed_Jacobian = J_inv

        return self.__cache_Reversed_Jacobian

    @property  # Displayed
    def Jacobian_reversed(self):
        return pd.DataFrame(
            self.__Jacobian_reversed,
            index=self.Jacobian.columns,
            columns=self.Jacobian.index,
        )

    ##############################################
    # Response coefficient : MCA

    # R_s_p
    @property  # Core
    def __R_s_p(self):
        if self.__cache_R_s_p is None:
            C = -np.dot(
                np.dot(self.Link_matrix[0], self.__Jacobian_reversed),
                self.Link_matrix[1],
            )

            self.__cache_R_s_p = np.dot(C, self.elasticity.p.df.to_numpy())

        return self.__cache_R_s_p

    @property  # Displayed
    def R_s_p(self):
        return pd.DataFrame(
            self.__R_s_p,
            index=self.metabolites.df.index,
            columns=self.parameters.df.index,
        )

    # R_v_p
    @property  # Core
    def __R_v_p(self):
        if self.__cache_R_v_p is None:
            self.__cache_R_v_p = (
                np.dot(self.elasticity.s.df.to_numpy(), self.__R_s_p)
                + self.elasticity.p.df.to_numpy()
            )

        return self.__cache_R_v_p

    @property  # Displayed
    def R_v_p(self):
        return pd.DataFrame(
            self.__R_v_p,
            index=self.reactions.df.index,
            columns=self.parameters.df.index,
        )

    # R_s_c
    @property  # Core
    def __R_s_c(self):
        if self.__cache_R_s_c is None:
            self.__cache_R_s_c = -np.dot(
                self.Jacobian_reversed,
                np.dot(self.__Stoichio_matrix, self.elasticity.s.df.to_numpy()),
            ) + np.identity(len(self.__Stoichio_matrix))

        return self.__cache_R_s_c

    @property  # Displayed
    def R_s_c(self):
        return pd.DataFrame(
            self.__R_s_c,
            index=self.metabolites.df.index,
            columns=self.metabolites.df.index,
        )

    # R_v_c
    @property  # Core
    def __R_v_c(self):
        if self.__cache_R_v_c is None:
            self.__cache_R_v_c = np.dot(self.elasticity.s.df.to_numpy(), self.__R_s_c)
        return self.__cache_R_v_c

    @property  # Displayed
    def R_v_c(self):
        return pd.DataFrame(
            self.__R_v_c,
            index=self.reactions.df.index,
            columns=self.metabolites.df.index,
        )

    ##########################
    # Big matrix of response R
    @property  # Core
    def __R(self):
        R = np.block([[self.__R_s_p], [self.__R_v_p]])

        return R

    @property  # Displayed
    def R(self):
        return pd.DataFrame(
            self.__R,
            index=self.metabolites.df.index.to_list()
            + self.reactions.df.index.to_list(),
            columns=self.parameters.df.index,
        )

    #########################################
    # Standard deviation of parameters vector
    @property
    def __Standard_deviations(self):
        return self.parameters.df["Standard deviation"]

    ###################
    # Covariance matrix
    @property  # Core
    def __covariance(self):
        # If the cache is empty, we recompute the cov matrix and atribute the result to the cache value
        if self.__cache_cov is None:
            R = self.__R

            covariance_dp = np.identity(len(self.__Standard_deviations))

            for i, parameter in enumerate(self.parameters.df.index):
                covariance_dp[i][i] = (
                    self.parameters.df.at[parameter, "Standard deviation"] ** 2
                )

            matrix_RC = np.dot(R, covariance_dp)

            Cov = np.block(
                [
                    [covariance_dp, np.dot(covariance_dp, R.T)],
                    [matrix_RC, np.dot(matrix_RC, R.T)],
                ]
            )

            self.__cache_cov = Cov

        # Then we return the cache value
        return self.__cache_cov

    @property  # Displayed
    def covariance(self):
        # Return the dataframe of the covariance matrix by a call of it
        return pd.DataFrame(
            self.__covariance,
            index=(
                self.parameters.df.index.to_list()
                + self.metabolites.df.index.to_list()
                + self.reactions.df.index.to_list()
            ),
            columns=(
                self.parameters.df.index.to_list()
                + self.metabolites.df.index.to_list()
                + self.reactions.df.index.to_list()
            ),
        )

    ################
    # Entropy matrix
    @property  # Core
    def __entropy(self):
        if self.__cache_h is None:
            vec_h = []
            for index in self.covariance.index:
                vec_h.append(
                    0.5
                    * (
                        np.log(2 * np.pi * np.exp(1))
                        + np.log(self.covariance.at[index, index])
                    )
                )

            self.__cache_h = vec_h

        return self.__cache_h

    @property  # Displayed
    def entropy(self):
        return pd.DataFrame(
            self.__entropy, index=self.covariance.index, columns=["Entropy"]
        )

    ######################
    # Joint entropy matrix
    @property  # Core
    def __joint_entropy(self):
        if self.__cache_joint_h is None:
            Cov = self.__covariance
            joint_h = np.zeros(Cov.shape)
            for i in range(Cov.shape[0]):
                for j in range(Cov.shape[1]):
                    if Cov[i][i] * Cov[j][j] - Cov[i][j] * Cov[j][i] <= 0:
                        joint_h[i][j] = np.inf

                    else:
                        joint_h[i][j] = np.log(2 * np.pi * np.exp(1)) + 0.5 * np.log(
                            Cov[i][i] * Cov[j][j] - Cov[i][j] * Cov[j][i]
                        )

            self.__cache_joint_h = joint_h

        return self.__cache_joint_h

    @property  # Displayed
    def joint_entropy(self):
        return pd.DataFrame(
            self.__joint_entropy,
            index=self.covariance.index,
            columns=self.covariance.columns,
        )

    ###########################
    # Conditional entropy
    @property  # Core
    def __entropy_conditional(self):
        if self.__cache_conditional_h is None:
            condi_h = np.zeros(shape=self.covariance.shape)
            for i in range(condi_h.shape[0]):
                for j in range(condi_h.shape[1]):
                    condi_h[i][j] = self.__joint_entropy[j][i] - self.__entropy[j]
            self.__cache_conditional_h = condi_h

        return self.__cache_conditional_h

    @property  # Displayed
    def entropy_conditional(self):
        return pd.DataFrame(
            self.__entropy_conditional,
            index=self.covariance.index,
            columns=self.covariance.columns,
        )

    ###########################
    # Correlation
    @property  # Core
    def __corelation(self):
        if self.__cache_rho is None:
            rho = np.zeros(shape=self.covariance.shape)
            for i in range(rho.shape[0]):
                for j in range(rho.shape[1]):
                    rho[i][j] = self.__covariance[i][j] / (
                        (self.__covariance[i][i] * self.__covariance[j][j]) ** 0.5
                    )
            self.__cache_rho = rho

        return self.__cache_rho

    @property  # Displayed
    def corelation(self):
        return pd.DataFrame(
            self.__corelation,
            index=self.covariance.index,
            columns=self.covariance.columns,
        )

    ###########################
    # Mutual information matrix
    @property  # Core
    def __MI(self):
        if self.__cache_MI is None:
            MI = np.zeros(shape=self.covariance.shape)
            for i in range(MI.shape[0]):
                for j in range(MI.shape[1]):
                    MI[i][j] = -0.5 * np.log(1 - self.__corelation[i][j] ** 2)
            self.__cache_MI = MI

        return self.__cache_MI

    @property  # Displayed
    def MI(self):
        return pd.DataFrame(
            self.__MI,
            index=self.covariance.index,
            columns=self.covariance.columns,
        )

    #############################################################################
    #############   Function reset some the values of the model  ################
    def _reset_value(self, session=""):
        if session.lower() == "e_s":
            # Reset the value of the cache data
            self.__cache_Jacobian = None
            self.__cache_Reversed_Jacobian = None

        elif session.lower() == "e_p":
            # Reset the value of the cache data
            self.__cache_Jacobian = None
            self.__cache_Reversed_Jacobian = None
            # Reset of the cache value of the MCA coeff
            self.__cache_R_s_p = None
            self.__cache_R_v_p = None
            self.__cache_R_s_c = None
            self.__cache_R_v_p = None

        else:
            # Reset the value of the cache data
            self.__cache_Jacobian = None
            self.__cache_Reversed_Jacobian = None
            # Reset of the cache value of the MCA coeff
            self.__cache_R_s_p = None
            self.__cache_R_v_p = None
            self.__cache_R_s_c = None
            self.__cache_R_v_p = None

        self.__cache_Link_matrix = None
        self.__cache_Reduced_Stoichio_matrix = None
        self.__cache_cov = None
        self.__cache_h = None
        self.__cache_joint_h = None
        self.__cache_rho = None
        self.__cache_MI = None
        self.__cache_conditional_h = None

    #############################################################################
    #############   Function to update after a modification of N  ###############

    # Call the update function when the matrix_Stoichio is modified
    @Stoichio_matrix.setter
    def Stoichio_matrix(self, new_df):
        self._Stoichio_matrix = new_df
        self._update_network()

    def _update_network(self, session="Matrix") -> None:
        ### Description of the fonction
        """
        Fonction to update the dataframes after atribuated a new values to the stoichio matrix
        """

        if session == "Matrix":
            # Put the dataframe to 0
            self.metabolites.df.drop(self.metabolites.df.index, inplace=True)
            self.reactions.df.drop(self.reactions.df.index, inplace=True)

        # Deal with the reactions
        # Loop on every reaction of the stoichiometry matrix
        for reaction in self.Stoichio_matrix.columns:
            # Creation of a dictionnary that will contain every metabolite (as keys) and their stoichiometries coeff (as values)
            dict_stochio = {}

            # We also add the stochiometric coefficent to the dataframe of reaction
            for meta in self.Stoichio_matrix.index:
                if self.Stoichio_matrix.loc[meta][reaction] != 0:
                    dict_stochio[meta] = self.Stoichio_matrix.loc[meta, reaction]

            # Then we add the reaction to the reactions Dataframe
            self.reactions._update(name=reaction, metabolites=dict_stochio)

        # Deal with the metabolites

        for meta in self.Stoichio_matrix.index:
            self.metabolites._update(meta)

        # Reset the value of the cache data
        self.__cache_Link_matrix = None
        self.__cache_Reduced_Stoichio_matrix = None
        self.__cache_Jacobian = None

        # We update the elasticities matrix based on the new stoichiometric matrix
        self._update_elasticity()

    #################################################################################
    ############     Function to the elaticities matrix of the model     ############

    def _update_elasticity(self, session="E_s"):
        ### Description of the fonction
        """
        Function to update the elasticities matrices of the model after a direct modification of the stoichiometric matrix
        """
        # First we add the metabolite (=column) to the E_s datafrmae
        # We define a fixed variable n_reaction because elasticity.s.df depend of the sub_elasticity
        n_reaction = len(self.elasticity.s.df.index)

        if session.lower() == "e_s":
            for meta in self.metabolites.df.index:
                if meta not in self.elasticity.s.df.columns:
                    self.elasticity.s.df[meta] = [0 for i in range(n_reaction)]
            # Pandas doesn't allow to add line before at least 1 column is add
            if self.elasticity.s.df.columns.size != 0:
                for reaction in self.reactions.df.index:
                    if reaction not in self.elasticity.s.df.index:
                        self.elasticity.s.df.loc[reaction] = [
                            0 for i in self.elasticity.s.df.columns
                        ]

            colonnes = self.elasticity.s.df.columns
            index = self.elasticity.s.df.index
            self.elasticity.s.thermo = pd.DataFrame(0, columns=colonnes, index=index)
            self.elasticity.s.enzyme = pd.DataFrame(0, columns=colonnes, index=index)
            self.elasticity.s.regulation = pd.DataFrame(
                0, columns=colonnes, index=index
            )

        elif session.lower() == "e_p":
            for para in self.parameters.df.index:
                if para not in self.elasticity.p.df.columns:
                    self.elasticity.p.df[para] = [0 for i in self.elasticity.p.df.index]

            # Pandas doesn't allow to add line before at least 1 column is add
            if self.elasticity.p.df.columns.size != 0:
                for reaction in self.reactions.df.index:
                    if reaction not in self.elasticity.p.df.index:
                        self.elasticity.p.df.loc[reaction] = [
                            0 for i in self.elasticity.p.df.columns
                        ]

        self._reset_value()

    #################################################################################
    ############     Function that return the correlation coefficient    ############
    def rho(self, dtype=float):
        ### Description of the fonction
        """
        Fonction to compute the correlation coefficient
        """
        # Line to deal with the 1/0
        np.seterr(divide="ignore", invalid="ignore")

        Cov_df = self.covariance
        Cov = Cov_df.to_numpy()

        rho = np.zeros((Cov.shape[0], Cov.shape[1]), dtype=dtype)

        for i in range(Cov.shape[0]):
            for j in range(Cov.shape[1]):
                rho[i][j] = Cov[i][j] / ((Cov[i][i] * Cov[j][j]) ** 0.5)

        # Line to retablish the warning
        np.seterr(divide="warn", invalid="warn")

        return pd.DataFrame(rho, index=Cov_df.index, columns=Cov_df.columns)

    #################################################################################
    #########    Function that return the entropy of group of variable   ############
    def group_entropy(self, groups=[]):
        ### Description of the fonction
        """
        Fonction to compute the entropy of group of variable (joint entropy)

        groups : a list or a dictionnary contenning a list of string of the variables/parameter to regroup

        if groups = [] (by defalut) we take all variables/parameters indivudually
        """
        # Line to deal with the ./0 case
        np.seterr(divide="ignore", invalid="ignore")

        Cov_df = self.covariance
        Cov = Cov_df.to_numpy()

        # If the groups variables is empty, we return the entropy of every single variables and parameters
        if groups == []:
            return self.entropy

        # Else it means that we study a group of variable
        # If groups is a list, we transform it into a dictionnary
        elif type(groups) == list:
            dictionnary = {}
            for i, group in enumerate(groups):
                dictionnary[f"group_{i}"] = group

        elif type(groups) == dictionnary:
            dictionnary = groups

        # First we make sure that every variables in the list of list is well in the covarience matrix
        # For every group
        for key in dictionnary.keys():
            group = dictionnary[key]
            # for every variables of a group
            for variable in group:
                # if the variable is not in the model, we raise an error
                if variable not in Cov_df.index:
                    raise NameError(
                        f"The variables {variable} is not in the covariance matrix !"
                    )

        # Initialisation of the MI matrix
        entropy = pd.DataFrame(
            index=dictionnary.keys(), columns=["Entropy"], dtype=float
        )

        # For each group (= key of dictionnary)
        for key in dictionnary.keys():
            group = dictionnary[key]
            # We recreate a smaller covariance matrice with only the element of the group
            Cov = Cov_df.loc[group, group].to_numpy()

            entropy.at[key, "Entropy"] = (len(Cov) / 2) * np.log(
                2 * np.pi * np.e
            ) + 0.5 * np.log(np.linalg.det(Cov))

        # Line to retablish the warning
        np.seterr(divide="warn", invalid="warn")

        return entropy

    ###############################################################################################
    #######    Function that return the joint entropy matrix for a group of variable   ############
    def group_joint_entropy(self, groups=[]):
        ### Description of the fonction
        """
        Fonction to compute the joint entropy of a group of variable

        groups : a list or a dictionnary contenning a list of string of the variables/parameter to regroup

        if groups = [] (by defalut) we take all variables/parameters indivudually
        """
        # Line to deal with the ./0 case
        np.seterr(divide="ignore", invalid="ignore")

        Cov_df = self.covariance
        Cov = Cov_df.to_numpy()

        # If the groups variables is empty, we return the joint entropy matrix of every single variables and parameters
        if groups == []:
            return self.joint_entropy

        # Else it mean that we study a group of variable
        elif type(groups) == list:
            dictionnary = {}
            for i, group in enumerate(groups):
                dictionnary[str(i)] = group

        elif type(groups) == dictionnary:
            dictionnary = groups

        # First we make sure that every variables in the list of list is well in the covarience matrix
        for key in dictionnary.keys():
            group = dictionnary[key]
            for variable in group:
                if variable not in Cov_df.index:
                    raise NameError(
                        f"The variables {variable} is not in the covariance matrix !"
                    )

        # Initialisation of the MI matrix
        joint_entropy = pd.DataFrame(
            index=dictionnary.keys(), columns=dictionnary.keys(), dtype=float
        )

        for key1 in dictionnary.keys():
            for key2 in dictionnary.keys():
                # extraction of the list of string
                group1 = dictionnary[key1]
                group2 = dictionnary[key2]

                Cov = Cov_df.loc[group1 + group2, group1 + group2].to_numpy()

                joint_entropy.at[key1, key2] = (
                    len(group1) + len(group2) / 2
                ) ** np.log(2 * np.pi * np.e) + 0.5 * np.log(np.linalg.det(Cov))

        # Line to retablish the warning
        np.seterr(divide="warn", invalid="warn")

        return joint_entropy

    ###############################################################################################
    #######    Function that return the joint entropy matrix for a group of variable   ############
    def group_conditional_entropy(self, groups=[]):
        ### Description of the fonction
        """
        Fonction to compute the conditional entropy of a group of variable

        groups : a list or a dictionnary contenning a list of string of the variables/parameter to regroup

        if groups = [] (by defalut) we take all variables/parameters indivudually
        """
        # Line to deal with the ./0 case
        np.seterr(divide="ignore", invalid="ignore")

        Cov_df = self.covariance
        Cov = Cov_df.to_numpy()

        # If the groups variables is empty, we return the joint entropy matrix of every single variables and parameters
        if groups == []:
            return self.entropy_conditional

        # Else it mean that we study a group of variable
        elif type(groups) == list:
            dictionnary = {}
            for i, group in enumerate(groups):
                dictionnary[str(i)] = group

        elif type(groups) == dictionnary:
            dictionnary = groups

        # First we make sure that every variables in the list of list is well in the covarience matrix
        for key in dictionnary.keys():
            group = dictionnary[key]
            for variable in group:
                if variable not in Cov_df.index:
                    raise NameError(
                        f"The variables {variable} is not in the covariance matrix !"
                    )

        # Initialisation of the MI matrix
        conditional_entropy = pd.DataFrame(
            index=dictionnary.keys(), columns=dictionnary.keys(), dtype=float
        )

        for key1 in dictionnary.keys():
            for key2 in dictionnary.keys():
                # extraction of the list of string
                group1 = dictionnary[key1]
                group2 = dictionnary[key2]

                # Creating the sub_covariance matrix
                Cov1 = Cov_df.loc[group1, group1].to_numpy()
                Cov2 = Cov_df.loc[group2, group2].to_numpy()
                # And the big one
                Cov = Cov_df.loc[group1 + group2, group1 + group2].to_numpy()

                conditional_entropy.at[key1, key2] = (
                    len(Cov) / 2 * np.log(2 * np.pi * np.e)
                    + 0.5 * np.log(np.linalg.det(Cov))
                    - len(Cov1) / 2 * np.log(2 * np.pi * np.e)
                    + 0.5 * np.log(np.linalg.det(Cov1))
                )

                conditional_entropy.at[key2, key1] = (
                    len(Cov) / 2 * np.log(2 * np.pi * np.e)
                    + 0.5 * np.log(np.linalg.det(Cov))
                    - len(Cov2) / 2 * np.log(2 * np.pi * np.e)
                    + 0.5 * np.log(np.linalg.det(Cov2))
                )

        # Line to retablish the warning
        np.seterr(divide="warn", invalid="warn")

        return conditional_entropy

    #################################################################################
    ############    Function that return the Mutual Inforamtion matrix   ############
    def group_MI(self, groups=[]):
        ### Description of the fonction
        """
        Fonction to compute the Mutual information

        groups : a list or a dictionnary contenning a list of string of the variables/parameter to regroup

        if groups = [] (by defalut) we take all variables/parameters indivudually
        """
        # Line to deal with the ./0 case
        np.seterr(divide="ignore", invalid="ignore")

        Cov_df = self.covariance
        Cov = Cov_df.to_numpy()

        # If the groups variables is empty, we return the mutual information of every single variables and parameters
        if groups == []:
            return self.MI

        # Else it mean that we study a group of variable
        elif type(groups) == list:
            dictionnary = {}
            for i, group in enumerate(groups):
                dictionnary[str(i)] = group

        elif type(groups) == dictionnary:
            dictionnary = groups

        # First we make sure that every variables in the list of list is well in the covarience matrix
        for key in dictionnary.keys():
            group = dictionnary[key]
            for variable in group:
                if variable not in Cov_df.index:
                    raise NameError(
                        f"The variables {variable} is not in the covariance matrix !"
                    )

        # Initialisation of the MI matrix
        MI = pd.DataFrame(
            index=dictionnary.keys(), columns=dictionnary.keys(), dtype=float
        )

        for key1 in dictionnary.keys():
            for key2 in dictionnary.keys():
                # extraction of the list of string
                group1 = dictionnary[key1]
                group2 = dictionnary[key2]

                Cov_1 = Cov_df.loc[group1, group1].to_numpy()
                Cov_2 = Cov_df.loc[group2, group2].to_numpy()
                Cov_3 = Cov_df.loc[group1 + group2, group1 + group2].to_numpy()

                MI.loc[key1, key2] = 0.5 * np.log(
                    np.linalg.det(Cov_1) * np.linalg.det(Cov_2) / np.linalg.det(Cov_3)
                )

        # Line to retablish the warning
        np.seterr(divide="warn", invalid="warn")

        return MI

    #################################################################################
    ############    Function that return the Mutual Inforamtion matrix   ############
    def objective(self, variable1: str, variable2: str):
        ### Description of the fonction
        """
        Fonction to compute the Mutual information bteween to variable

        variable1, variable2 : string of the name of the variable that we want to compute the mutual information

        """
        Cov_df = self.covariance

        MI = (1 / (2 * np.log(2))) * np.log(
            Cov_df.at[variable1, variable1]
            * Cov_df.at[variable2, variable2]
            / (
                Cov_df.at[variable1, variable1] * Cov_df.at[variable2, variable2]
                - Cov_df.at[variable1, variable2] * Cov_df.at[variable2, variable1]
            )
        )
        return MI

    #############################################################################
    ###############  Function to creat a simple linear network ##################
    def creat_linear(self, n: int):
        ### Description of the fonction
        """
        Fonction to create a linear system of n metabolite

        n         : Number of metabolite in the linear network

        """
        if n <= 1:
            raise TypeError("Please enter an integer >= 2")

        else:
            # reinitialisation of the data
            self.__init__()

            matrix = np.array([[0 for i in range(n - 1)] for k in range(n)])

            for i in range(n):
                for j in range(n - 1):
                    if i == j:
                        matrix[i][j] = -1
                    elif i - 1 == j:
                        matrix[i][j] = 1

            noms_lignes = [f"meta_{i}" for i in range(n)]
            noms_colonnes = [f"reaction_{i}" for i in range(n - 1)]

            # Attribution of the new stoichiometic matrix
            self.Stoichio_matrix = pd.DataFrame(
                matrix, index=noms_lignes, columns=noms_colonnes
            )

            self.metabolites.df.loc[f"meta_{0}", "External"] = True
            self.metabolites.df.loc[f"meta_{n-1}", "External"] = True

    #############################################################################
    ##################   Function to read a CSV/XLS file  #######################
    def read_CSV(self, file="./Exemples/XLS/ecoli_core_model.xls"):
        ### Description of the fonction
        """
        Fonction read an Excel file

        file     : string the specify the directory of the Excel file

        """
        self.parameters.df.reset_index(inplace=True)
        self.enzymes.df.reset_index(inplace=True)
        self.elasticity.s.df.reset_index(inplace=True)
        self.elasticity.p.df.reset_index(inplace=True)

        df = pd.read_excel(file)
        N = df.drop(df.columns[0], axis=1)
        N = N.drop(N.index)

        for ligne in df.to_numpy():
            N.loc[ligne[0]] = ligne[1:]

        self._Stoichio_matrix = N

        self._update_network

        for meta in self.metabolites.df.index:
            if meta[-3:] == "(e)":
                self.metabolites.df.loc[meta]["External"] = True

    #############################################################################
    ###################   Function to read a SBML file  #########################
    def read_SBML(
        self,
        directory="./Exemples/SBML/",
        file_SBML="E_coli_CCM.xml",
        reference_state_metabolites="reference_state_metabolites.tsv",
        reference_state_c="reference_state_c.tsv",
        reference_state_reactions="reference_state_reactions.tsv",
        reference_state_v="reference_state_v.tsv",
        reference_state_keq="reference_state_keq.tsv",
    ):
        ### Description of the fonction
        """
        Fonction read a SBML file

        file     : string the specify the directory of the SBML file

        """
        import libsbml

        # Reset of the model
        self.reset

        reader = libsbml.SBMLReader()
        document = reader.readSBML(directory + file_SBML)

        n_error = document.getNumErrors()
        if n_error != 0:
            raise ValueError(
                f"There is {n_error} in your SBML file, please fix it before to use this function"
            )

        else:
            print(f"0 error detected in your SBML file")
            model = document.getModel()

            N = pd.DataFrame(dtype=float)

            for reaction in model.reactions:
                N[reaction.getName()] = pd.Series([0] * len(N.index), dtype="float64")

                reactants = reaction.getListOfReactants()
                for reactant in reactants:
                    specie = model.getSpecies(reactant.getSpecies())
                    stoichio = reactant.getStoichiometry()

                    if specie.getName() not in N.index:
                        N.loc[specie.getName()] = pd.Series(
                            [0] * len(N.columns), index=N.columns, dtype="float64"
                        )

                    N.loc[specie.getName(), reaction.getName()] = stoichio

                products = reaction.getListOfProducts()
                for product in products:
                    specie = model.getSpecies(product.getSpecies())
                    stoichio = product.getStoichiometry()

                    if specie.getName() not in N.index:
                        N.loc[specie.getName()] = pd.Series(
                            [0] * len(N.columns), index=N.columns, dtype="float64"
                        )

                    N.loc[specie.getName(), reaction.getName()] = stoichio

                list_species = []
                for specie in model.species:
                    list_species.append(specie.getName())
                for specie in list_species:
                    if specie not in N.index:
                        N.loc[specie] = pd.Series(
                            [0] * len(N.columns), index=N.columns, dtype="float64"
                        )

                N.fillna(0, inplace=True)

            self.Stoichio_matrix = N

            # Set the metabolite as external
            for specie in model.species:
                if specie.boundary_condition:
                    self.metabolites.df.at[specie.name, "External"] = True

        # now we read the reference state
        def tsv_to_list(file_tsv: str):
            # Work only if the file is a single colomn
            file_tsv = open(file_tsv)
            list_meta_tsv = file_tsv.readlines()
            file_tsv.close()
            list_meta_tsv = [element.rstrip("\n") for element in list_meta_tsv]
            return list_meta_tsv

        import os

        # Reading the metabolites list
        if os.path.exists(directory + reference_state_metabolites):
            list_metabolites = tsv_to_list(directory + reference_state_metabolites)
            # Reading the reference states of the concentrations of the metabolites
            if os.path.exists(directory + reference_state_c):
                list_concentrations = tsv_to_list(directory + reference_state_c)
                for i in range(len(list_metabolites)):
                    if list_metabolites[i] in self.metabolites.df.index:
                        # Attribution of the concentrations to the dataframe
                        self.metabolites.df.at[
                            list_metabolites[i], "Concentration (mmol/gDW)"
                        ] = float(list_concentrations[i])
                    else:
                        print(
                            f"Warning : The metabolite {list_metabolites[i]} is not in the SBML file of the metabolic network !"
                        )
        # Reading the reactions list
        if os.path.exists(directory + reference_state_reactions):
            list_reactions = tsv_to_list(directory + reference_state_reactions)
            # Reading the reference state of the reaction
            if os.path.exists(directory + reference_state_v):
                list_flux = tsv_to_list(directory + reference_state_v)
                for i in range(len(list_reactions)):
                    if list_reactions[i] in self.reactions.df.index:
                        # Attribution of the flux to the dataframe
                        self.reactions.df.at[
                            list_reactions[i], "Flux (mmol/gDW/h)"
                        ] = float(list_flux[i])
                    else:
                        print(
                            f"Warning : The reaction {list_reactions[i]} is not in the SBML file of the metabolic network !"
                        )

            # reading the referecne state of the equilibrium constant
            if os.path.exists(directory + reference_state_keq):
                list_keq = tsv_to_list(directory + reference_state_keq)
                for i in range(len(list_reactions)):
                    if list_reactions[i] in self.reactions.df.index:
                        # Attributon of the keq to the dataframe
                        self.reactions.df.at[
                            list_reactions[i], "Equilibrium constant"
                        ] = float(list_keq[i])

    #############################################################################
    ###################   Function to check the model   #########################
    @property
    def check(self):
        # Check the reaction
        unused_reactions = []
        for react in self._Stoichio_matrix.columns.to_list():
            counter = 0
            for meta in self._Stoichio_matrix.index.to_list():
                counter += np.abs(self._Stoichio_matrix.loc[meta, react])
            if counter == 0:
                unused_reactions.append(react)

        # Check the metabolite
        unused_metabolites = []
        for meta in self._Stoichio_matrix.index.to_list():
            counter = 0
            for react in self._Stoichio_matrix.columns.to_list():
                counter += np.abs(self._Stoichio_matrix.loc[meta, react])
            if counter == 0:
                unused_metabolites.append(meta)

        print("The following reactions are unused : \n")
        for unused_react in unused_reactions:
            print(f"-{unused_react} \n")

        print("\n \n")
        print("The following metabolites are unused : \n")
        for unused_meta in unused_metabolites:
            print(f"-{unused_meta} \n")

        return (unused_reactions, unused_metabolites)

    #############################################################################
    ###################   Function to check the model   #########################
    @property
    def check_unstable(self):
        eigen_values = np.linalg.eigvals(self.Jacobian.to_numpy())

        positif = False
        for value in eigen_values:
            if np.real(value) > 0:
                positif = True

        if positif == True:
            print(
                "The jacobian matrix have positive eigen values, that could lead to an unstable state"
            )
        return eigen_values

    #############################################################################
    ###################   Function plot the MI matrix   #########################
    def plot(
        self, result="MI", title="", label=False, value_in_cell=False, index_to_keep=[]
    ):
        import matplotlib
        import matplotlib.pyplot as plt

        # Get the dataframe of the result
        result = result.lower()

        if result == "mi" or result == "mutual information":
            data_frame = self.group_MI()
        elif result == "rho" or result == "correlation":
            data_frame = self.rho()
        elif result == "cov" or result == "covariance":
            data_frame = self.covariance

        # Look the index to keep for the plot of the matrix
        index_to_keep_bis = []
        # If nothing is specified, we keep everything
        # else
        if index_to_keep != []:
            # We take a look at every index that the user enter
            for index in index_to_keep:
                # If one of them is not in the model, we told him
                if index not in data_frame.index:
                    raise NameError(f"- {index} is not in the correlation matrix")
                # else, we keep in memory the index that are in the model
                else:
                    index_to_keep_bis.append(index)

        else:
            index_to_keep_bis = data_frame.index

        # Then we create a new matrix with only the index specified
        data_frame = data_frame.loc[index_to_keep_bis, index_to_keep_bis]
        matrix = data_frame.to_numpy()

        data_frame

        fig, ax = plt.subplots()

        if result == "mi":
            custom_map = matplotlib.colors.LinearSegmentedColormap.from_list(
                "custom", ["white", "blue"]
            )

            im = plt.imshow(matrix, cmap=custom_map, norm=matplotlib.colors.LogNorm())

        elif result == "rho":
            custom_map = matplotlib.colors.LinearSegmentedColormap.from_list(
                "custom", ["red", "white", "blue"]
            )

            im = plt.imshow(matrix, cmap=custom_map, vmin=-1, vmax=1)

        elif result == "cov":
            custom_map = matplotlib.colors.LinearSegmentedColormap.from_list(
                "custom", ["white", "blue"]
            )

            im = plt.imshow(matrix, cmap=custom_map)

        # Display the label next to the axis
        if label == True:
            ax.set_xticks(np.arange(len(data_frame.index)), labels=data_frame.index)
            ax.set_yticks(np.arange(len(data_frame.index)), labels=data_frame.index)

            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

        # Display the value of each cell
        if value_in_cell == True:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    text = ax.text(
                        j,
                        i,
                        round(matrix[i, j], 2),
                        ha="center",
                        va="center",
                        color="black",
                    )

        if title == "":
            if result.lower() == "mi":
                title = "Mutual information"
            elif result.lower() == "rho":
                title = "Correlation"

        # Title of the plot
        ax.set_title(title)
        fig.tight_layout()

        # Plot of the black line to separate the parameters from the variables
        # Width of the line
        line_width = 1
        # Number of parameters
        N_para = self.parameters.df.shape[0]
        # Position of the line
        x_p_e = [-0.5, N_para - 0.5]
        y_p_e = [N_para - 0.5, N_para - 0.5]
        plt.plot(x_p_e, y_p_e, "black", linewidth=line_width)
        plt.plot(y_p_e, x_p_e, "black", linewidth=line_width)

        x_p = [-0.5, N_para - 0.5]
        y_p = [N_para - 0.5, N_para - 0.5]
        plt.plot(x_p, y_p, "black", linewidth=line_width)
        plt.plot(y_p, x_p, "black", linewidth=line_width)

        plt.colorbar()
        plt.show()

    #############################################################################
    ###################   Function add sampling data    #########################
    def add_sampling_data(
        self, name, type_variable: str, mean=True, SD=1, distribution="uniform"
    ):
        ### Description of the fonction
        """
        Fonction add a new data to the data_sampling dataframe

        name           : string name of the variable to sample, list of 2 string in the case of elasticity
        type_variable  : string that make reference to the type of the variable to sample
        mean           : Mean of the variable, if mean = True, mean take the current value of the variable
        SD             : float of the standard deviation of the random draw of the variable
        distribution   : string that make reference to the type of distribution of the random draw of this variable
        """

        # Case where the elasticity p is sampled
        if type_variable == "elasticity_p":
            # If the name is not a list in the case of the elasticity, it's bad
            if type(name) != list:
                raise TypeError(
                    "For the elasticity, be sure to use a list of 2 string, the first for the flux name and the second for the differtial of the elasticity"
                )

            # If the list have more or less than 2 elements, it's not valide
            elif len(name) != 2:
                raise TypeError(
                    "For the elasticity, be sure to use a list of 2 string, the first for the flux name and the second for the differtial of the elasticity"
                )

            else:
                # We attribute both elements of the list, the first must be the flux name and second the differential name's
                flux, differential = name
                # We check if the flux is in the model
                if flux not in self.elasticity.p.df.index:
                    raise NameError(
                        f'The flux name "{flux}" is not in the elasticity matrix'
                    )

                elif differential not in self.elasticity.p.df.columns:
                    raise NameError(
                        f'The differential name "{differential}" is not in the elasticity matrices E_p'
                    )

                if type(mean) == bool:
                    mean = self.elasticity.p.df.at[flux, differential]

        # Case where the elasticity s is sampled
        elif type_variable == "elasticity_s":
            # If the name is not a list in the case of the elasticity, it's bad
            if type(name) != list:
                raise TypeError(
                    "For the elasticity, be sure to use a list of 2 string, the first for the flux name and the second for the differtial of the elasticity"
                )

            # If the list have more or less than 2 elements, it's not valide
            elif len(name) != 2:
                raise TypeError(
                    "For the elasticity, be sure to use a list of 2 string, the first for the flux name and the second for the differtial of the elasticity"
                )

            else:
                # We attribute both elements of the list, the first must be the flux name and second the differential name's
                flux, differential = name
                # We check if the flux is in the model
                if flux not in self.elasticity.s.df.index:
                    raise NameError(
                        f'The flux name "{flux}" is not in the elasticity matrix'
                    )

                elif differential not in self.elasticity.s.columns:
                    raise NameError(
                        f'The differential name "{differential}" is not in the elasticity matrices E_s'
                    )

                if type(mean) == bool:
                    mean = self.elasticity.s.df.at[flux, differential]

        # Case where a parameter is sampled
        elif type_variable == "parameter":
            if name not in self.parameters.df.index:
                raise NameError(
                    f'The parameter name "{name}" is not in the parameters dataframe'
                )

            if type(mean) == bool:
                mean = self.parameters.at[name, "Mean values"]

        # Case where the metabolite concentration is sampled
        elif type_variable == "metabolite" or type_variable == "concentration":
            if name not in self.metabolites.df.index:
                raise NameError(
                    f'The metabolite name "{name}" is not in the metabolites dataframe'
                )

            if type(mean) == bool:
                mean = self.metabolites.at[name, "Concentration (mmol/gDW)"]

        # Case where the flux is sampled
        elif type_variable == "flux" or type_variable == "reaction":
            if name not in self.reactions.df.index:
                raise NameError(
                    f'The flux name "{name}" is not in the reactions dataframe'
                )

            if type(mean) == bool:
                mean = self.reactions.at[name, "Flux (mmol/gDW/h)"]

        # Case where a enzyme concentration/activity is sampled
        elif type_variable == "enzyme":
            if name not in self.enzymes.df.index:
                raise NameError(
                    f'The enzyme name "{name}" is not in the enzymes dataframe'
                )
            if type(mean) == bool:
                mean = self.reactions.at[name, "Concentration / Activity"]

        else:
            raise NameError(
                f'The type "{type_variable}" is not available \n\nThe type of variable allowed are :\n- elasticity_p\n- elasticity_s\n- parameter\n- metabolite or concentration\n- reaction or flux\n- enzyme'
            )

        # Let's check if the name of the distribution are correct
        distribution_allowed = ["uniform", "normal", "lognormal", "beta"]
        if distribution.lower() not in distribution_allowed:
            raise NameError(
                f'The name of the distribution "{distribution}" is not handle by the programme !\n\nHere is the distribution allowed :\n- uniform\n- normal\n- lognormal\n- beta'
            )

        index = 0
        while index in self.data_sampling.index:
            index += 1

        self.data_sampling.loc[index] = [name, type_variable, mean, SD, distribution]

    #############################################################################
    ###################   Function sampled the model    #########################
    def sampling(self, N: int, result="MI", seed_constant=1):
        ### Description of the fonction
        """
        Fonction launch a sampling study, it return the mean value of the matrix

        N              : Number of random draw done for each variable of the .data_sampling dataframe
        result         : matrix returned by the code
        seed_constant  : float that is the seed of our radom draw
        """

        # If the number of sample asked if < 1 = bad
        if N < 1:
            raise ValueError("The number of sample must be greater or egual to 1 !")

        # Internal function that define the random draw
        def value_rand(type_samp: str, mean: float, SD: float):
            if type_samp.lower() == "uniform":
                deviation = (9 * SD) ** 0.25
                return np.random.uniform(mean - deviation, mean + deviation)

            elif type_samp.lower() == "normal":
                return np.random.normal(mean, SD)

            elif type_samp.lower() == "lognormal":
                return np.random.lognormal(mean, SD)

            elif type_samp.lower() == "beta":
                alpha = (((1 - mean) / ((np.sqrt(SD)) * (2 - mean) ** 2)) - 1) / (
                    2 - mean
                )
                beta = alpha * (1 - mean)
                return np.random.beta(alpha, beta)

        # We call of a dataframe in order to initialise the variable with the good shape and get the name of the indexs and columns
        if result == "MI":
            data_frame = self.MI()
        else:
            data_frame = self.rho()

        matrix_sampled = data_frame.to_numpy()

        # Conditional line to deal with the seed of the random values generator
        # If seed_constant is an int, than we use this int as seed to generate the seed of other random value
        if type(seed_constant) == int:
            np.random.seed(seed_constant)
            seed = np.random.randint(0, 2**32, N)
        else:
            # Else it is tottaly random
            seed = np.random.randint(0, 2**32, N)

        # We save the original value of the model
        self.__save_state()

        # Time Counter
        import time

        start = time.time()

        for i in range(N):
            # Seed of the generation of random value
            np.random.seed(seed[i])
            seed_2 = np.random.randint(
                low=0, high=2**32, size=self.data_sampling.shape[0]
            )

            for i, index in enumerate(self.data_sampling.index):
                # Change of the seed of the radom value generator
                np.random.seed(seed_2[i])

                if self.data_sampling.at[index, "Type"].lower() == "elasticity_p":
                    flux, differential = self.data_sampling.at[index, "Name"]
                    self.elasticity.p.df.at[flux, differential] = value_rand(
                        self.data_sampling.at[index, "Distribution"],
                        self.data_sampling.at[index, "Standard deviation"],
                        self.data_sampling.at[index, "Mean"],
                    )

                elif self.data_sampling.at[index, "Type"].lower() == "elasticity_s":
                    flux, differential = self.data_sampling.at[index, "Name"]
                    self.elasticity.s.df.at[flux, differential] = value_rand(
                        self.data_sampling.at[index, "Distribution"],
                        self.data_sampling.at[index, "Standard deviation"],
                        self.data_sampling.at[index, "Mean"],
                    )

                elif self.data_sampling.at[index, "Type"].lower() == "parameter":
                    self.parameters.df.at[
                        self.data_sampling.at[index, "Name"], "Mean values"
                    ] = value_rand(
                        self.data_sampling.at[index, "Distribution"],
                        self.data_sampling.at[index, "Standard deviation"],
                        self.data_sampling.at[index, "Mean"],
                    )

                elif self.data_sampling.at[index, "Type"].lower() == "metabolite":
                    self.metabolites.df.at[
                        self.data_sampling.at[index, "Name"], "Concentration (mmol/gDW)"
                    ] = value_rand(
                        self.data_sampling.at[index, "Distribution"],
                        self.data_sampling.at[index, "Standard deviation"],
                        self.data_sampling.at[index, "Mean"],
                    )

                elif self.data_sampling.at[index, "Type"].lower() == "flux":
                    self.metabolites.df.at[
                        self.data_sampling.at[index, "Name"], "Flux (mmol/gDW/h)"
                    ] = value_rand(
                        self.data_sampling.at[index, "Distribution"],
                        self.data_sampling.at[index, "Standard deviation"],
                        self.data_sampling.at[index, "Mean"],
                    )

                elif self.data_sampling.at[index, "Type"].lower() == "enzyme":
                    self.metabolites.df.at[
                        self.data_sampling.at[index, "Name"], "Concentration / Activity"
                    ] = value_rand(
                        self.data_sampling.at[index, "Distribution"],
                        self.data_sampling.at[index, "Standard deviation"],
                        self.data_sampling.at[index, "Mean"],
                    )

            if result == "MI":
                matrix_sampled += self.MI().to_numpy()
            else:
                matrix_sampled += self.rho().to_numpy()

        matrix_sampled /= N + 1

        self.__upload_state()

        running_time = time.time() - start
        print(
            f"running time of the code : {running_time} \nSo {running_time/N} per occurences !"
        )
        return matrix_sampled

    #############################################################################
    ###################   function to save a state   ########################
    def __save_state(self):
        # Use of copy.deepcopy because dataframe are mutable = change also there renferencements
        import copy

        self.__original_atributes = {}

        self.__original_atributes["stoichiometry"] = copy.deepcopy(self.Stoichio_matrix)
        self.__original_atributes["metabolites"] = copy.deepcopy(self.metabolites.df)
        self.__original_atributes["reactions"] = copy.deepcopy(self.reactions.df)
        self.__original_atributes["parameters"] = copy.deepcopy(self.parameters.df)
        self.__original_atributes["elasticities_s"] = copy.deepcopy(
            self.elasticity.s._df
        )
        self.__original_atributes["elasticities_p"] = copy.deepcopy(
            self.elasticity.p._df
        )
        self.__original_atributes["enzymes"] = copy.deepcopy(self.enzymes.df)

    #############################################################################
    ################   function to upload the saved state   #####################
    def __upload_state(self):
        self.Stoichio_matrix = self.__original_atributes["stoichiometry"]
        self.metabolites.df = self.__original_atributes["metabolites"]
        self.reactions.df = self.__original_atributes["reactions"]
        self.parameters.df = self.__original_atributes["parameters"]
        self.elasticity.s._df = self.__original_atributes["elasticities_s"]
        self.elasticity.p._df = self.__original_atributes["elasticities_p"]
        self.enzymes.df = self.__original_atributes["enzymes"]

    #############################################################################
    ###################   Function to reset the model   #########################
    @property
    def reset(self):
        self.__init__()
