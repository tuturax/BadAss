#####################
# Library
#####################
import pandas as pd


####################
# Class Reactions
#####################
class Reaction_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_model_instance):
        # Private attribute for the instance of the Main class
        self.__class_model_instance = class_model_instance

        self.df = pd.DataFrame(
            columns=["Metabolites", "Equilibrium constant", "Law", "Flux (mmol/gDW/h)"]
        )

    ################################################################################
    #########           Return the Dataframe of the reactions             ##########
    def __repr__(self) -> str:
        return str(self.df)

    ################################################################################
    #########        Fonction to return the number of reaction            ##########
    @property
    def len(self):
        return len(self.df)

    #################################################################################
    #########           Fonction to add a reaction                         ##########
    def add(self, name: str, metabolites={}, k_eq=1.0, law="", flux=1.0) -> None:
        ### Description of the fonction
        """
        Fonction to add a reaction to the model

        name        : Name of the reaction.
        metabolites : Dictionnary that take as keys the names of the metabolites (str) and as value the stoichiometric coefficient (float)

        k_eq        : Equilibre constant of the reaction
        law         : Reaction law
        flux        : Flux of the reaction at the reference state

        """
        # Look if the reaction class was well intialised
        if type(self.df) != type(pd.DataFrame()):
            self.df = pd.DataFrame(
                columns=[
                    "Metabolites",
                    "Equilibrium constant",
                    "Law",
                    "Flux (mmol/gDW/h)",
                ]
            )

        # Look if the reaction is already in the model
        if name in self.df.index:
            raise TypeError('The reaction "' + name + '" is already in the model !')

        # Else, the reaction is add to the model by an add to the DataFrame
        else:
            self.df.loc[name] = [metabolites, k_eq, law, flux]

            for reaction in self.df.index:
                # If the the reaction is not in the orginal Stoichiometry matrix => it was add
                if reaction not in self.__class_model_instance.Stoichio_matrix.columns:
                    # We add a colomn of 0
                    self.__class_model_instance.Stoichio_matrix[reaction] = [
                        0
                        for i in range(
                            self.__class_model_instance.Stoichio_matrix.shape[0]
                        )
                    ]

                    # We check the stoichiometric coefficient link to this reaction in order to automatically add them to the matrix
                    for meta in list(self.df.loc[reaction, "Metabolites"].keys()):
                        if (
                            meta
                            not in self.__class_model_instance.Stoichio_matrix.index
                        ):
                            # If the metabolite is not in the model, we add it
                            self.__class_model_instance.metabolites.add(meta)

                        # Then we add the correct stoichiometric coefficients
                        self.__class_model_instance.Stoichio_matrix.loc[
                            meta, reaction
                        ] = self.df.loc[reaction, "Metabolites"][meta]

            # Updating the network
            # self.__class_model_instance._update_network(session="reaction")
            self.__class_model_instance._update_elasticity()

    #################################################################################
    #########           Fonction to remove a reaction                      ##########

    def remove(self, name: str) -> None:
        ### Description of the fonction
        """
        Fonction to remove a reaction to the model

        name        : Name of the reaction to remove
        """

        # Look if the reaction isn't in the model
        if name not in self.df.index:
            print("Please enter a valide name \n")

        # Else, the reaction is remove from the model
        else:
            self.df.drop(name, inplace=True)

            # For a reaction in the stoichiometric matrix
            for reaction in self.__class_model_instance.Stoichio_matrix.columns:
                # If the the reaction is not in the modified reaction dataframe => it was deleted
                if reaction not in self.df.index:
                    self.__class_model_instance.Stoichio_matrix.drop(
                        reaction, axis=1, inplace=True
                    )

            # Updating the network
            self.__class_model_instance._update_network

    #################################################################################
    #########           Fonction to add a reaction                         ##########
    def _update(self, name: str, metabolites={}, k_eq=1.0, law="", flux=1) -> None:
        ### Description of the fonction
        """
        Internal function to update the reaction dataframe after a change of the stoichiometric matrix

        name        : Name of the reaction.
        metabolites : Dictionnary that take as keys the names of the metabolites (str) and as value the stoichiometric coefficient (float)

        k_eq        : Equilibre constant of the reaction
        law         : Reaction law

        """
        # Look if the reaction class was well intialised
        if type(self.df) != type(pd.DataFrame()):
            self.df = pd.DataFrame(
                columns=[
                    "Metabolites",
                    "Equilibrium constant",
                    "Law",
                    "Flux (mmol/gDW/h)",
                ]
            )

        # Look if the reaction is already in the model
        if name in self.df.index:
            True
            # raise NameError("The reaction \""+ name +"\" is already in the model !")

        # Else, the reaction is add to the model by an add to the DataFrame
        else:
            self.df.loc[name] = [metabolites, k_eq, law, flux]

            # We check the stoichiometric coefficient link to this reaction in order to automatically add them to the matrix
            for meta in list(self.df.loc[name, "Metabolites"].keys()):
                if meta not in self.__class_model_instance.Stoichio_matrix.index:
                    # If the metabolite is not in the model, we add it
                    self.__class_model_instance.metabolites._update(meta)
