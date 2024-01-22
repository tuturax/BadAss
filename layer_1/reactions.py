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
    def __init__(self, class_MODEL_instance):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

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
            raise NameError('The reaction "' + name + '" is already in the model !')

        # Else, the reaction is add to the model by an add to the DataFrame
        else:
            # Add the reaction to the reactions dataframe
            self.df.loc[name] = [metabolites, k_eq, law, flux]

            # Add a null columns to the stoichio matrix N
            if name not in self.__class_MODEL_instance.Stoichio_matrix.columns:
                self.__class_MODEL_instance.Stoichio_matrix[name] = 0.0

            for meta in list(metabolites.keys()):
                if meta not in self.__class_MODEL_instance.Stoichio_matrix.index:
                    # If the metabolite is not in the model, we add it
                    self.__class_MODEL_instance.metabolites.add(meta)

                # Then we add the correct stoichiometric coefficients
                self.__class_MODEL_instance.Stoichio_matrix.at[meta, name] = self.df.at[
                    name, "Metabolites"
                ][meta]

            # Updating the network
            self.__class_MODEL_instance._update_network()
            self.__class_MODEL_instance._update_elasticity()

    #################################################################################
    #########           Fonction to change a reaction                      ##########

    def change(self, name: str, metabolites=None, k_eq=None, law="", flux=None):
        ### Description of the fonction
        """
        Fonction to change a reaction properties in the model

        name           : Name of the reaction to change
        metabolites    : Dictionnary of the metabolites used in this reaction
        k_eq           : Float for the equilibrium constant of the reaction
        law            : Strig to specify the type of law of th ereactino
        flux           : Float for th evalue of the lux

        """

        if name not in self.df.index:
            raise NameError(f"The name '{name}' is not in the reactions dataframe")

        else:
            if metabolites != None:
                self.df.at[name, "Metabolites"] = metabolites
            if k_eq != None:
                self.df.at[name, "Equilibrium constant"] = k_eq
            if law != None:
                self.df.at[name, "Law"] = law
            if flux != None:
                self.df.at[name, "Flux (mmol/gDW/h)"] = flux

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
            for reaction in self.__class_MODEL_instance.Stoichio_matrix.columns:
                # If the the reaction is not in the modified reaction dataframe => it was deleted
                if reaction not in self.df.index:
                    self.__class_MODEL_instance.Stoichio_matrix.drop(
                        reaction, axis=1, inplace=True
                    )

            # Updating the network
            self.__class_MODEL_instance._update_network

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
        if not isinstance(self.df, pd.DataFrame):
            self.df = pd.DataFrame(
                columns=[
                    "Metabolites",
                    "Equilibrium constant",
                    "Law",
                    "Flux (mmol/gDW/h)",
                ]
            )

        # Look if the reaction is already in the model
        if name not in self.df.index:
            self.df.loc[name] = [metabolites, k_eq, law, flux]

            # We check the stoichiometric coefficient link to this reaction in order to automatically add them to the matrix
            for meta in list(self.df.loc[name, "Metabolites"].keys()):
                if meta not in self.__class_MODEL_instance.Stoichio_matrix.index:
                    # If the metabolite is not in the model, we add it
                    self.__class_MODEL_instance.metabolites._update(meta)
