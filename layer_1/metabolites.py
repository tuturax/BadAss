#####################
# Library
#####################
import pandas as pd


#####################
# Class Metabolites
#####################
class Metabolite_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        # Private list to deal with the fact that a dataframe cannot be filled if there is no collumn in the dataframe
        self.__cache_meta = []

        self.df = pd.DataFrame(columns=["External", "Concentration (mmol/gDW)"])

    #################################################################################
    #########           Return the Dataframe of the metabolites            ##########
    def __repr__(self) -> str:
        return str(self.df)

    #################################################################################
    #########        Fonction to return the number of metabolites          ##########
    @property
    def len(self):
        return len(self.df)

    #################################################################################
    #########           Fonction to add a metabolite                         ##########
    def add(self, name: str, external=False, concentration=1.0):
        ### Description of the fonction
        """
        Fonction to add a matabolite to the model

        name          : Name of the metabolite

        external      : Boolean to say if the metabolite is external
        concentration : Concentration of the metabolite

        """
        # Look if the metabolite class was well intialised
        if type(self.df) != type(pd.DataFrame()):
            self.df = pd.DataFrame(columns=["External", "Concentration (mmol/gDW)"])

        # Look if the metabolite is already in the model
        elif name in self.df.index:
            print('The metabolite "' + name + '" is already in the model !')

        # Else, the metabolite is add to the model by an add to the DataFrame
        else:
            self.df.loc[name] = [external, concentration]

            # If there is no reaction in the columns of the soichio metric matrix, we keep in memeory the metabolite
            if self.__class_MODEL_instance.Stoichio_matrix.columns.size == 0:
                self.__cache_meta.append(name)
                print(
                    "Don't worry, the metabolite will be add after the add of the 1st reaction"
                )

            # Else, we add every metabolite that we keeped into memory to the stoichiometrix matrix
            else:
                self.__cache_meta.append(name)
                for meta in self.__cache_meta:
                    if meta not in self.__class_MODEL_instance.Stoichio_matrix.index:
                        self.__class_MODEL_instance.Stoichio_matrix.loc[meta] = 0.0

                self.__cache_meta = []

            # Updating the network
            # self.__class_MODEL_instance._update_network()
            self.__class_MODEL_instance._update_elasticity()

    #################################################################################
    #########           Fonction to change a metabolite                    ##########
    def change(self, name: str, external=None, concentration=None):
        ### Description of the fonction
        """
        Fonction to change a metabolite properties in the model

        name           : Name of the metabolite to change
        external       : Bool to specifies if the metabolite is external
        concentration  : Float for the concentration of the metabolites

        /!\ If you use external metabolite as parameters
        """
        if name not in self.df.index:
            raise NameError(f"The name '{name}' is not in the metabolite dataframe")

        else:
            if external != None:
                if type(external) != bool:
                    raise TypeError(
                        f"The input variable '{external}' is a type '{type(external)}', a boolean True/False is expected for the argument 'External' !"
                    )
                else:
                    # If it was internal and become external => we remove it from the elasticity matrix
                    self.df.at[name, "External"] = external
                    self.__class_MODEL_instance._update_elasticity()

            if concentration != None:
                if isinstance(concentration, (int, float, complex)):
                    if concentration < 0:
                        raise ValueError(
                            f"The value of the input '{concentration}' must be greater than 0 !"
                        )
                    else:
                        self.df.at[name, "Concentration (mmol/gDW)"] = concentration

                else:
                    raise TypeError(
                        f"The input variable '{concentration}' is a type '{type(concentration)}', a value is expected for the argument 'concentration' !"
                    )

    #################################################################################
    #########           Fonction to remove a metabolite                    ##########

    def remove(self, name: str):
        ### Description of the fonction
        """
        Fonction to remove a metabolite to the model

        name        : Name of the metabolite to remove
        """

        # Look if the metabolite is in the model
        if name not in self.df.index:
            raise NameError(
                f"Please enter a valide name, {name} isn't in the model ! \n"
            )

        else:
            # Else, the metabolite is remove from the dataframe
            self.df.drop(name, inplace=True)

            for meta in self.__class_MODEL_instance.Stoichio_matrix.index:
                # If the the meta is not in the modified metabolite dataframe => it was deleted
                if meta not in self.df.index:
                    self.__class_MODEL_instance.Stoichio_matrix.drop(
                        meta, axis=0, inplace=True
                    )

            # And from every mention of it in the reaction dataframe
            for reaction in self.__class_MODEL_instance.reactions.df.index:
                keys_to_remove = [
                    cle
                    for cle in self.__class_MODEL_instance.reactions.df.loc[
                        reaction, "Metabolites"
                    ].keys()
                    if name in cle
                ]
                for key_to_remove in keys_to_remove:
                    self.__class_MODEL_instance.reactions.df.loc[
                        reaction, "Metabolites"
                    ].pop(key_to_remove)

            # Updating the network
            self.__class_MODEL_instance._update_network

            # Remove this metabolite from the elasticity matrix E_s
            self.__class_MODEL_instance._update_elasticity()

    #################################################################################
    #########           Fonction to update the meta dataframe              ##########
    def _update(self, name=None, external=False, concentration=1):
        ### Description of the fonction
        """
        Internal function to update the metabolite dataframe after a change of the stoichiometric matrix

        name          : Name of the metabolite

        external      : Boolean to say if the metabolite is external
        concentration : Concentration of the metabolite

        """
        # Look if the metabolite class was well intialised
        if type(self.df) != type(pd.DataFrame()):
            self.df = pd.DataFrame(columns=["External", "Concentration (mmol/gDW)"])

        # Look if the metabolite is already in the model
        if name not in self.df.index:
            self.df.loc[name] = [external, concentration]
