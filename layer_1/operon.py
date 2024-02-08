#####################
# Library
#####################
import pandas as pd


#####################
# Class Enzymes
#####################
class Operon_class:
    #############################################################################
    ###############             Initialisation              #####################
    def __init__(self, class_MODEL_instance):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self.df = pd.DataFrame(columns=["Enzymes linked", "Mixed covariance"])

    #################################################################################
    ###########           Return the Dataframe of the operons           #############
    def __repr__(self) -> str:
        return str(self.df)

    #################################################################################
    #########           Fonction to return the number of operon            ##########
    @property
    def len(self):
        return len(self.df.index)

    #################################################################################
    #########                  Fonction to add an operon                   ##########
    def add(self, name="", enzymes_linked=[], mixed_covariance = 0.0) -> None:
        ### Description of the fonction
        """
        Fonction to add an operon to the model

        name                : name of the operon to add to the dataframe
        enzymes_linked      : list of enzymes linked to this operon
        """

        # Look if the operon is already in the model
        if name in self.df.index.to_list():
            raise NameError(
                f"The input operon {name} is already in the operon dataframe !\n"
            )

        # We look for every enzyme linked to this operon
        for enzyme in enzymes_linked:
            # If the enzyme is in the enzyme's dataframe
            if enzyme not in self.__class_MODEL_instance.enzyme.df.index:
                raise NameError(
                    f"The input enzyme {enzyme} is not in the enzyme dataframe !\n"
                )
            # And if the enzyme is already link to an other operon
            for operon in self.df.index:
                if enzyme in self.df.at[operon, "Enzymes linked"]:
                    raise NameError(
                        f"The input enzyme {enzyme} is already linked to the operon {operon} ! \n"
                    )

        # If everything is ok, we add the operon and its linked enzyme
        self.df.loc[name] = [enzymes_linked, mixed_covariance]

    #################################################################################
    #########      Fonction to link enzyme to existing opero               ##########
    def add_enzymes_to_operon(self, name="", enzymes_to_add=[]) -> None:
        ### Description of the fonction
        """
        Fonction to add enzymes to an existong operon of the model

        name                : name of the operon
        enzymes_to_add      : list of enzymes to link to this operon
        """

        # Look if the operon is already in the model
        if name not in self.df.index.to_list():
            raise NameError(
                f"The input operon {name} is not in the operon dataframe !\n"
            )

        # We look for every enzyme to add
        for enzyme in enzymes_to_add:
            # If the enzyme is in the enzyme's dataframe
            if enzyme not in self.__class_MODEL_instance.enzyme.df.index:
                raise NameError(
                    f"The input enzyme {enzyme} is not in the enzyme dataframe !\n"
                )

            # And if the enzyme is already linked to an other operon
            for operon in self.df.index:
                if enzyme in self.df.at[operon, "Enzymes linked"]:
                    raise NameError(
                        f"The input enzyme {enzyme} is already linked to the operon {operon} ! \n"
                    )

        # Finally, we create a list the is the sum of the old list and the new one of enzyme to add
        # And we attribute this list to the right operon in the dataframe
        self.df.at[name, "Enzymes linked"] = (
            self.df.at[name, "Enzymes linked"] + enzymes_to_add
        )

    #################################################################################
    #########           Fonction to remove an operon                       ##########

    def remove(self, name: str) -> None:
        ### Description of the fonction
        """
        Fonction to remove an operon from the dataframe

        name        : Name of the operon to remove a enzyme
        """

        # Look if the operon is in the model
        if name not in self.df.index:
            raise NameError(
                f"The operon {name} is not in the operon dataframe, please enter a valide name \n"
            )

        else:
            # Else, the oeron is remove from the dataframe
            self.df.drop(name, inplace=True)

    #################################################################################
    #########           Fonction to remove an operon                       ##########

    def remove_enzyme_from_operon(self, name: str, enzymes_to_remove=[]) -> None:
        ### Description of the fonction
        """
        Fonction to remove enzyme from an operon

        name        : Name of the operon to remove a enzyme
        enzymes     : list of enzyme to remove from the operon line
        """

        # Look if the operon is in the model
        if name not in self.df.index:
            raise NameError(
                f"The operon {name} is not in the operon dataframe, please enter a valide name \n"
            )

        # We keep in memeory the list of enzyme linked to this operon
        new_list = list(self.df.at[name, "Enzymes linked"])
        # If an enzyme that we want to remove is in this list, we remove this enzyme
        for enzyme in enzymes_to_remove:
            if enzyme in new_list:
                new_list.remove(enzyme)

        # Finaly, we attribute the the list of enzyme without the removed enzyme to the operon
        self.df.at[name, "Enzymes linked"] = new_list
