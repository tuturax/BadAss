#####################
# Library
#####################
import pandas as pd
import numpy as np


#####################
# Class Regulation
#####################
class Regulation_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_MODEL_instance):
        # Private attribute for the instance of the Main class
        self.__class_MODEL_instance = class_MODEL_instance

        self.df = pd.DataFrame(
            columns=[
                "Regulated flux",
                "Regulator",
                "Coefficient of regulation",
                "Type regulation",
            ]
        )

    #################################################################################
    #########           Return the Dataframe of the            ##########
    def __repr__(self) -> str:
        return str(self.df)

    #################################################################################
    #########        Fonction to return the shape of the matrix            ##########
    @property
    def len(self):
        return len(self.df.shape)

    #################################################################################
    #########           Fonction to add a regulation                       ##########
    def add(
        self, name: str, regulated: str, regulator: str, coefficient=1, allosteric=True
    ):
        ### Description of the fonction
        """
        Fonction to add a regulation to the model

        regulated       : Name of regulated flux
        regulator       : Name of the metabolite that regulate
        coefficient     : Foat for the coefficient of regulation, coef > 0 => activation, coef < 0 => inihibition
        allosteric      : Boolean to specify the type of reaction, True => allosteric, False => transcriptional

        """
        if allosteric == True:
            type_regulation = "allosteric"
        else:
            type_regulation = "transcriptional"

        # Look if the metabolite class was well intialised
        if type(self.df) != type(pd.DataFrame()):
            self.df = pd.DataFrame(
                columns=[
                    "Regulated flux",
                    "Regulator",
                    "Coefficient of regulation",
                    "Type regulation",
                ]
            )

        # Look if the regulation is already in the regulation dataframe
        elif name in self.df.index:
            raise NameError(
                f"The name of the regulation '{name}' is already in the regulation dataframe !"
            )

        # Look if the regulated flux is in the model
        elif regulated not in self.__class_MODEL_instance.reactions.df.index:
            raise NameError(
                f'The reaction "{regulated}" is not in the reaction dataframe !'
            )

        #  Look if the regulator metabolite is in the model
        elif regulator not in self.__class_MODEL_instance.metabolites.df.index:
            raise NameError(
                f'The metabolite "{regulator}" is not in the metabolite dataframe !'
            )
        elif regulator not in self.__class_MODEL_instance.N_without_ext.index:
            raise NameError(
                f'The metabolite "{regulator}" is in the metabolite dataframe but is external !'
            )

        # Else it's allright :D
        self.df.loc[name] = [regulated, regulator, coefficient, type_regulation]

        if allosteric == True:
            self.__class_MODEL_instance.elasticity.s.df.at[
                regulated, regulator
            ] += coefficient

        else:
            # name of the enzyme linked to this regulation
            enzyme = "enzyme_" + name

            # We concidere now this enzyme as a metabolite
            self.__class_MODEL_instance.metabolites.add(name=enzyme)
            self.__class_MODEL_instance.reactions.add(
                name="creation_" + name, metabolites={enzyme: 1}
            )
            self.__class_MODEL_instance.reactions.add(
                name="destruction_" + name, metabolites={enzyme: -1}
            )
            self.__class_MODEL_instance.elasticity.s.df.at[
                "creation_" + name, regulator
            ] += coefficient

        self.__class_MODEL_instance._update_elasticity()

    #################################################################################
    #########           Fonction to change a regulation coefficient        ##########
    def change_coeff(self, name_regu: str, new_coeff: float) -> None:
        ### Description of the fonction
        """
        Fonction to change the coefficient of a regulation effect

        name_regu       : String for the name of regulation effect to change
        new_coeff       : Folat for the new value of the regulation coefficient

        """
        # Check if the regulation name is in the dataframe
        if name_regu not in self.df.index:
            raise NameError(
                f"The regulation name '{name_regu}' is not in the regulation dataframe"
            )

        else:
            regulated = self.df.at[name_regu, "Regulated flux"]
            regulator = self.df.at[name_regu, "Regulator"]
            # if it is an allosteric/direct regulation
            if self.df.at[name_regu, "Type regulation"] == "allosteric":
                # Soustraction of the old value and addition of the new one on the E_s matrix
                self.__class_MODEL_instance.elasticity.s.df.at[
                    regulated, regulator
                ] += (new_coeff - self.df.at[name_regu, "Coefficient of regulation"])

            # Case of a transcriptional/undirect regulation
            else:
                # Soustraction of the old value and addition of the new one on the E_s matrix
                self.__class_MODEL_instance.elasticity.s.df.at[
                    "creation_" + name_regu, regulator
                ] += (new_coeff - self.df.at[name_regu, "Coefficient of regulation"])

            # Attribution of the new value to the coeff
            self.df.at[name_regu, "Coefficient of regulation"] = new_coeff
