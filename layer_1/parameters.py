#####################
# Library
#####################
import pandas as pd


#####################
# Class Parameters
#####################
class Parameter_class:
    #############################################################################
    #############             Initialisation                #####################
    def __init__(self, class_model_instance):
        # Private attribute for the instance of the Main class
        self.__class_model_instance = class_model_instance

        self.df = pd.DataFrame(
            index=["Temperature"], columns=["Mean values", "Standard deviation"]
        )
        self.df.loc["Temperature", "Mean values"] = 273.15
        self.df.loc["Temperature", "Standard deviation"] = 1.0

    #################################################################################
    #########           Return the Dataframe of the parameters             ##########
    def __repr__(self) -> str:
        return str(self.df)

    #################################################################################
    #########        Fonction to return the number of parameters           ##########
    @property
    def len(self):
        return len(self.df)

    #################################################################################
    #########           Fonction to add a parameters                       ##########
    def add(self, name: str, mean=1, Standard_deviation=1.0) -> None:
        ### Description of the fonction
        """
        Fonction to add a parameter to the model

        name                : Name of the parameter
        mean                : Mean value of the parameter
        Standard_deviation  : Standard deviation of the parameter

        """

        # Look if the parameter is already in the model
        if name in self.df.index.to_list():
            raise NameError('The parameter "' + name + '" is already in the model !')

        # Else, the parameter is add to the model by an add to the DataFrame
        else:
            self.df.loc[name] = [mean, Standard_deviation]

    ##################################################################################
    #########           Fonction to remove a parameter                      ##########

    def remove(self, name: str) -> None:
        ### Description of the fonction
        """
        Fonction to remove a parameter to the model

        name        : Name of the parameter to remove
        """

        # Look if the parameter is in the model
        if name not in self.df.index.to_list():
            raise NameError("Please enter a name of a parameter of the model \n")

        else:
            # Else, the parameter is remove from the dataframe
            self.df.drop(name, inplace=True)

            print(f"Name of the removed parameter : {name}")

            # Removing this parameter from the elasticity matrix E_p
            if name in self.__class_model_instance.elasticity.p.columns:
                self.__class_model_instance.elasticity.p.drop(
                    name, axis=1, inplace=True
                )
            self.__class_model_instance._update_elasticity()

    ##################################################################################
    #########         Fonction to add all enzyme to the model               ##########
    def add_enzymes(self) -> None:
        ### Description of the fonction
        """
        Fonction to consider all enzymes as parameters
        """

        # For every enzymes of the models
        for enzyme in self.__class_model_instance.enzymes.df.index:
            # if this one is not already considered as a parameter
            if enzyme + "_para" not in self.df.index:
                # We add it in the parameter
                self.add(
                    enzyme + "_para",
                    mean=self.__class_model_instance.enzymes.df.loc[
                        enzyme, "Concentration / Activity"
                    ],
                )
                # We add a new column of 0 to the parameters elasticity dataframe
                self.__class_model_instance.elasticity.p[enzyme + "_para"] = [
                    0.0
                    for i in range(self.__class_model_instance.elasticity.p.shape[0])
                ]

                # We add 1 to the enzyme linked to reaction
                for reaction in self.__class_model_instance.enzymes.df.loc[
                    enzyme, "Reactions linked"
                ]:
                    self.__class_model_instance.elasticity.p.loc[
                        reaction, enzyme + "_para"
                    ] = 1.0

    ##################################################################################
    #########         Fonction to add all external metabolite               ##########
    def add_externals(self) -> None:
        ### Description of the fonction
        """
        Fonction to consider all external metabolite as parameters
        """

        # For every metabolite of the model
        for meta in self.__class_model_instance.metabolites.df.index:
            # If this one is external
            if self.__class_model_instance.metabolites.df.loc[meta, "External"] == True:
                # If this one is not already in the parameter dataframe
                if (meta + "_para") not in self.df.index:
                    # We add it to the parameter dataframe
                    self.add(meta + "_para")
                    # And add a column to the parameter elasticity matrix
                    self.__class_model_instance.elasticity.p[meta + "_para"] = [
                        0
                        for i in range(
                            self.__class_model_instance.elasticity.p.shape[0]
                        )
                    ]

                    for reaction in self.__class_model_instance.Stoichio_matrix.columns:
                        if (
                            self.__class_model_instance.Stoichio_matrix.loc[
                                meta, reaction
                            ]
                            != 0
                        ):
                            self.__class_model_instance.elasticity.p.loc[
                                reaction, meta + "_para"
                            ] = (
                                -0.5
                                * self.__class_model_instance.Stoichio_matrix.loc[
                                    meta, reaction
                                ]
                            )
