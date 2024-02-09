from main import *

momo = MODEL()

momo.creat_linear(4)

momo.enzymes.add_to_all_reaction()
momo.parameters.add_enzymes()

momo.parameters.add_externals()

momo.elasticity.s.half_satured()

momo.graphic_interface(result="rho", title="Guess the elasticity", label = True)
