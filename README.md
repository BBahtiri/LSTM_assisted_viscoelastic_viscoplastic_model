# viscoelastic_viscoplastic_model

3d viscoelastic viscoplastic model depending on moisture content and nanoparticle volume fraction

The stress response of a nanoparticle/ epoxy system is decomposed into an equilibrium part and two viscous parts to capture 
the nonlinear and rate-depended behavior of the material. We introduce the nanoparticle dependency through an amplification factor, 
which is a function of fillers’ volume fraction.

Our model incorporates experimental characteristics by using a decomposition
of the material behavior into a viscoelastic and a viscoplastic part,
corresponding to the time-dependent response and to the irreversible molecular
chain sliding, respectively. We further decompose the viscoelastic stress
response into a hyperelastic network, capturing the equilibrium of the viscoelastic
response and a viscous network, capturing the rate-dependent nonequilibrium
response. The time-independent hyperelastic part of the stressstrain
behavior contains an elastic spring, which is associated to the entropy
change due to deformations while the time-dependent viscous network is
composed of an elastic spring and a viscous dashpot capturing the rate- and
temperature dependent behavior of the material at hand. Additionally, the
quasi-irreversible sliding of the molecular chains results in stress softening
in the material known as the Mullins effect, which is implemented within
our constitutive model.

Compile it using the following commands
```
cmake -DDEAL_II_DIR=/path/to/deal.II .
```
