# Viscoelastic-Viscoplastic damage model

In this work we take a classical approach to solving the equations governing quasi-static finite-strain compressible elasticity, with code based on ```step-44``` and the code ```Quasi_static_Finite_strain_Compressible_Elasticity``` from the code gallery. The formulation adopted here is an updated lagrangian formulation and run the model for cyclic loading-unloading conditions at 0 and 10% nanoparticle weight fraction and dry/saturated state.

The stress response of a nanoparticle/ epoxy system is decomposed into an equilibrium part and two viscous parts to capture 
the nonlinear and rate-depended behavior of the material. We introduce the nanoparticle dependency through an amplification factor, 
which is a function of fillers’ weight fraction.

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
our constitutive model. The schematic structure of the model is presented in the picture below.

![This is an image](/rheo.PNG)

**Note: Functions "lstm_forward" are used to incorporate the pretrained Deep-Learning model. Use the options "set Switch to ML = Off" and "set At which timestep switch to ML ? = 2147483646" in the parameter file to exclude the Deep-Learning model and continue with the constitutive model.**

## Run the code
Compile it using the following commands and run in release mode
```
cmake -DDEAL_II_DIR=/path/to/deal.II .
make release
make
```
you can run it by typing
```
./vevp_ml_model
```
**Note: Create an "output" folder in the running directory before running.**

The parameter file is already in the current directory, and has the following options:
```
subsection Assembly method
  # The automatic differentiation order to be used in the assembly of the linear system.
  # Order = 0: Both the residual and linearisation are computed manually.
  # Order = 1: The residual is computed manually but the linearisation is performed using AD.
  # Order = 2: Both the residual and linearisation are computed using AD. 
  set Automatic differentiation order = 0
end

subsection Boundary conditions
  # Positive stretch in mm applied length-ways to the model (Driver = Dirichlet)
  set First stretch = 0.5
  set Second stretch = 0.7
  set Third stretch = 0.9
  set Fourth stretch = 1.1
  set Fifth stretch = 1.3
  set Sixth stretch = 1.5
  set Seventh stretch = 1.65

  # Load type: cyclic or none
  set Load type = cyclic_to_zero
  
  # Total cycles (is set to 7 here)
  set Total cycles = 7
  
end

subsection Finite element system
  # Displacement system polynomial order
  set Polynomial degree = 1

  # Gauss quadrature order
  set Quadrature order  = 2

  # Switch to ML ?
  set Switch to ML = Off

  # Hidden Size of LSTM
  set LSTM = 150

  # Alpha for perturbation
  set alpha = 1e-4
end

subsection Linear solver
  # Linear solver iterations (multiples of the system matrix size)
  set Max iteration multiplier  = 1.0

  # Linear solver residual (scaled by residual norm)
  set Residual                  = 1e-3

  # Preconditioner type
  set Preconditioner type        = ssor

  # Preconditioner relaxation value
  set Preconditioner relaxation  = 0.9

  # Type of solver used to solve the linear system
  set Solver type               = Direct
end


subsection Material properties
  # mu1
  set mu1 = 760.0
  
  # mu2
  set mu2 = 790.0

  # m
  set m = 0.657
  
  # gamma_dot_0
  set gamma_dot_0 = 9.7746e11
  
  # dG
  set dG = 1.9761e-19
  
  # Ad
  set Ad = 320.00

  # tau0
  set tau0 = 85.0
  
  # d0s (not needed)
  set d0s = 0.1
  
  # m_tau (not needed)
  set m_tau = 2

  # a
  set a = 0.179
  
  # b
  set b = 0.910

  # sigma0
  set sigma0 = 5.5

  # nu1
  set nu1 = 0.23

  # nu2
  set nu2 = 0.0

  # de (not needed actually)
  set de = 1e-4

  # Temperature
  set temp = 296.0
  
  # Water content
  set zita = 0.0
  
  # NP weight fraction 
  set wnp = 0.0

  # y0 for tau
  set y0 = 75

  # x0 for tau
  set x0 = 0.2369

  # parameter a in tauHat
  set a_t = -48.23

  # parameter b in tauHat
  set b_t = 0.06786
end


subsection Nonlinear solver
  # Number of Newton-Raphson iterations allowed
  set Max iterations Newton-Raphson = 100

  # Displacement error tolerance
  set Tolerance displacement        = 1.0e-3

  # Force residual tolerance
  set Tolerance force               = 1.0e-3
end


subsection Time
  # End time
  set End time       = 1451

  # Time step size (this is actually updated and we controll delta u to keep it constant instead of delta t)
  set Time step size = 0.5

  set Time step size 2 = 1.0

  # At which timestep change to ML (set 0 if only ML) (max integer is 2147483647)
  set At which timestep switch to ML ? = 2147483646

  #Delta u
  set Delta de     = 5e-4

  #Load rate
  set load_rate     = 0.0165

  # Change timestep at this timestep
  set reduce_at_timestep = 2147483646

  # Set how much to reduce timestep 
  set reduce_timestep = 0.1
end

subsection Output parameters
  # Output data every given time step number for Paraview output files
  set Time step number output = 1
  
  # For "solution" file output data associated with integration point values averaged on:  elements | nodes
  set Averaged results           = nodes
end

```

## Results

These results were produced with the material parameters as presented in the parameters file and fine mesh for 0% and 10% Boehmite nanoparticles at saturated and dry state.
![This is an image](/fem4.png)

