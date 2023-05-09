/* ---------------------------------------------------------------------
* Author: Betim Bahtiri, Leibniz Universit\"at Hannover, 2023"
*
*
* The deal.II library is free software; you can use it, redistribute
* it, and/or modify it under the terms of the GNU Lesser General
* Public License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
* The full text of the license can be found in the file LICENSE at
* the top level of the deal.II distribution.
 */


// We start by including all the necessary deal.II header files and some C++
// related ones. They have been discussed in detail in previous tutorial
// programs, so you need only refer to past tutorials for details.
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/quadrature_point_data.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/base/config.h>
#if DEAL_II_VERSION_MAJOR >= 9 && defined(DEAL_II_WITH_TRILINOS)
#include <deal.II/differentiation/ad.h>
#define ENABLE_SACADO_FORMULATION
#endif
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/transformations.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cmath>
#include <vector>
#include <cstdio>
#include </bigwork/nhgebaht/FEM/Eigen3/Eigen/Dense>

namespace vevpd_model
{
  using namespace dealii;
  using Eigen::MatrixXd;
  using Eigen::VectorXd;
  using Eigen::ArrayXXd;
  using Eigen::ArrayXd;

// There are several parameters that can be set in the code so we set up a
// ParameterHandler object to read in the choices at run-time.
  namespace Parameters
  {
// @sect4{Assembly method}

// Here we specify whether automatic differentiation is to be used to assemble
// the linear system, and if so then what order of differentiation is to be
// employed.
    struct AssemblyMethod
    {
      unsigned int automatic_differentiation_order;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    void AssemblyMethod::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Assembly method");
      {
        prm.declare_entry("Automatic differentiation order", "0",
                          Patterns::Integer(0,2),
                          "The automatic differentiation order to be used in the assembly of the linear system.\n"
                          "# Order = 0: Both the residual and linearisation are computed manually.\n"
                          "# Order = 1: The residual is computed manually but the linearisation is performed using AD.\n"
                          "# Order = 2: Both the residual and linearisation are computed using AD.");
      }
      prm.leave_subsection();
    }

    void AssemblyMethod::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Assembly method");
      {
        automatic_differentiation_order = prm.get_integer("Automatic differentiation order");
      }
      prm.leave_subsection();
    }

    struct BoundaryConditions
    {

      double stretch1;
      double stretch2;
      double stretch3;
      double stretch4;
      double stretch5;
      double stretch6;
      double stretch7;

      double unload_time_first;
      double unload_time_second;
      double unload_time_third;
      double unload_time_fourth;
      double unload_time_fifth;
      double unload_time_sixth;   
      std::string load_type;
      int total_cycles;



      static void
      declare_parameters(ParameterHandler &prm);
      void
      parse_parameters(ParameterHandler &prm);
    };


    void BoundaryConditions::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary conditions");
      {
        prm.declare_entry("First stretch", "1.04",
                          Patterns::Double(0.0),
                          "Positive stretch applied length-ways to the model");

        prm.declare_entry("Second stretch", "1.04",
                          Patterns::Double(0.0),
                          "Positive stretch applied length-ways to the model");

        prm.declare_entry("Third stretch", "1.04",
                          Patterns::Double(0.0),
                          "Positive stretch applied length-ways to the model");                   

        prm.declare_entry("Fourth stretch", "1.04",
                          Patterns::Double(0.0),
                          "Positive stretch applied length-ways to the model");

        prm.declare_entry("Fifth stretch", "1.04",
                          Patterns::Double(0.0),
                          "Positive stretch applied length-ways to the model");

        prm.declare_entry("Sixth stretch", "1.04",
                          Patterns::Double(0.0),
                          "Positive stretch applied length-ways to the model");

        prm.declare_entry("Seventh stretch", "1.04",
                          Patterns::Double(0.0),
                          "Positive stretch applied length-ways to the model");
  
        prm.declare_entry("Load type", "cyclic_to_zero",
                          Patterns::Selection("none|cyclic_to_zero"),
                          "Type of loading");                                                  

        prm.declare_entry("Total cycles", "1.04",
                          Patterns::Double(0.0),
                          "Total number of cycles");
      }
      prm.leave_subsection();
    }

     void BoundaryConditions::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary conditions");
      {
        stretch1 = prm.get_double("First stretch");
        stretch2 = prm.get_double("Second stretch");     
        stretch3 = prm.get_double("Third stretch");
        stretch4 = prm.get_double("Fourth stretch");
        stretch5 = prm.get_double("Fifth stretch");
        stretch6 = prm.get_double("Sixth stretch");
        stretch7 = prm.get_double("Seventh stretch");


        load_type = prm.get("Load type");
        total_cycles = prm.get_integer("Total cycles"); 
      }
      prm.leave_subsection();
    }   

// Here we specify the polynomial order used to approximate the solution.
// The quadrature order should be adjusted accordingly.
    struct FESystem
    {
      unsigned int poly_degree;
      unsigned int quad_order;
      std::string switchML;
      int hidden_size;
      double alpha;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree", "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");

        prm.declare_entry("Quadrature order", "3",
                          Patterns::Integer(0),
                          "Gauss quadrature order");

        prm.declare_entry("Switch to ML", "On",
                          Patterns::Selection("On|Off"),
                          "Switch to ML");  

        prm.declare_entry("LSTM", "2",
                          Patterns::Integer(0),
                          "LSTM"); 

        prm.declare_entry("alpha", "1e-4",
                          Patterns::Double(0.0),
                          "alpha");                                             
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order = prm.get_integer("Quadrature order");
        switchML = prm.get("Switch to ML");
        hidden_size = prm.get_integer("LSTM");
        alpha = prm.get_double("alpha");
      }
      prm.leave_subsection();
    }


// @sect4{Materials}

// We also need the shear modulus $ \mu $ and Poisson ration $ \nu $ for the
// neo-Hookean material.
    struct Materials
    {
      double mu1;
      double mu2;
      double m;
      double gamma_dot_0;
      double dG;
      double Ad;
      double tau0;
      double d0s;
      double m_tau;
      double a;
      double b;
      double sigma0;
      double nu1;
      double nu2;
      double de;
      double temp;
      double zita;
      double wnp;
      double y0;
      double x0;
      double a_t;
      double b_t;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Materials::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        prm.declare_entry("mu1", "800.0",
                          Patterns::Double(),
                          "mu1");

        prm.declare_entry("mu2", "401.2",
                          Patterns::Double(),
                          "mu2");

        prm.declare_entry("m", "0.692",
                          Patterns::Double(),
                          "m");

        prm.declare_entry("gamma_dot_0", "3.403e-2",
                          Patterns::Double(),
                          "gamma_dot_0");                          

        prm.declare_entry("dG", "3.783e-25",
                          Patterns::Double(),
                          "dG");      

        prm.declare_entry("Ad", "455.30",
                          Patterns::Double(),
                          "Ad"); 

        prm.declare_entry("tau0", "27.11",
                          Patterns::Double(),
                          "tau0");  

        prm.declare_entry("d0s", "0.851",
                          Patterns::Double(),
                          "d0s");     

        prm.declare_entry("m_tau", "7.96",
                          Patterns::Double(),
                          "m_tau");  

        prm.declare_entry("a", "0.996",
                          Patterns::Double(),
                          "a");   

        prm.declare_entry("b", "0.234",
                          Patterns::Double(),
                          "b"); 

        prm.declare_entry("sigma0", "20.2",
                          Patterns::Double(),
                          "sigma0"); 

        prm.declare_entry("nu1", "0.4318",
                          Patterns::Double(-1.0,0.5),
                          "nu1");  


        prm.declare_entry("nu2", "0.4318",
                          Patterns::Double(-1.0,0.5),
                          "nu2");  

        prm.declare_entry("de", "1e-4",
                          Patterns::Double(),
                          "de");

        prm.declare_entry("temp", "296.0",
                          Patterns::Double(),
                          "temp"); 

        prm.declare_entry("zita", "0.0",
                          Patterns::Double(),
                          "zita");                          

        prm.declare_entry("wnp", "0.0",
                          Patterns::Double(),
                          "wnp");   

        prm.declare_entry("y0", "0.0",
                          Patterns::Double(),
                          "wnp");  

        prm.declare_entry("x0", "0.0",
                          Patterns::Double(),
                          "wnp");   

        prm.declare_entry("a_t", "0.0",
                          Patterns::Double(),
                          "wnp");      

        prm.declare_entry("b_t", "0.0",
                          Patterns::Double(),
                          "wnp");
      }
      prm.leave_subsection();
    }

    void Materials::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        mu1 = prm.get_double("mu1");
        mu2 = prm.get_double("mu2");
        m = prm.get_double("m");
        gamma_dot_0 = prm.get_double("gamma_dot_0");
        dG = prm.get_double("dG");
        Ad = prm.get_double("Ad");
        tau0 = prm.get_double("tau0");
        d0s = prm.get_double("d0s");
        m_tau = prm.get_double("m_tau");
        a = prm.get_double("a");
        b = prm.get_double("b");
        sigma0 = prm.get_double("sigma0");
        nu1 = prm.get_double("nu1");
        nu2 = prm.get_double("nu2");
        de = prm.get_double("de");
        temp = prm.get_double("temp");
        zita = prm.get_double("zita");
        wnp = prm.get_double("wnp");
        y0 = prm.get_double("y0");
        x0 = prm.get_double("x0");
        a_t = prm.get_double("a_t");
        b_t = prm.get_double("b_t");
      }
      prm.leave_subsection();
    }

// @sect4{Linear solver}

// Next, we choose both solver and preconditioner settings.  The use of an
// effective preconditioner is critical to ensure convergence when a large
// nonlinear motion occurs within a Newton increment.
    struct LinearSolver
    {
      std::string type_lin;
      double      tol_lin;
      double      max_iterations_lin;
      std::string preconditioner_type;
      double      preconditioner_relaxation;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void LinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.declare_entry("Solver type", "CG",
                          Patterns::Selection("CG|Direct"),
                          "Type of solver used to solve the linear system");

        prm.declare_entry("Residual", "1e-6",
                          Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");

        prm.declare_entry("Max iteration multiplier", "1",
                          Patterns::Double(0.0),
                          "Linear solver iterations (multiples of the system matrix size)");

        prm.declare_entry("Preconditioner type", "ssor",
                          Patterns::Selection("jacobi|ssor"),
                          "Type of preconditioner");

        prm.declare_entry("Preconditioner relaxation", "0.65",
                          Patterns::Double(0.0),
                          "Preconditioner relaxation value");
      }
      prm.leave_subsection();
    }

    void LinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        type_lin = prm.get("Solver type");
        tol_lin = prm.get_double("Residual");
        max_iterations_lin = prm.get_double("Max iteration multiplier");
        preconditioner_type = prm.get("Preconditioner type");
        preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
      }
      prm.leave_subsection();
    }

// @sect4{Nonlinear solver}

// A Newton-Raphson scheme is used to solve the nonlinear system of governing
// equations.  We now define the tolerances and the maximum number of
// iterations for the Newton-Raphson nonlinear solver.
    struct NonlinearSolver
    {
      unsigned int max_iterations_NR;
      double       tol_f;
      double       tol_u;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations Newton-Raphson", "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Tolerance force", "1.0e-9",
                          Patterns::Double(0.0),
                          "Force residual tolerance");

        prm.declare_entry("Tolerance displacement", "1.0e-6",
                          Patterns::Double(0.0),
                          "Displacement error tolerance");
      }
      prm.leave_subsection();
    }

    void NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f = prm.get_double("Tolerance force");
        tol_u = prm.get_double("Tolerance displacement");
      }
      prm.leave_subsection();
    }

// @sect4{Time}

// Set the timestep size $ \varDelta t $ and the simulation end-time.
    struct Time
    {
      double delta_t_1;
      double delta_t_2;
      double end_time;
      int intToML;
      double delta_de; 
      double load_rate;
      int intRedDe;
      double RedAmount;


      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1",
                          Patterns::Double(),
                          "End time");

        prm.declare_entry("Time step size", "0.1",
                          Patterns::Double(),
                          "Time step size");

        prm.declare_entry("Time step size 2", "0.01",
                          Patterns::Double(),
                          "Time step size 2");         

        prm.declare_entry("At which timestep switch to ML ?", "0",
                          Patterns::Integer(),
                          "At which timestep switch to ML ?"); 

        prm.declare_entry("Delta de", "1e-5",
                          Patterns::Double(0.0),
                          "Delta de");
        prm.declare_entry("load_rate", "1e-4",
                          Patterns::Double(0.0),
                          "load_rate");     

        prm.declare_entry("reduce_at_timestep", "500",
                          Patterns::Integer(0),
                          "reduce_at_timestep");   

        prm.declare_entry("reduce_timestep", "0.5",
                          Patterns::Double(0.0),
                          "reduce_timestep");                                                                                                                              
      }
      prm.leave_subsection();
    }

    void Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time = prm.get_double("End time");
        delta_t_1 = prm.get_double("Time step size");
        delta_t_2 = prm.get_double("Time step size 2");
        intToML = prm.get_integer("At which timestep switch to ML ?");
        delta_de = prm.get_double("Delta de");
        load_rate = prm.get_double("load_rate");
        intRedDe = prm.get_integer("reduce_at_timestep");
        RedAmount = prm.get_double("reduce_timestep");
      }
      prm.leave_subsection();
    }

    struct OutputParam
      {
        unsigned int timestep_output;
        std::string  outtype;

        static void
        declare_parameters(ParameterHandler &prm);

        void
        parse_parameters(ParameterHandler &prm);
      };

    void OutputParam::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Output parameters");
      {
        prm.declare_entry("Time step number output", "1",
                          Patterns::Integer(0),
                          "Output data for time steps multiple of the given "
                          "integer value.");
        prm.declare_entry("Averaged results", "nodes",
                            Patterns::Selection("elements|nodes"),
                            "Output data associated with integration point values"
                            " averaged on elements or on nodes.");
      }
      prm.leave_subsection();
    }

    void OutputParam::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Output parameters");
      {
        timestep_output = prm.get_integer("Time step number output");
        outtype = prm.get("Averaged results");
      }
      prm.leave_subsection();
    }
    

// @sect4{All parameters}

// Finally we consolidate all of the above structures into a single container
// that holds all of our run-time selections.
    struct AllParameters :
      public AssemblyMethod,
      public BoundaryConditions,
      public FESystem,
      public Materials,
      public LinearSolver,
      public NonlinearSolver,
      public Time,
      public OutputParam
    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      AssemblyMethod::declare_parameters(prm);
      BoundaryConditions::declare_parameters(prm);
      FESystem::declare_parameters(prm);
      Materials::declare_parameters(prm);
      LinearSolver::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      Time::declare_parameters(prm);
      OutputParam::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      AssemblyMethod::parse_parameters(prm);
      BoundaryConditions::parse_parameters(prm);
      FESystem::parse_parameters(prm);
      Materials::parse_parameters(prm);
      LinearSolver::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      Time::parse_parameters(prm);
      OutputParam::parse_parameters(prm);
    }
  }


// @sect3{Time class}

// A simple class to store time data. Its functioning is transparent so no
// discussion is necessary. For simplicity we assume a constant time step
// size.
  class Time
  {
  public:
    Time (const double time_end,
          const double delta_t_1, const double delta_t_2, const double delta_de,
          const double load_rate)
      :
      timestep(0.0),
      time_current(0.0),
      time_end(time_end),
      delta_t_1(delta_t_1),
      delta_t_2(delta_t_2),
      delta_de(delta_de),
      load_rate(load_rate),
      delta_t(delta_de/load_rate)
    {}

    virtual ~Time()
    {}

    double current() const
    {
      return time_current;
    }
    double end() const
    {
      return time_end;
    }
    double get_delta_t() const
    {
      return delta_t;
    }   
    int get_timestep() const
    {
      return timestep;
    }
    void increment()
    {
      time_current += delta_t;
      ++timestep;
    }
    void adjust_timestep_size(const int &flag)
    {
      if (flag == 0)
        delta_t = delta_t_2;
      else if (flag == 1)
        delta_t = delta_t_1;
    }   
    int timestep; 
    double time_current;
    const double time_end;
    double delta_t_1;
    double delta_t_2;
    double delta_de;
    double load_rate;
    double delta_t;
  };


// Here the viscoelastic, viscoplastic, damage model is implemented
  template <int dim,typename NumberType>
  class Material_Compressible_Network
  {
  public:   
    Material_Compressible_Network(const Parameters::AllParameters &parameters,
                        const Time                      &time)
      :
      mu1(parameters.mu1),
      mu2(parameters.mu2),
      nu1(parameters.nu1),
      nu2(parameters.nu2),
      lambda_nh1((2*mu1*nu1)/(1-2*nu1)),
      lambda_nh2((2*mu2*nu2)/(1-2*nu2)),
      m(parameters.m),
      gamma_dot_0(parameters.gamma_dot_0),
      dG(parameters.dG),
      Ad(parameters.Ad),
      tau0(parameters.tau0),
      d0s(parameters.d0s),
      m_tau(parameters.m_tau),
      a(parameters.a),
      b(parameters.b),
      sigma0(parameters.sigma0),
      de(parameters.de),
      y0(parameters.y0),
      x0(parameters.x0),
      a_t(parameters.a_t),
      b_t(parameters.b_t),
      d(0.0),
      d_converged(0.0),
      eps0(0.0),
      eps0_converged(0.0),
      F_b_t(Physics::Elasticity::StandardTensors<dim>::I),
      F_p_t(Physics::Elasticity::StandardTensors<dim>::I),
      F_b_t_converged(Physics::Elasticity::StandardTensors<dim>::I),
      F_p_t_converged(Physics::Elasticity::StandardTensors<dim>::I),
      lc_max(1.0),
      lc_max_converged(1.0),
      strain(0.0),
      strain_converged(0.0),
      time(time),
      kappa((2.0 * mu1 * (1.0 + nu1)) / (3.0 * (1.0 - 2.0 * nu1))),
      c_1(mu1 / 2.0),
      Temper(parameters.temp),
      zita(parameters.zita),
      Tref(296.0),
      alphaZ(1+0.057*pow(zita,2)-9.5*zita),
      alphaT(2-exp(0.0126*(Temper-Tref))),
      wnp(parameters.wnp),
      ro_p(1.2),
      ro_np(3.0),
      vnp(wnp * ro_p / (ro_np + wnp*ro_p - ro_np*wnp)),
      X((1+5.0*vnp+18*pow(vnp,2))*alphaZ*alphaT),
      changetoML(parameters.intToML),
      intHiSi(parameters.hidden_size),
      alpha(parameters.alpha)
    {
      Assert(lambda_nh1 > 0, ExcInternalError());
    }

    virtual ~Material_Compressible_Network()
    {}


    void
    read_inML()
    {
      for (int k = 0; k < files; k++)
          {
            double myArray[file_rows[k]][file_columns[k]];
            if (k == 0)
            {
              std::ifstream inputfile("muX.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->muX(r,c) = 0.0;
                    this->muX(r,c) = myArray[r][c];
                  }
              }
            }
            else if(k == 1)
            {
              std::ifstream inputfile("muT.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->muT(r,c) = 0.0;
                    this->muT(r,c) = myArray[r][c];
                  }
              }
            }
            else if(k == 2)
            {
              std::ifstream inputfile("sigmaX.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->sigmaX(r,c) = 0.0;
                    this->sigmaX(r,c) = myArray[r][c];
                  }
              }
            }
            else if(k == 3)
            {
              std::ifstream inputfile("sigmaT.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->sigmaT(r,c) = 0.0;
                    this->sigmaT(r,c) = myArray[r][c];
                  }
              }
            }
            else if(k == 4)
            {
              std::ifstream inputfile("iweights_lstm1.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->iweights_lstm1(r,c) = 0.0;
                    this->iweights_lstm1(r,c) = myArray[r][c];
                  }
              }
            }
            else if(k == 5)
            {
              std::ifstream inputfile("iweights_lstm2.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->iweights_lstm2(r,c) = 0.0;
                    this->iweights_lstm2(r,c) = myArray[r][c];
                  }
              }
            }
            else if(k == 6)
            {
              std::ifstream inputfile("rweights_lstm1.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->rweights_lstm1(r,c) = 0.0;
                    this->rweights_lstm1(r,c) = myArray[r][c];
                  }
              }
            }
            else if(k == 7)
            {
              std::ifstream inputfile("rweights_lstm2.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->rweights_lstm2(r,c) = 0.0;
                    this->rweights_lstm2(r,c) = myArray[r][c];
                  }
              }
            }
            else if(k == 8)
            {
              std::ifstream inputfile("bias_lstm1.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->bias_lstm1(r,c) = 0.0;
                    this->bias_lstm1(r,c) = myArray[r][c];
                  }
              }
            }
            else if(k == 9)
            {
              std::ifstream inputfile("bias_lstm2.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->bias_lstm2(r,c) = 0.0;
                    this->bias_lstm2(r,c) = myArray[r][c];
                  }
              }
            }
            else if(k == 10)
            {
              std::ifstream inputfile("weights_cl.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->weights_cl(r,c) = 0.0;
                    this->weights_cl(r,c) = myArray[r][c];
                  }
              }
            }
            else if(k == 11)
            {
              std::ifstream inputfile("bias_cl.txt");
              for (int r = 0; r < file_rows[k]; r++) //Outer loop for rows
              {
                  for (int c = 0; c < file_columns[k]; c++) //inner loop for columns
                  {
                    inputfile >> myArray[r][c];  //Take input from file and put into myArray
                    this->bias_cl(r,c) = 0.0;
                    this->bias_cl(r,c) = myArray[r][c];
                  }
              }
            }        
      }
      for(int r = 0; r < intHiSi; r++)
      {
          this->cstate1(r) = 0.0;
          this->cstate2(r) = 0.0;
          this->hstate1(r) = 0.0;
          this->hstate2(r) = 0.0;
          this->Ypert(r) = 0.0;
          this->Cpert(r) = 0.0;
          this->cstate1_converged(r) = 0.0;
          this->cstate2_converged(r) = 0.0;
          this->hstate1_converged(r) = 0.0;
          this->hstate2_converged(r) = 0.0;
       }
    }

    void
    update_internal_equilibrium(const Tensor<2,dim,NumberType> &F, const NumberType &det_F,
                                const SymmetricTensor<2,dim,NumberType> &bstar, const int &flag)
    { 
      // At this point the time integration is done
      this->F_b_t = this->F_b_t_converged;
      this->F_p_t = this->F_p_t_converged;
      this->eps0 = this->eps0_converged;
      this->lc_max = this->lc_max_converged;
      this->strain = this->strain_converged;
      double d_t = this->d_converged;
      const NumberType k = 1.380649e-23;
      const NumberType T = Temper;
      const NumberType J = det_F;

      Tensor<2,dim,NumberType> F_ve_t = F*invert(F_p_t);
      Tensor<2,dim,NumberType> F_e_t = F_ve_t*invert(F_b_t);
      Tensor<2,dim,NumberType> F_p_t_dt = F_p_t;
      Tensor<2,dim,NumberType> F_b_t_dt = F_b_t;

      SymmetricTensor<2,dim,NumberType> TA_t = this->get_cauchy_eq(J,bstar);
      const SymmetricTensor<2,dim,NumberType> bstar_ve = Physics::Elasticity::Kinematics::b(F);
      const NumberType Ibar1 = trace(bstar_ve);
      NumberType f2 = Ibar1/3.0;
      NumberType lc = std::sqrt(f2);
      const NumberType lc_t = std::sqrt(X*(pow(lc,2)-1)+1);
      SymmetricTensor<2, dim, NumberType> v_aux;
      SymmetricTensor<2, dim, NumberType> lnv;
      Tensor<1, dim, NumberType > lambdas_e_1;
      Tensor< 1, dim, NumberType > lambdas_e_1_tr;
      Tensor<2,dim,NumberType> F_b_t_new;
      Tensor<2,dim,NumberType> F_p_t_new;
      Tensor<2,dim,NumberType> F_e_t_new;
      Tensor<2,dim,NumberType> F_ve_t_new;
      SymmetricTensor<2,dim,NumberType> TB_t;
      //SymmetricTensor<2,dim,NumberType> SB_t;
      NumberType sum_of_sq;
      NumberType strain_now;
      

    

      int it_step = 0;
      int val = 1; 

      while(val == 1){
            ++it_step;
            SymmetricTensor<2,dim,NumberType> b_e = Physics::Elasticity::Kinematics::b(F_e_t);
            TB_t = this->get_cauchy_neq_calc(J,b_e);
            
            std::array< std::pair< NumberType, Tensor< 1, dim, NumberType > >, dim >
            eigen_b = eigenvectors(b_e, SymmetricTensorEigenvectorMethod::ql_implicit_shifts);
            for (int a = 0; a < dim; ++a)
            {
                lambdas_e_1[a] = std::sqrt(eigen_b[a].first);
            }
            SymmetricTensor<2, dim, NumberType> V;
            SymmetricTensor<2, dim, NumberType> V_aux;
            for (int a = 0; a < dim; ++a)
            {
                V_aux = symmetrize(outer_product(eigen_b[a].second,eigen_b[a].second));
                V_aux *= lambdas_e_1[a];
                V += V_aux;
            } 

            Tensor<2, dim, NumberType> R = invert(V) * F_e_t;
    
            // Here starts Fi_backward
            SymmetricTensor<2,dim,NumberType> T_hat_v = symmetrize(transpose(R)*TB_t*R);

            SymmetricTensor<2, dim, NumberType> devStress = deviator(T_hat_v);
 
            sum_of_sq = 0.0;
            for(int i=0; i<dim ;i++)
            {
              for(int j=0; j<dim; j++)
              {
                sum_of_sq += (devStress[i][j] * devStress[i][j]); 
              }
            }

            NumberType tau = std::sqrt(sum_of_sq);
            NumberType tauHat = y0+a_t/(1+std::exp(-(this->d-x0)/b_t));
            NumberType base2 = tau/tauHat;     
            NumberType parameter_exp = (dG/(k*T))*(pow(base2,m)-1);   
            NumberType gamDot = gamma_dot_0*std::exp(parameter_exp);
            NumberType prefac = 0.0;
         
            if (tau != 0.0){
              prefac = gamDot / tau;
              F_b_t_new = (prefac*devStress*F_b_t)*time.get_delta_t() + F_b_t_dt; 
              }
            else{
                F_b_t_new = F_b_t;
            }        


            SymmetricTensor<2,dim,NumberType> be = Physics::Elasticity::Kinematics::b(F);
            std::array< std::pair< NumberType, Tensor< 1, dim, NumberType > >, dim >
                    eigen_be = eigenvectors(be, SymmetricTensorEigenvectorMethod::ql_implicit_shifts);
            for (int a = 0; a < dim; ++a)
            {
                lambdas_e_1_tr[a] = std::sqrt(eigen_be[a].first);
            }
            SymmetricTensor<2, dim, NumberType> v_aux;
            SymmetricTensor<2, dim, NumberType> lnv;
            strain_now = 0.0;
            for (int a = 0; a < dim; ++a)
            {
                SymmetricTensor<2, dim, NumberType>
                v_aux = symmetrize(outer_product(eigen_be[a].second,eigen_be[a].second));
                v_aux *= std::log(lambdas_e_1_tr[a]);
                lnv += v_aux;
            } 
            sum_of_sq = 0.0;
            for(int i=0; i<dim ;i++)
            {
              for(int j=0; j<dim; j++)
              {
                sum_of_sq += (lnv[i][j] * lnv[i][j]); 
              }
            }
            strain_now = std::sqrt(sum_of_sq);
            NumberType strainDot = abs((strain_now - strain) / time.get_delta_t());
            SymmetricTensor<2,dim,NumberType> b_ve = Physics::Elasticity::Kinematics::b(F_ve_t);
            SymmetricTensor<2,dim,NumberType> StressN = this->get_cauchy_eq(J,b_ve);
            SymmetricTensor<2,dim,NumberType> Stress_all = StressN + TB_t;
            SymmetricTensor<2, dim, NumberType> devStress_all = deviator(Stress_all);
            sum_of_sq = 0.0;
            for(int i=0; i<dim ;i++)
            {
              for(int j=0; j<dim; j++)
              {
                sum_of_sq += (devStress_all[i][j] * devStress_all[i][j]); 
              }
            }
            NumberType tau_all = std::sqrt(sum_of_sq);  
            NumberType base3 = abs(strain_now-eps0);
            NumberType gamDotP =  a*pow(base3,b)*strainDot;    
            prefac = 0.0;     
            if(tau_all > sigma0){
              if(eps0 < 1e-16){
                eps0 = strain_now;
                F_p_t_new = F_p_t;
              }
              else{
                prefac = gamDotP/tau_all;
                F_p_t_new = (prefac * (invert(F_ve_t)*devStress_all*F))*time.get_delta_t() + F_p_t_dt;
              }
            }
            else{
              eps0 = 0.0;
              F_p_t_new = F_p_t;
            }

            F_ve_t_new = F*invert(F_p_t_new);
            F_e_t_new = F_ve_t_new*invert(F_b_t_new); 

            Tensor<2,dim,NumberType> diff1 = F_b_t_new - F_b_t;
            Tensor<2,dim,NumberType> diff2 = F_p_t_new - F_p_t;
            sum_of_sq = 0;
            for(int i=0; i<dim ;i++)
            {
              for(int j=0; j<dim; j++)
              {
                sum_of_sq += (diff1[i][j] * diff1[i][j]); 
              }
            }           
            NumberType err1 = std::sqrt(sum_of_sq); 

            sum_of_sq = 0.0;
            for(int i=0; i<dim ;i++)
            {
              for(int j=0; j<dim; j++)
              {
                sum_of_sq += (diff2[i][j] * diff2[i][j]); 
              }
            }           
            NumberType err2 = std::sqrt(sum_of_sq); 
            NumberType error = err1;

            error = err1+err2;

            F_b_t = F_b_t_new;
            F_e_t = F_e_t_new;
            F_ve_t = F_ve_t_new;
            F_p_t = F_p_t_new;

            if(error < 1e-5){
                  val = 2;
            }
            if(it_step > 10000){
                  val = 2;
            }   

      }

      if(lc_t > lc_max){
        lc_max = lc_t;
        double exponent = Ad*(1.0-lc_max);
        d_t = 1.0 - std::exp(exponent);
      }

      // save internal variables here

      if(flag == 1){
        for (unsigned int a = 0; a < dim; ++a){
          for (unsigned int b = 0; b < dim; ++b){
            this->F_b_t[a][b]= Tensor<0,dim,NumberType>(F_b_t_new[a][b]);
            this->F_p_t[a][b]= Tensor<0,dim,NumberType>(F_p_t_new[a][b]);
            this->cauchy_neq_1[a][b] = Tensor<0,dim,NumberType>(TB_t[a][b]);
            this->cauchy_eq_1[a][b] = Tensor<0,dim,NumberType>(TA_t[a][b]);
        }
      }
        this->eps0 = eps0; 
        this->lc_max = lc_max;
        this->d_converged = d_t;
        this->strain = strain_now;
      }
      else if(flag == 0){ 
        // Save only the cauchy stress for the perturbation method
        for (unsigned int a = 0; a < dim; ++a){
          for (unsigned int b = 0; b < dim; ++b){
            this->cauchy_neq_1[a][b] = Tensor<0,dim,NumberType>(TB_t[a][b]);
            this->cauchy_eq_1[a][b] = Tensor<0,dim,NumberType>(TA_t[a][b]);
          }
        }      
      }

    }

    // These two voids do the same actually !!

    void update_end_timestep()
    {
        this->F_b_t_converged = this->F_b_t;
        this->F_p_t_converged = this->F_p_t;
        this->eps0_converged = this->eps0;
        this->d = this->d_converged;
        this->strain_converged = this->strain;
        this->lc_max_converged = this->lc_max;
        this->cstate1_converged = this->cstate1;
        this->hstate1_converged = this->hstate1;
        this->cstate2_converged = this->cstate2;
        this->hstate2_converged = this->hstate2;
    }

    void update_end_iteration()
    {
        this->F_b_t_converged = this->F_b_t;
        this->F_p_t_converged = this->F_p_t;
        this->eps0_converged = this->eps0;
        this->strain_converged = this->strain;
        this->d = this->d_converged;
        this->lc_max_converged = this->lc_max;
        this->cstate1_converged = this->cstate1;
        this->hstate1_converged = this->hstate1;
        this->cstate2_converged = this->cstate2;
        this->hstate2_converged = this->hstate2;        
    }    

    NumberType
    get_damage() const
    {
      return this->d;
    }

    void update_damage(const Tensor<2,dim,NumberType> &F)
    {
      const Tensor<2,dim,NumberType> Fbar = Physics::Elasticity::Kinematics::F_iso(F);
      const SymmetricTensor<2,dim,NumberType> bstar_iso = Physics::Elasticity::Kinematics::b(Fbar);
      NumberType d_t = this->d_converged;
      this->lc_max = this->lc_max_converged;
      const NumberType Ibar1 = trace(bstar_iso);
      NumberType f2 = Ibar1/3.0;
      NumberType lc = std::sqrt(f2);
      const NumberType lc_t = std::sqrt(X*(pow(lc,2)-1)+1);    
      if(lc_t > lc_max){
        lc_max = lc_t;
        NumberType exponent = Ad*(1.0-lc_max);
        d_t = 1.0 - std::exp(exponent);
      }  

      this->lc_max = lc_max;
      this->d_converged = d_t;
    } 


    void lstm_forward(const Tensor<2,dim,NumberType> &F,
                        const SymmetricTensor<2,dim,NumberType> &bstar, const int &flag) {
      NumberType dt = time.get_delta_t();
      Eigen::Matrix<double,7,1> input(7);
      VectorXd b(6) ;// {{bstar(0,0),bstar(0,1),bstar(0,2),bstar(1,1),bstar(2,2)}};
      this->hstate1 = this->hstate1_converged;
      this->cstate1 = this->cstate1_converged;
      this->hstate2 = this->hstate2_converged;
      this->cstate2 = this->cstate2_converged;
      b(0) = bstar[0][0];
      b(1) = bstar[0][1];
      b(2) = bstar[0][2];
      b(3) = bstar[1][1];
      b(4) = bstar[1][2];
      b(5) = bstar[2][2];
      for (int i = 0; i < 6; i++) 
      {
        input(i) = 0.0;
        input(i) = (b(i) - muX(i)) / sigmaX(i);
      }
      input(6)= (dt- muX(6)) / sigmaX(6);
      G = iweights_lstm1*input+rweights_lstm1*this->hstate1+bias_lstm1;
      Go = G(Eigen::seq(intHiSi*3,(intHiSi*4)-1));
      Gz = G(Eigen::seq(intHiSi*2,(intHiSi*3)-1));
      Gf = G(Eigen::seq(intHiSi,(intHiSi*2)-1));
      Gi = G(Eigen::seq(0,intHiSi-1));
      // Do nonlinear gate operation
      Gz.tanh();
      Gi = 1.0 / (1.0 + ((Gi*-1).exp()));
      Gf = 1.0 / (1.0 + ((Gf*-1).exp()));
      Go = 1.0 / (1.0 + ((Go*-1).exp()));
      C = Gz*Gi + Gf * this->cstate1.array();
      Y = C.tanh() * Go;
      if(flag == 1)
      {
        this->cstate1 = C.matrix();
        this->hstate1 = Y.matrix();
        this->lstm_forward2(F,flag,this->cstate2,this->hstate1,
          this->hstate2,
			    iweights_lstm2,
          rweights_lstm2,bias_lstm2);
      }
      else
      {
        this->lstm_forward2(F,flag,this->cstate2,Y.matrix(),this->hstate2,
			    iweights_lstm2,rweights_lstm2,bias_lstm2);
      }
    }

    void lstm_forward2(const Tensor<2,dim,NumberType> &F, const int &flag, Eigen::Matrix<double,200,1> C2,
                    Eigen::Matrix<double,200,1> H1,Eigen::Matrix<double,200,1> H2,
                    MatrixXd &iweights_lstm2,
                    MatrixXd &rweights_lstm2,
                    Eigen::Matrix<double,800,1> &bias_lstm2) {
      G = iweights_lstm2*H1+rweights_lstm2*H2+bias_lstm2;
      Go = G(Eigen::seq(intHiSi*3,(intHiSi*4)-1));
      Gz = G(Eigen::seq(intHiSi*2,(intHiSi*3)-1));
      Gf = G(Eigen::seq(intHiSi,(intHiSi*2)-1));
      Gi = G(Eigen::seq(0,intHiSi-1));
      // Do nonlinear gate operation
      Gz.tanh();
      Gi = 1.0 / (1.0 + ((Gi*-1).exp()));
      Gf = 1.0 / (1.0 + ((Gf*-1).exp()));
      Go = 1.0 / (1.0 + ((Go*-1).exp()));    
      C = Gz*Gi + Gf * C2.array();
      Y = C.tanh() * Go;
      if (flag == 1)
      {
        this->cstate2 = C.matrix();
        this->hstate2 = Y.matrix();
        this->cauchy_ML = this->cauchy_baseML();
        this->update_damage(F);
      }
      else if (flag == 0)
      {
        this->Cpert = C.matrix();
        this->Ypert = Y.matrix();
        this->cauchy_ML_pert = this->cauchy_baseMLPert(this->Ypert);     
      }
    }

    SymmetricTensor<2, dim, NumberType>
    get_cauchy_baseML() const
    {
      return this->cauchy_ML;
    }

    SymmetricTensor<2, dim, NumberType>
    get_cauchy_baseMLPert() const
    {
      return this->cauchy_ML_pert;
    }            

    SymmetricTensor<2, dim, NumberType>
    cauchy_baseML() const
    {
      Eigen::Matrix<double,6,1> YPred1(6);
      Eigen::Matrix<double,6,1> Output1(6);
      for (int i = 0; i < 6; i++)
      {
        Output1(i) = 0.0;
        YPred1(i) = 0.0;
      }      
      YPred1 = weights_cl*this->hstate2+bias_cl;
      for (int i = 0; i < 6; i++)
      {
        Output1(i) = YPred1(i)*sigmaT(i) + muT(i);
      }
      SymmetricTensor<2,dim,NumberType> OutputTensor;
      OutputTensor[0][0] = Output1(0);
      OutputTensor[0][1] = Output1(1);
      OutputTensor[0][2] = Output1(2);
      OutputTensor[1][1] = Output1(3);
      OutputTensor[1][2] = Output1(4);
      OutputTensor[2][2] = Output1(5);

      const SymmetricTensor<2,dim,NumberType> sigma1 = (1.0 - this->d) * (OutputTensor);
      return sigma1;
    }

    SymmetricTensor<2, dim, NumberType>
    cauchy_baseMLPert(Eigen::Matrix<double,200,1> H) const
    {
      Eigen::Matrix<double,6,1> YPred1(6);
      Eigen::Matrix<double,6,1> Output1(6);
      for (int i = 0; i < 6; i++)
      {
        Output1(i) = 0.0;
        YPred1(i) = 0.0;
      }      
      YPred1 = weights_cl*H+bias_cl;
      for (int i = 0; i < 6; i++)
      {
        Output1(i) = YPred1(i)*sigmaT(i) + muT(i);
      }
      SymmetricTensor<2,dim,NumberType> OutputTensor;
      OutputTensor[0][0] = Output1(0);
      OutputTensor[0][1] = Output1(1);
      OutputTensor[0][2] = Output1(2);
      OutputTensor[1][1] = Output1(3);
      OutputTensor[1][2] = Output1(4);
      OutputTensor[2][2] = Output1(5);
      const SymmetricTensor<2,dim,NumberType> sigma2 = (1.0 - this->d) * (OutputTensor);

      return sigma2;
    }       

    NumberType 
    get_eps0() const
    {
      return this->eps0_converged;
    }

    SymmetricTensor<2,dim,NumberType>
    get_cauchy_eq(const NumberType &det_F, const SymmetricTensor<2,dim,NumberType> &bstar) const
    {
      const NumberType mu_mod = X*mu1;
      const NumberType kappa_mod = X*kappa;
      const NumberType J = det_F;
      //const NumberType Jt = 1 + alphaT*(Temper-Tref);
      //const NumberType Jz = 1 + this->alphaZ*this->zita;
      const NumberType Jt = 1.0;
      const NumberType Jz = 1.0;
      const NumberType Jm = J/(Jt*Jz);
      const SymmetricTensor<2,dim,NumberType> devbstar = deviator(bstar);
      const SymmetricTensor<2,dim,NumberType> sigma = mu_mod/J * devbstar + 0.5 * kappa_mod*(Jm - 1/Jm)/(Jt*Jz)*Physics::Elasticity::StandardTensors<dim>::I;
      
      return sigma;
    }

    SymmetricTensor<2,dim,NumberType>
    get_cauchy_neq_calc(const NumberType &det_F, const SymmetricTensor<2,dim,NumberType> &bstar) const
    {
      const NumberType mu_mod = X*mu2;
      const NumberType J = det_F;
      const SymmetricTensor<2,dim,NumberType> devbstar = deviator(bstar);
      const SymmetricTensor<2,dim,NumberType> sigma = mu_mod/J * devbstar;



      return sigma;
    }    

    SymmetricTensor<2,dim,NumberType>
    get_cauchy_neq() const
    {

      return cauchy_neq_1;
    }

    SymmetricTensor<2,dim,NumberType>
    get_cauchy_eq_saved() const
    {

      return cauchy_eq_1;
    }        

    SymmetricTensor<2,dim,NumberType>
    get_s_piola_eq(const NumberType &det_F,
    const SymmetricTensor<2,dim,NumberType> &C) const
    {
      const SymmetricTensor<2,dim,NumberType> invc = invert(C);
      const NumberType               lnJ = std::log(det_F);


      SymmetricTensor<2,dim,NumberType> S_piola_eq = lambda_nh1*lnJ*invc + mu1*(Physics::Elasticity::StandardTensors<dim>::I - invc);

      

      return S_piola_eq;
    }

    SymmetricTensor<2,dim,NumberType>
    get_s_piola_neq_calc(const NumberType &det_F,
    const SymmetricTensor<2,dim,NumberType> &C) const
    {
      const SymmetricTensor<2,dim,NumberType> invc = invert(C);
      const NumberType               lnJ = std::log(det_F);
      const SymmetricTensor<2,dim,NumberType> S_piola_neq = lambda_nh2*lnJ*invc + mu2*(Physics::Elasticity::StandardTensors<dim>::I - invc);


      return S_piola_neq;
    }

    SymmetricTensor<2,dim,NumberType>
    get_s_piola_neq() const
    {
      return piola_neq_1;
    }    

    SymmetricTensor<2, dim, NumberType>
    get_cauchy_base() const
    {
      
      const SymmetricTensor<2,dim,NumberType> sigma = (1.0 - this->d) * (get_cauchy_neq() + get_cauchy_eq_saved());

        return sigma;
    }    
            

    SymmetricTensor<2, dim, NumberType>
    get_piola_base(const Tensor<2,dim, NumberType> &F, const NumberType &det_F, 
                    const SymmetricTensor<2,dim,NumberType> &C) const
    {
        return ( (1.0 - this->d) * (get_s_piola_neq() + get_s_piola_eq(det_F,C)) );
    }      


    SymmetricTensor<4,dim,NumberType>
    get_Jc(const Tensor<2, dim, NumberType> &F,
          const SymmetricTensor<2,dim,NumberType> &sigma_cauchy,
          const NumberType &det_F)
          {
            SymmetricTensor<4,dim,NumberType> C_num_final_tens;
            SymmetricTensor<4,dim,NumberType> C_num_final_tens2;
            NumberType C_num[9][9], C_num_final[9][9];
            int ini[3], inj[3];
            double ei[3][3], ej[3][3], ei_t[3][3], ej_t[3][3], el[3][3], er[3][3];
            SymmetricTensor<2,dim,NumberType> bp;
            SymmetricTensor<2,dim,NumberType> bp_bar;
            Tensor<2, dim, NumberType> delF;
            Tensor<2, dim, NumberType> Fp;
            Tensor<2, dim, NumberType> Fp_bar;
            Tensor<2, dim, NumberType> Fl;
            Tensor<2, dim, NumberType> Fr;
            NumberType Jp;
            for(int i=0;i<3;i++)    
            {    
              for(int j=0;j<3;j++)    
              {    
                el[i][j] = 0;
                er[i][j] = 0;
                ej[i][j] = 0;  
                ei[i][j] = 0; 
                ei_t[i][j] = 0;
                ej_t[i][j] = 0;
              }    
            }

            for(int i = 0; i < dim; i++){
                if(i == 0){
                  ini[0] = 0;
                  ini[1] = 1;
                  ini[2] = 2;
                } else if(i == 1) {
                  ini[0] = 3;
                  ini[1] = 4;
                  ini[2] = 5;
                } else if(i == 2) {
                  ini[0] = 6;
                  ini[1] = 7;
                  ini[2] = 8;
                }
                for(int l = 0; l < 3; l++){
                  ei[0][l] = 0;
                  ei_t[l][0] = 0;
                }                
                  ei[0][i] = 1;
                  ei_t[i][0] = 1;

                for(int j = 0; j < dim; j++){
                  if(j == 0){
                    inj[0] = 0;
                    inj[1] = 1;
                    inj[2] = 2;
                  } else if(j == 1) {
                    inj[0] = 3;
                    inj[1] = 4;
                    inj[2] = 5;
                  } else if(j == 2) {
                    inj[0] = 6;
                    inj[1] = 7;
                    inj[2] = 8;
                  }

                  for(int l = 0; l < 3; l++){
                    ej[0][l] = 0;
                    ej_t[l][0] = 0;
                  }
                  ej[0][j] = 1;
                  ej_t[j][0] = 1;
                   


                  for(int o = 0; o < dim; o++){
                    for(int p = 0; p < dim; p++){
                      el[o][p] = 0;
                      er[o][p] = 0;
                      for(int k = 0; k < dim; k++){
                        el[o][p] += ei_t[o][k] * ej[k][p];
                        er[o][p] += ej_t[o][k] * ei[k][p];
                      }
                    }
                  }


                  for(int o = 0; o < dim; o++){
                    for(int p = 0; p < dim; p++){
                      Fl[o][p] = 0;
                      Fr[o][p] = 0;
                      for(int k = 0; k < dim; k++){
                        Fl[o][p] += el[o][k] * F[k][p];
                        Fr[o][p] += er[o][k] * F[k][p];
                      }
                    }
                  }
                                              

                  for(int o = 0; o < dim; o++){
                    for(int  p = 0; p < dim; p++){
                      delF[o][p] = (this->alpha/2.0)*(Fl[o][p]+Fr[o][p]);
                    }			
                  }

                  
                  Fp = F + delF;
                  Fp_bar = Physics::Elasticity::Kinematics::F_iso(Fp);
                  Jp = determinant(Fp);
                  bp = Physics::Elasticity::Kinematics::b(Fp);
                  bp_bar = Physics::Elasticity::Kinematics::b(Fp_bar);
                  SymmetricTensor<2,dim,NumberType> sigma_cauchy_p;
                  if(time.get_timestep() < changetoML)
                  {
                    this->update_internal_equilibrium(Fp_bar,Jp,bp_bar,0);
                    sigma_cauchy_p = this->get_cauchy_base()*Jp;

                  }
                  else if(time.get_timestep() >= changetoML)
                  {
                    this->lstm_forward(Fp,bp,0);
                    sigma_cauchy_p = this->get_cauchy_baseMLPert()*Jp;
                  }
                  //this->update_internal_equilibrium(Fp_bar,Jp,bp,0);
                  //this->lstm_forward(Fp,Jp,bp,0);
                  //SymmetricTensor<2,dim,NumberType> sigma_cauchy_p = this->get_cauchy_baseMLPert(this->Ypert)*Jp;

                  for(int x = 0; x < dim; x++){
                    for(int y = 0; y < dim; y++){
                      C_num[ini[x]][inj[y]] = (sigma_cauchy_p[x][y] - sigma_cauchy[x][y]*det_F) / this->alpha;				
                    }
                  }
                }
            }

            	for(int i=0;i<dim*3;i++)    
              {    
                for(int j=0;j<dim*3;j++)    
                {    
                C_num_final[i][j] = C_num[i][j]/det_F;
                }
              }



          for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
              for(int k = 0; k < 3; k++){
                for(int l = 0; l < 3; l++){
                  C_num_final_tens[i][j][k][l] = 0.0;
                }
              }
            }
          } 

 /*           // First row
          C_num_final_tens[0][0][0][0] = C_num_final[0][0];
          C_num_final_tens[0][0][0][1] = C_num_final[0][1];
          C_num_final_tens[0][0][0][2] = C_num_final[0][2];
          C_num_final_tens[0][1][0][0] = C_num_final[0][3];
          C_num_final_tens[0][1][0][1] = C_num_final[0][4];
          C_num_final_tens[0][1][0][2] = C_num_final[0][5];
          C_num_final_tens[0][2][0][0] = C_num_final[0][6];
          C_num_final_tens[0][2][0][1] = C_num_final[0][7];
          C_num_final_tens[0][2][0][2] = C_num_final[0][8];
          // Second row
          C_num_final_tens[0][0][1][0] = C_num_final[1][0];
          C_num_final_tens[0][0][1][1] = C_num_final[1][1];
          C_num_final_tens[0][0][1][2] = C_num_final[1][2];
          C_num_final_tens[0][1][1][0] = C_num_final[1][3];
          C_num_final_tens[0][1][1][1] = C_num_final[1][4];
          C_num_final_tens[0][1][1][2] = C_num_final[1][5];
          C_num_final_tens[0][2][1][0] = C_num_final[1][6];
          C_num_final_tens[0][2][1][1] = C_num_final[1][7];
          C_num_final_tens[0][2][1][2] = C_num_final[1][8];
          // Third row
          C_num_final_tens[0][0][2][0] = C_num_final[2][0];
          C_num_final_tens[0][0][2][1] = C_num_final[2][1];
          C_num_final_tens[0][0][2][2] = C_num_final[2][2];
          C_num_final_tens[0][1][2][0] = C_num_final[2][3];
          C_num_final_tens[0][1][2][1] = C_num_final[2][4];
          C_num_final_tens[0][1][2][2] = C_num_final[2][5];
          C_num_final_tens[0][2][2][0] = C_num_final[2][6];
          C_num_final_tens[0][2][2][1] = C_num_final[2][7];
          C_num_final_tens[0][2][2][2] = C_num_final[2][8];
          // Fourth row
          C_num_final_tens[1][0][0][0] = C_num_final[3][0];
          C_num_final_tens[1][0][0][1] = C_num_final[3][1];
          C_num_final_tens[1][0][0][2] = C_num_final[3][2];
          C_num_final_tens[1][1][0][0] = C_num_final[3][3];
          C_num_final_tens[1][1][0][1] = C_num_final[3][4];
          C_num_final_tens[1][1][0][2] = C_num_final[3][5];
          C_num_final_tens[1][2][0][0] = C_num_final[3][6];
          C_num_final_tens[1][2][0][1] = C_num_final[3][7];
          C_num_final_tens[1][2][0][2] = C_num_final[3][8];
          // Fifth row
          C_num_final_tens[1][0][1][0] = C_num_final[4][0];
          C_num_final_tens[1][0][1][1] = C_num_final[4][1];
          C_num_final_tens[1][0][1][2] = C_num_final[4][2];
          C_num_final_tens[1][1][1][0] = C_num_final[4][3];
          C_num_final_tens[1][1][1][1] = C_num_final[4][4];
          C_num_final_tens[1][1][1][2] = C_num_final[4][5];
          C_num_final_tens[1][2][1][0] = C_num_final[4][6];
          C_num_final_tens[1][2][1][1] = C_num_final[4][7];
          C_num_final_tens[1][2][1][2] = C_num_final[4][8];
          // Sixth row
          C_num_final_tens[1][0][2][0] = C_num_final[5][0];
          C_num_final_tens[1][0][2][1] = C_num_final[5][1];
          C_num_final_tens[1][0][2][2] = C_num_final[5][2];
          C_num_final_tens[1][1][2][0] = C_num_final[5][3];
          C_num_final_tens[1][1][2][1] = C_num_final[5][4];
          C_num_final_tens[1][1][2][2] = C_num_final[5][5];
          C_num_final_tens[1][2][2][0] = C_num_final[5][6];
          C_num_final_tens[1][2][2][1] = C_num_final[5][7];
          C_num_final_tens[1][2][2][2] = C_num_final[5][8];
          // Seventh row
          C_num_final_tens[2][0][0][0] = C_num_final[6][0];
          C_num_final_tens[2][0][0][1] = C_num_final[6][1];
          C_num_final_tens[2][0][0][2] = C_num_final[6][2];
          C_num_final_tens[2][1][0][0] = C_num_final[6][3];
          C_num_final_tens[2][1][0][1] = C_num_final[6][4];
          C_num_final_tens[2][1][0][2] = C_num_final[6][5];
          C_num_final_tens[2][2][0][0] = C_num_final[6][6];
          C_num_final_tens[2][2][0][1] = C_num_final[6][7];
          C_num_final_tens[2][2][0][2] = C_num_final[6][8];
          // Eighth row
          C_num_final_tens[2][0][1][0] = C_num_final[7][0];
          C_num_final_tens[2][0][1][1] = C_num_final[7][1];
          C_num_final_tens[2][0][1][2] = C_num_final[7][2];
          C_num_final_tens[2][1][1][0] = C_num_final[7][3];
          C_num_final_tens[2][1][1][1] = C_num_final[7][4];
          C_num_final_tens[2][1][1][2] = C_num_final[7][5];
          C_num_final_tens[2][2][1][0] = C_num_final[7][6];
          C_num_final_tens[2][2][1][1] = C_num_final[7][7];
          C_num_final_tens[2][2][1][2] = C_num_final[7][8];
          // Ninth row
          C_num_final_tens[2][0][2][0] = C_num_final[8][0];
          C_num_final_tens[2][0][2][1] = C_num_final[8][1];
          C_num_final_tens[2][0][2][2] = C_num_final[8][2];
          C_num_final_tens[2][1][2][0] = C_num_final[8][3];
          C_num_final_tens[2][1][2][1] = C_num_final[8][4];
          C_num_final_tens[2][1][2][2] = C_num_final[8][5];
          C_num_final_tens[2][2][2][0] = C_num_final[8][6];
          C_num_final_tens[2][2][2][1] = C_num_final[8][7];
          C_num_final_tens[2][2][2][2] = C_num_final[8][8]; */

         C_num_final_tens2[0][0][0][0] = C_num_final[0][0];
          C_num_final_tens2[1][1][1][1] = C_num_final[4][4];
          C_num_final_tens2[2][2][2][2] = C_num_final[8][8];
          C_num_final_tens2[0][1][0][1] = C_num_final[0][4];
          C_num_final_tens2[0][2][0][2] = C_num_final[0][8];
          C_num_final_tens2[0][2][0][2] = C_num_final[0][8];
          C_num_final_tens2[1][2][1][2] = C_num_final[4][8];
          C_num_final_tens2[0][0][1][1] = C_num_final[1][1];
          C_num_final_tens2[0][0][2][2] = C_num_final[2][2];
          C_num_final_tens2[1][1][2][2] = C_num_final[5][5]; 

          C_num_final_tens2[1][1][0][0] = C_num_final[1][1];
          C_num_final_tens2[2][2][0][0] = C_num_final[2][2];
          C_num_final_tens2[2][2][1][1] = C_num_final[5][5]; 


          const SymmetricTensor<4,dim,NumberType> C_num_final_tens_spatial = C_num_final_tens2; 
          return C_num_final_tens_spatial;
          }

//  NEO HOOK is here implemented for comparison and to validate the perturbation method. Not needed anymore !!
    NumberType
    get_Psi(const NumberType                        &det_F,
            const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return get_Psi_vol(det_F) + get_Psi_iso(b_bar);
    }

    // The second function determines the Kirchhoff stress $\boldsymbol{\tau}
    // = \boldsymbol{\tau}_{\textrm{iso}} + \boldsymbol{\tau}_{\textrm{vol}}$
    SymmetricTensor<2,dim,NumberType>
    get_tau(const NumberType                        &det_F,
            const SymmetricTensor<2,dim,NumberType> &b_bar)
    {
      // See Holzapfel p231 eq6.98 onwards
      return get_tau_vol(det_F) + get_tau_iso(b_bar);
    }

    // The fourth-order elasticity tensor in the spatial setting
    // $\mathfrak{c}$ is calculated from the SEF $\Psi$ as $ J
    // \mathfrak{c}_{ijkl} = F_{iA} F_{jB} \mathfrak{C}_{ABCD} F_{kC} F_{lD}$
    // where $ \mathfrak{C} = 4 \frac{\partial^2 \Psi(\mathbf{C})}{\partial
    // \mathbf{C} \partial \mathbf{C}}$
    SymmetricTensor<4,dim,NumberType>
    get_Jc_const(const NumberType                        &det_F,
           const SymmetricTensor<2,dim,NumberType> &b_bar, const Tensor<2,dim,NumberType> &F) const
    {
      const SymmetricTensor<4,dim,NumberType> Jc_const = get_Jc_vol(det_F) + get_Jc_iso(b_bar);
        
      return Jc_const;
    }


  protected:
    const double mu1;
    const double mu2;
    const double nu1;
    const double nu2;
    const double lambda_nh1;
    const double lambda_nh2;
    const double m;
    const double gamma_dot_0;
    const double dG;
    const double Ad;
    const double tau0;  
    const double d0s;
    const double m_tau;
    const double a;
    const double b;
    const double sigma0;
    double de;
    const double y0;
    const double x0;
    const double a_t;
    const double b_t;
    double d;
    double d_converged;
    double eps0;
    double eps0_converged;
    Tensor<2, dim, double> F_b_t;
    Tensor<2, dim, double> F_p_t;
    Tensor<2, dim, double> F_b_t_converged;
    Tensor<2, dim, double> F_p_t_converged;
    double lc_max;
    double lc_max_converged;
    double strain;
    double strain_converged;
    const Time  &time;
    SymmetricTensor<2, dim, NumberType> cauchy_neq_1;
    SymmetricTensor<2, dim, NumberType> cauchy_eq_1;
    SymmetricTensor<2, dim, NumberType> piola_neq_1;
    SymmetricTensor<2,dim,NumberType> cauchy_ML;
    SymmetricTensor<2,dim,NumberType> cauchy_ML_pert;
    double dt;
    const double kappa;
    const double c_1;
    const double Temper;
    const double zita;
    const double Tref;
    const double alphaZ;
    const double alphaT;
    const double wnp;
    const double ro_p;
    const double ro_np;
    const double vnp;
    const double X;
    const int changetoML;
    const int intHiSi;
    int file_columns[12] = {1,1,1,1,7,intHiSi,intHiSi,intHiSi,1,1,intHiSi,1};
    int file_rows[12] = {7,6,7,6,intHiSi*4,intHiSi*4,intHiSi*4,intHiSi*4,intHiSi*4,intHiSi*4,6,6};
    int files = 12;
    Eigen::Matrix<double,7,1> muX;
    Eigen::Matrix<double,6,1> muT;
    Eigen::Matrix<double,7,1> sigmaX;
    Eigen::Matrix<double,6,1> sigmaT;
    Eigen::Matrix<double,800,7> iweights_lstm1;
    MatrixXd iweights_lstm2 = MatrixXd(intHiSi*4,intHiSi);
    MatrixXd rweights_lstm1 = MatrixXd(intHiSi*4,intHiSi);
    MatrixXd rweights_lstm2 = MatrixXd(intHiSi*4,intHiSi);
    Eigen::Matrix<double,800,1> bias_lstm1;
    Eigen::Matrix<double,800,1> bias_lstm2;
    Eigen::Matrix<double,6,200> weights_cl;
    Eigen::Matrix<double,6,1> bias_cl;
    Eigen::Matrix<double,200,1> cstate1;
    Eigen::Matrix<double,200,1> cstate2;
    Eigen::Matrix<double,200,1> hstate1;
    Eigen::Matrix<double,200,1> hstate2;
    Eigen::Matrix<double,200,1> cstate1_converged;
    Eigen::Matrix<double,200,1> cstate2_converged;
    Eigen::Matrix<double,200,1> hstate1_converged;
    Eigen::Matrix<double,200,1> hstate2_converged;
    Eigen::Matrix<double,200,1> Cpert;
    Eigen::Matrix<double,200,1> Ypert;

    Eigen::Array<double,800,1> G;
    Eigen::Array<double,200,1> Gz;
    Eigen::Array<double,200,1> Gi;
    Eigen::Array<double,200,1> Gf;
    Eigen::Array<double,200,1> Go;
    Eigen::Array<double,200,1> C;
    Eigen::Array<double,200,1> Y;        
    const NumberType alpha;
    // Value of the volumetric free energy
    NumberType
    get_Psi_vol(const NumberType &det_F) const
    {
      return (kappa / 4.0) * (det_F*det_F - 1.0 - 2.0*std::log(det_F));
    }

    // Value of the isochoric free energy
    NumberType
    get_Psi_iso(const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return c_1 * (trace(b_bar) - dim);
    }

    // Derivative of the volumetric free energy with respect to
    // $J$ return $\frac{\partial
    // \Psi_{\text{vol}}(J)}{\partial J}$
    NumberType
    get_dPsi_vol_dJ(const NumberType &det_F) const
    {
      return (kappa / 2.0) * (det_F - 1.0 / det_F);
    }

    // The following functions are used internally in determining the result
    // of some of the public functions above. The first one determines the
    // volumetric Kirchhoff stress $\boldsymbol{\tau}_{\textrm{vol}}$.
    // Note the difference in its definition when compared to step-44.
    SymmetricTensor<2,dim,NumberType>
    get_tau_vol(const NumberType &det_F) const
    {
      return NumberType(get_dPsi_vol_dJ(det_F) * det_F) * Physics::Elasticity::StandardTensors<dim>::I;
    }

    // Next, determine the isochoric Kirchhoff stress
    // $\boldsymbol{\tau}_{\textrm{iso}} =
    // \mathcal{P}:\overline{\boldsymbol{\tau}}$:
    SymmetricTensor<2,dim,NumberType>
    get_tau_iso(const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return Physics::Elasticity::StandardTensors<dim>::dev_P * get_tau_bar(b_bar);
    }

    // Then, determine the fictitious Kirchhoff stress
    // $\overline{\boldsymbol{\tau}}$:
    SymmetricTensor<2,dim,NumberType>
    get_tau_bar(const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return 2.0 * c_1 * b_bar;
    }

    // Second derivative of the volumetric free energy wrt $J$. We
    // need the following computation explicitly in the tangent so we make it
    // public.  We calculate $\frac{\partial^2
    // \Psi_{\textrm{vol}}(J)}{\partial J \partial
    // J}$
    NumberType
    get_d2Psi_vol_dJ2(const NumberType &det_F) const
    {
      return ( (kappa / 2.0) * (1.0 + 1.0 / (det_F * det_F)));
    }

    // Calculate the volumetric part of the tangent $J
    // \mathfrak{c}_\textrm{vol}$. Again, note the difference in its
    // definition when compared to step-44. The extra terms result from two
    // quantities in $\boldsymbol{\tau}_{\textrm{vol}}$ being dependent on
    // $\boldsymbol{F}$.
    SymmetricTensor<4,dim,NumberType>
    get_Jc_vol(const NumberType &det_F) const
    {
      // See Holzapfel p265
      return det_F
             * ( (get_dPsi_vol_dJ(det_F) + det_F * get_d2Psi_vol_dJ2(det_F))*Physics::Elasticity::StandardTensors<dim>::IxI
                 - (2.0 * get_dPsi_vol_dJ(det_F))*Physics::Elasticity::StandardTensors<dim>::S );
    }

    // Calculate the isochoric part of the tangent $J
    // \mathfrak{c}_\textrm{iso}$:
    SymmetricTensor<4,dim,NumberType>
    get_Jc_iso(const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      const SymmetricTensor<2, dim> tau_bar = get_tau_bar(b_bar);
      const SymmetricTensor<2, dim> tau_iso = get_tau_iso(b_bar);
      const SymmetricTensor<4, dim> tau_iso_x_I
        = outer_product(tau_iso,
                        Physics::Elasticity::StandardTensors<dim>::I);
      const SymmetricTensor<4, dim> I_x_tau_iso
        = outer_product(Physics::Elasticity::StandardTensors<dim>::I,
                        tau_iso);
      const SymmetricTensor<4, dim> c_bar = get_c_bar();

      return (2.0 / dim) * trace(tau_bar)
             * Physics::Elasticity::StandardTensors<dim>::dev_P
             - (2.0 / dim) * (tau_iso_x_I + I_x_tau_iso)
             + Physics::Elasticity::StandardTensors<dim>::dev_P * c_bar
             * Physics::Elasticity::StandardTensors<dim>::dev_P;
    }

    // Calculate the fictitious elasticity tensor $\overline{\mathfrak{c}}$.
    // For the material model chosen this is simply zero:
    SymmetricTensor<4,dim,double>
    get_c_bar() const
    {
      return SymmetricTensor<4, dim>();
    }
  };

// As seen in step-18, the <code> PointHistory </code> class offers a method
// for storing data at the quadrature points.  Here each quadrature point
// holds a pointer to a material description.  Thus, different material models
// can be used in different regions of the domain.  Among other data, we
// choose to store the Kirchhoff stress $\boldsymbol{\tau}$ and the tangent
// $J\mathfrak{c}$ for the quadrature points.
  template <int dim,typename NumberType>
  class PointHistory
  {
  public: 
    PointHistory()
    : old_stress(SymmetricTensor<2, dim,NumberType>())
    {}

    virtual ~PointHistory()
    {}

    // The first function is used to create a material object and to
    // initialize all tensors correctly: The second one updates the stored
    // values and stresses based on the current deformation measure
    // $\textrm{Grad}\mathbf{u}_{\textrm{n}}$.
    void setup_lqp (const Parameters::AllParameters &parameters,
                            const Time              &time)
    {
      material = std::make_shared<Material_Compressible_Network<dim,NumberType>>(parameters,time);
      if(parameters.switchML == "On" )
        material->read_inML();
    }

    void update_values(const Tensor<2, dim,NumberType> &Grad_u_n,
                       const Parameters::AllParameters &parameters,
                       const Time              &time)
    {
      // This called after each timestep to update the internal variables
      Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(Grad_u_n);
      Tensor<2, dim> Fbar = Physics::Elasticity::Kinematics::F_iso(F);
      SymmetricTensor<2,dim,NumberType> b = Physics::Elasticity::Kinematics::b(F);
      SymmetricTensor<2,dim,NumberType> bstar = Physics::Elasticity::Kinematics::b(Fbar);
      NumberType det_F = determinant(F);
      if (time.get_timestep() < parameters.intToML)
      {
        if(parameters.switchML == "On" )
          material->lstm_forward(F,bstar,1);
        
	      
        material->update_internal_equilibrium(Fbar,det_F,bstar,1);
      }
      else if (time.get_timestep() >= parameters.intToML)
      {
        material->lstm_forward(F,b,1);
      }
      material->update_end_iteration();
	    
    }
    // We offer an interface to retrieve certain data.
    // This is the strain energy:

    // Here are the kinetic variables. These are used in the material and
    // global tangent matrix and residual assembly operations:
    // First is the Kirchhoff stress:
    SymmetricTensor<2,dim,NumberType>
    get_s_piola_eq(const NumberType                        &det_F,
            const SymmetricTensor<2,dim,NumberType> &C) const
    {
      return material->get_s_piola(det_F,C);
    }

    SymmetricTensor<2,dim,NumberType>
    get_s_piola_neq_calc(const NumberType                        &det_F,
            const SymmetricTensor<2,dim,NumberType> &C) const
    {
      return material->get_s_piola_neq_calc(det_F,C);
    }

    SymmetricTensor<2,dim,NumberType>
    get_cauchy_eq(const NumberType &det_F, const SymmetricTensor<2,dim,NumberType> &bstar) const
    {
      return material->get_cauchy_eq(det_F,bstar);
    }

    SymmetricTensor<2,dim,NumberType>
    get_cauchy_neq_calc( const NumberType &det_F, const SymmetricTensor<2,dim,NumberType> &bstar) const
    {
      return material->get_cauchy_neq_calc(det_F,bstar);
    }

    SymmetricTensor<2,dim,NumberType>
    get_cauchy_base() const
    {
      return material->get_cauchy_base();
    }  
    
    
    SymmetricTensor<2,dim,NumberType>
    get_piola_base(const Tensor<2,dim, NumberType> &F, const NumberType &det_F, 
                    const SymmetricTensor<2,dim,NumberType> &C) const
    {
      return material->get_piola_base(F,det_F,C);
    }

    NumberType
    get_damage() const
    {
      return material->get_damage();
    }

    NumberType 
    get_eps0() const
    {
      return material->get_eps0();
    }


    void
    update_end_timestep()
    {
        material->update_end_timestep();
    }

    void
    update_end_iteration()
    {
        material->update_end_iteration();
    }

    void
    update_internal_equilibrium(const Tensor<2,dim,NumberType> &Fbar, const NumberType &det_F,
                                const SymmetricTensor<2,dim,NumberType> &bstar, const int &flag)
    {
        material->update_internal_equilibrium(Fbar,det_F,bstar,flag);
    }

    void
    lstm_forward(const Tensor<2,dim,NumberType> &F,
                                const SymmetricTensor<2,dim,NumberType> &b, const int &flag)
    {
        material->lstm_forward(F,b,flag);
    }

    SymmetricTensor<2,dim,NumberType>
    get_cauchy_baseML() const
    {
      return material->get_cauchy_baseML();
    }  

    void
    update_damage(const Tensor<2,dim,NumberType> &F)
    {
        material->update_damage(F);
    }    

    // And the tangent:
    SymmetricTensor<4,dim,NumberType>
    get_Jc(const Tensor<2,dim,NumberType>          &F,
           const SymmetricTensor<2,dim,NumberType> &sigma_cauchy,
           const NumberType                        &det_F) const
    {
      return material->get_Jc(F,sigma_cauchy,det_F);
    }

   // We offer an interface to retrieve certain data.
    // This is the strain energy:
    NumberType
    get_Psi(const NumberType                        &det_F,
            const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return material->get_Psi(det_F,b_bar);
    }

    // Here are the kinetic variables. These are used in the material and
    // global tangent matrix and residual assembly operations:
    // First is the Kirchhoff stress:
    SymmetricTensor<2,dim,NumberType>
    get_tau(const NumberType                        &det_F,
            const SymmetricTensor<2,dim,NumberType> &b_bar) const
    {
      return material->get_tau(det_F,b_bar);
    }

    // And the tangent:
    SymmetricTensor<4,dim,NumberType>
    get_Jc_const(const NumberType                        &det_F,
           const SymmetricTensor<2,dim,NumberType> &b_bar, const Tensor<2,dim,NumberType> &F) const
    {
      return material->get_Jc_const(det_F,b_bar,F);
    }



    SymmetricTensor<2,dim,NumberType> old_stress;

    // In terms of member functions, this class stores for the quadrature
    // point it represents a copy of a material type in case different
    // materials are used in different regions of the domain, as well as the
    // inverse of the deformation gradient...
  private:
    std::shared_ptr< Material_Compressible_Network<dim,NumberType> > material;
  };


// @sect3{Quasi-static compressible finite-strain solid}

  // Forward declarations for classes that will
  // perform assembly of the linear system.
  template <int dim,typename NumberType>
  struct Assembler_Base;
  template <int dim,typename NumberType>
  struct Assembler;

    // Class for output but does not do a good job actually !

  template <int dim>
  class Postprocessor  : public DataPostprocessor<dim>
  {
  public:
    Postprocessor ();

    virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
    std::vector<Vector<double> > &computed_quantities) const override;
    virtual std::vector<std::string>
    get_names() const override;
    virtual std::vector<
    DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;
    virtual UpdateFlags
    get_needed_update_flags() const override;
  };
  
  template <int dim>
  Postprocessor<dim>::Postprocessor()
  {}
 
  template <int dim>
  void
  Postprocessor<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &input_data,
    std::vector<Vector<double>> &               computed_quantities) const
  {
    const unsigned int n_evaluation_points = input_data.solution_values.size();

    Assert(n_evaluation_points == input_data.solution_gradients.size(),
           ExcInternalError());
    Assert(computed_quantities.size() == n_evaluation_points,
           ExcInternalError());
    Assert(input_data.solution_values[0].size() == dim, ExcInternalError());

    Assert(computed_quantities[0].size() == ((dim * dim) + dim),
           ExcInternalError());


    for (unsigned int p = 0; p < n_evaluation_points; ++p)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            computed_quantities[p][d] = input_data.solution_values[p][d];
            for (unsigned int e = 0; e < dim; ++e)
              computed_quantities[p]
                                 [Tensor<2, dim>::component_to_unrolled_index(
                                    TableIndices<2>(d, e))] =
                                   (input_data.solution_gradients[p][d][e] +
                                    input_data.solution_gradients[p][e][d]) /
                                   2;
          }
      }


  }

  template <int dim>
  std::vector<std::string>
  Postprocessor<dim>::get_names() const
  {
    std::vector<std::string> names;
    for (unsigned int d = 0; d < dim; ++d)
      names.emplace_back("displacement_calc");

    static const char suffixes[] = {'x', 'y', 'z'};

    const std::string strain_name = "strain";
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        names.emplace_back(strain_name + '_' + suffixes[d] + suffixes[e]);

    return names;
  }

  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  Postprocessor<dim>::get_data_component_interpretation() const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation;
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int d = 0; d < (dim * dim); ++d)
      interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }

  template <int dim>
  UpdateFlags
  Postprocessor<dim>::get_needed_update_flags() const
  {
    return update_values | update_gradients;
  }

// The Solid class is the central class in that it represents the problem at
// hand. It follows the usual scheme in that all it really has is a
// constructor, destructor and a <code>run()</code> function that dispatches
// all the work to private functions of this class:
  template <int dim,typename NumberType>
  class Solid
  {
  public:
    Solid(const Parameters::AllParameters &parameters);

    virtual
    ~Solid();

    void
    run();

  private:
    struct PerTaskData_UQPH;
    struct ScratchData_UQPH;

    std::ofstream pointfile;

    // We start the collection of member functions with one that builds the
    // grid:
    void
    make_grid();

    // Set up the finite element system to be solved:
    void
    system_setup();

    // Several functions to assemble the system and right hand side matrices
    // using multithreading. Each of them comes as a wrapper function, one
    // that is executed to do the work in the WorkStream model on one cell,
    // and one that copies the work done on this one cell into the global
    // object that represents it:
    void
    assemble_system(const BlockVector<double> &solution_delta);

    // We use a separate data structure to perform the assembly. It needs access
    // to some low-level data, so we simply befriend the class instead of
    // creating a complex interface to provide access as necessary.
    friend struct Assembler_Base<dim,NumberType>;
    friend struct Assembler<dim,NumberType>;

    // Apply Dirichlet boundary conditions on the displacement field
    void
    make_constraints(const int &it_nr);

    // Create and update the quadrature points. Here, no data needs to be
    // copied into a global object, so the copy_local_to_global function is
    // empty:
    void
    setup_qph();

    void update_qph_incremental(const BlockVector<double> &solution_delta);
    void update_qph_incremental_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_UQPH &                                    scratch,
      PerTaskData_UQPH &                                    data);    

    void copy_local_to_global_UQPH(const PerTaskData_UQPH & /*data*/)
    {}    

    // Solve for the displacement using a Newton-Raphson method. We break this
    // function into the nonlinear loop and the function that solves the
    // linearized Newton-Raphson step:
    void
    solve_nonlinear_timestep(BlockVector<double> &solution_delta);

    std::pair<unsigned int, double>
    solve_linear_system(BlockVector<double> &newton_update);

    // Store the converged values of the internal variables at the end of each timestep
    void update_end_timestep();

    // Solution retrieval as well as post-processing and writing data to file:
    BlockVector<double>
    get_total_solution(const BlockVector<double> &solution_delta) const;

    //void define_tracked_vertices(std::vector<Point<dim> > &tracked_vertices);
    void
    output_results(BlockVector<double> &solution) const;


    void output_results_to_plot(
                                const double current_time,
                                BlockVector<double> &solution_delta,
                                std::vector<Point<dim> > &tracked_vertices,
                                std::ofstream &pointfile) const;    
    void print_plot_file_header(std::vector<Point<dim> > &tracked_vertices,
                                std::ofstream &pointfile) const;
    void print_plot_file_footer( std::ofstream &pointfile) const;     

    void calculate_reaction_force(
                                  BlockVector<double> &solution_delta,
                                  std::vector<Point<dim> > &tracked_vertices,
                                  double* add_force, double* add_displacement) const;                       
    // Finally, some member variables that describe the current state: A
    // collection of the parameters used to describe the problem setup...
    const Parameters::AllParameters &parameters;

    //For parallel communication
    MPI_Comm                         mpi_communicator;
    const unsigned int               n_mpi_processes;
    const unsigned int               this_mpi_process;
    mutable ConditionalOStream       pcout;

    // ...the volume of the reference and current configurations...
    double                           vol_reference;
    double                           vol_current;
    int                              run_t;
    int loading;
    int unloading;
    int current_cycle;
    int cycles;

    // ...and description of the geometry on which the problem is solved:
    parallel::shared::Triangulation<dim>  triangulation;

    // Also, keep track of the current time and the time spent evaluating
    // certain functions
    Time                             time;
    TimerOutput                      timer;

    // A storage object for quadrature point information. As opposed to
    // step-18, deal.II's native quadrature point data manager is employed here.
    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim,NumberType> > quadrature_point_history;

    // A description of the finite-element system including the displacement
    // polynomial degree, the degree-of-freedom handler, number of DoFs per
    // cell and the extractor objects used to retrieve information from the
    // solution vectors:
    const unsigned int               degree;
    const FESystem<dim>              fe;
    DoFHandler<dim>                  dof_handler_ref;
    const unsigned int               dofs_per_cell;
    const FEValuesExtractors::Vector u_fe;

    // Description of how the block-system is arranged. There is just 1 block,
    // that contains a vector DOF $\mathbf{u}$.
    // There are two reasons that we retain the block system in this problem.
    // The first is pure laziness to perform further modifications to the
    // code from which this work originated. The second is that a block system
    // would typically necessary when extending this code to multiphysics
    // problems.
    static const unsigned int        n_blocks = 1;
    static const unsigned int        n_components = dim;
    static const unsigned int        first_u_component = 0;

    enum
    {
      u_dof = 0
    };

    std::vector<types::global_dof_index>  dofs_per_block;

    // Rules for Gauss-quadrature on both the cell and faces. The number of
    // quadrature points on both cells and faces is recorded.
    const QGauss<dim>                qf_cell;
    const QGauss<dim - 1>            qf_face;
    const unsigned int               n_q_points;
    const unsigned int               n_q_points_f;

    // Objects that store the converged solution and right-hand side vectors,
    // as well as the tangent matrix. There is a AffineConstraints object used
    // to keep track of constraints.  We make use of a sparsity pattern
    // designed for a block system.
    AffineConstraints<double>        constraints;
    BlockSparsityPattern             sparsity_pattern;
    BlockSparseMatrix<double>        tangent_matrix;
    BlockVector<double>              system_rhs;
    BlockVector<double>              solution_n;

    std::ofstream myfile;
    
    // Then define a number of variables to store norms and update norms and
    // normalisation factors.
    struct Errors
    {
      Errors()
        :
        norm(1.0), u(1.0)
      {}

      void reset()
      {
        norm = 1.0;
        u = 1.0;
      }
      void normalise(const Errors &rhs)
      {
        if (rhs.norm != 0.0)
          norm /= rhs.norm;
        if (rhs.u != 0.0)
          u /= rhs.u;
      }

      double norm, u;
    };

    Errors error_residual, error_residual_0, error_residual_norm, error_update,
           error_update_0, error_update_norm;

    // Methods to calculate error measures
    void
    get_error_residual(Errors &error_residual);

    void
    get_error_update(const BlockVector<double> &newton_update,
                     Errors &error_update);

    // Print information to screen in a pleasing way...
    static
    void
    print_conv_header();

    void
    print_conv_footer();

    void
    print_vertical_tip_displacement();
  };

// @sect3{Implementation of the <code>Solid</code> class}

// @sect4{Public interface}

// We initialise the Solid class using data extracted from the parameter file.
  template <int dim,typename NumberType>
  Solid<dim,NumberType>::Solid(const Parameters::AllParameters &parameters)
    :
    parameters(parameters),
    mpi_communicator(MPI_COMM_WORLD),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout(myfile, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
    vol_reference (0.0),
    vol_current (0.0),
    run_t(0),
    loading(1),
    unloading(0),
    current_cycle(0),
    cycles(7),
    triangulation(mpi_communicator,Triangulation<dim>::maximum_smoothing),
    time(parameters.end_time, parameters.delta_t_1, parameters.delta_t_2,
          parameters.delta_de,parameters.load_rate),
    timer(MPI_COMM_WORLD,
          pcout,
          TimerOutput::every_call_and_summary,
          TimerOutput::cpu_and_wall_times),
    degree(parameters.poly_degree),
    // The Finite Element System is composed of dim continuous displacement
    // DOFs.
    fe(FE_Q<dim>(parameters.poly_degree), dim), // displacement
    dof_handler_ref(triangulation),
    dofs_per_cell (fe.dofs_per_cell),
    u_fe(first_u_component),
    dofs_per_block(n_blocks),
    qf_cell(parameters.quad_order),
    qf_face(parameters.quad_order),
    n_q_points (qf_cell.size()),
    n_q_points_f (qf_face.size())
  {

  }

// The class destructor simply clears the data held by the DOFHandler
  template <int dim,typename NumberType>
  Solid<dim,NumberType>::~Solid()
  {
    dof_handler_ref.clear();
  }


// In solving the quasi-static problem, the time becomes a loading parameter,
// i.e. we increasing the loading linearly with time, making the two concepts
// interchangeable. We choose to increment time linearly using a constant time
// step size.
//
// We start the function with preprocessing, and then output the initial grid
// before starting the simulation proper with the first time (and loading)
// increment.
//
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::run()
  {
    make_grid();
    system_setup();
    //Define points for post-processing
    std::vector<Point<dim> > tracked_vertices (2);
    tracked_vertices[0][0] = 61.0;
    tracked_vertices[0][1] = 0.0;
    tracked_vertices[0][2] = 0.5;

    tracked_vertices[1][0] = 61.0;
    tracked_vertices[1][1] = 10.0;
    tracked_vertices[1][2] = 0.0;
    myfile.open ("data_times.txt"); // Use the text file to print out CPU- and Wall time for assembly



    //define_tracked_vertices(tracked_vertices);
    std::vector<Point<dim>> reaction_force;

    output_results(solution_n);
    if (this_mpi_process == 0)
    {
      pointfile.open("data-for-gnuplot.sol"); // Print out force and displacement at specific vertices
      print_plot_file_header(tracked_vertices, pointfile);
    }
    output_results_to_plot(time.current(),
                            solution_n,
                            tracked_vertices,
                            pointfile);    
    time.increment();
    BlockVector<double> solution_delta(dofs_per_block);
    run_t = 1;
    loading = 1;
    unloading = 0;
    current_cycle = 0;
    cycles = parameters.total_cycles; //
    int change = 0;
    while (run_t == 1)
      {
        solution_delta = 0.0;
        if(time.get_timestep() > parameters.intRedDe && change == 0)
        {
          double current_de = time.delta_de;
          double new_de = current_de*parameters.RedAmount;
          time.delta_de = new_de;
          time.delta_t = time.delta_de/time.load_rate;
              if (this_mpi_process == 0)
              {
                printf("\n-------------------------------------------------------\n");
                std::cout<<"Changing timestep to: "<<time.get_delta_t()<<std::endl;
                std::cout<<"\nNew delta u: "<<time.delta_de<<std::endl;
                printf("\n-------------------------------------------------------\n");
              }
          change = 1;
        }

        // ...solve the current time step and update total solution vector
        // $\mathbf{\Xi}_{\textrm{n}} = \mathbf{\Xi}_{\textrm{n-1}} +
        // \varDelta \mathbf{\Xi}$...
        solve_nonlinear_timestep(solution_delta);
        solution_n += solution_delta;
        // Store the converged values of the internal variables
        //update_end_timestep();
        update_qph_incremental(solution_delta);
        // ...and plot the results before moving on happily to the next time
        // step:
        if ( (time.get_timestep()%parameters.timestep_output) == 0 )
            {
              output_results(solution_n);
            }
              output_results_to_plot(time.current(),
                                     solution_n,
                                    tracked_vertices,
                                     pointfile);            
        time.increment();
      }
    if (this_mpi_process == 0)
    {
      print_plot_file_footer(pointfile);
      pointfile.close ();
    }

    myfile.close();
  }

  template <int dim,typename NumberType>
  struct Solid<dim,NumberType>::PerTaskData_UQPH
  {
    void reset()
    {}
  };


  template <int dim,typename NumberType>
  struct Solid<dim,NumberType>::ScratchData_UQPH
  {
    const BlockVector<double> &solution_total;
 
    std::vector<Tensor<2, dim>> solution_grads_u_total;
 
    FEValues<dim> fe_values;
 
    ScratchData_UQPH(const FiniteElement<dim> & fe_cell,
                     const QGauss<dim> &        qf_cell,
                     const UpdateFlags          uf_cell,
                     const BlockVector<double> &solution_total)
      : solution_total(solution_total)
      , solution_grads_u_total(qf_cell.size())
      , fe_values(fe_cell, qf_cell, uf_cell)
    {}
 
    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      : solution_total(rhs.solution_total)
      , solution_grads_u_total(rhs.solution_grads_u_total)
      , fe_values(rhs.fe_values.get_fe(),
                  rhs.fe_values.get_quadrature(),
                  rhs.fe_values.get_update_flags())
    {}
 
    void reset()
    {
      const unsigned int n_q_points = solution_grads_u_total.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          solution_grads_u_total[q]  = 0.0;
        }
    }
  };



  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::make_grid()
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    //Here is the mesh imported
    std::ifstream f("t3_3D_coarse.msh");
    grid_in.read_msh(f);

    // Since we wish to apply a Dirichlet BC, we
    // must find the cell faces in this part of the domain and mark them with
    // a distinct boundary ID number.  The faces we are looking for are on the
    // +x surface and will get boundary ID 11.
    // Dirichlet boundaries exist on the left-hand face of the beam also (this fixed
    // boundary will get ID 1) and on the +Z and -Z faces (which correspond to
    // ID 2 and we will use to impose the plane strain condition but we jusst mark them here. Not needed for 3D)
    const double tol_boundary = 1e-4;
    typename Triangulation<dim>::active_cell_iterator cell =
      triangulation.begin_active(), endc = triangulation.end();
    for (; cell != endc; ++cell)
      for (unsigned int face = 0;
           face < GeometryInfo<dim>::faces_per_cell; ++face)
        if (cell->face(face)->at_boundary() == true)
          {
            if (std::abs(cell->face(face)->center()[0] - 61.0) < tol_boundary)    
                cell->face(face)->set_all_boundary_ids(10); // +X faces
            else if (std::abs(cell->face(face)->center()[1] - 10.0) < tol_boundary)
              cell->face(face)->set_all_boundary_ids(13); // +Y faces
            else if (std::abs(cell->face(face)->center()[0] - 115.0) < tol_boundary)
              cell->face(face)->set_all_boundary_ids(11); // -X faces  
            else if (dim == 3 && std::abs(std::abs(cell->face(face)->center()[2]) - 2.0) < tol_boundary)
              cell->face(face)->set_all_boundary_ids(12); // +Z and -Z faces
            else if (dim == 3 && std::abs(std::abs(cell->face(face)->center()[2]) - 0.0) < tol_boundary)
              cell->face(face)->set_boundary_id(12); // +Z and -Z faces                 
          };

    vol_reference = GridTools::volume(triangulation);
    vol_current = vol_reference;
    std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;

    std::ofstream out("grid-1.msh");
    GridOut       grid_out;
    grid_out.write_msh(triangulation,out);
    std::cout << "Grid written to grid-1.msh" << std::endl; 
  }


// @sect4{Solid::system_setup}

// Next we describe how the FE system is setup.  We first determine the number
// of components per block. Since the displacement is a vector component, the
// first dim components belong to it.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::system_setup()
  {

    std::vector<unsigned int> block_component(n_components, u_dof); // Displacement

    // The DOF handler is then initialised and we renumber the grid in an
    // efficient manner. We also record the number of DOFs per block.
    dof_handler_ref.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler_ref);
    DoFRenumbering::component_wise(dof_handler_ref, block_component);
    dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler_ref, block_component);

    std::cout << "Triangulation:"
              << "\n\t Number of active cells: " << triangulation.n_active_cells()
              << "\n\t Number of degrees of freedom: " << dof_handler_ref.n_dofs()
              << std::endl;

    // Setup the sparsity pattern and tangent matrix
    tangent_matrix.clear();
    {
      const types::global_dof_index n_dofs_u = dofs_per_block[u_dof];

      BlockDynamicSparsityPattern csp(n_blocks, n_blocks);

      csp.block(u_dof, u_dof).reinit(n_dofs_u, n_dofs_u);
      csp.collect_sizes();

      // Naturally, for a one-field vector-valued problem, all of the
      // components of the system are coupled.
      Table<2, DoFTools::Coupling> coupling(n_components, n_components);
      for (unsigned int ii = 0; ii < n_components; ++ii)
        for (unsigned int jj = 0; jj < n_components; ++jj)
          coupling[ii][jj] = DoFTools::always;
      DoFTools::make_sparsity_pattern(dof_handler_ref,
                                      coupling,
                                      csp,
                                      constraints,
                                      false);
      sparsity_pattern.copy_from(csp);
    }

    tangent_matrix.reinit(sparsity_pattern);

    // We then set up storage vectors
    system_rhs.reinit(dofs_per_block);
    system_rhs.collect_sizes();

    solution_n.reinit(dofs_per_block);
    solution_n.collect_sizes();

    // ...and finally set up the quadrature
    // point history:
    setup_qph();

  }


// @sect4{Solid::setup_qph}
// The method used to store quadrature information is already described in
// step-18 and step-44. Here we implement a similar setup for a SMP machine.
//
// Firstly the actual QPH data objects are created. This must be done only
// once the grid is refined to its finest level.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::setup_qph()
  {
    std::cout << "    Setting up quadrature point data..." << std::endl;

    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(),
                                        n_q_points);

    // Next we setup the initial quadrature point data. Note that when
    // the quadrature point data is retrieved, it is returned as a vector
    // of smart pointers.
    for (typename Triangulation<dim>::active_cell_iterator cell =
           triangulation.begin_active(); cell != triangulation.end(); ++cell)
      {
        const std::vector<std::shared_ptr<PointHistory<dim,NumberType> > > lqph =
          quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(parameters,time);
      }
  }

   template <int dim,typename NumberType>
  void
  Solid<dim,NumberType>::update_qph_incremental(const BlockVector<double> &solution_delta)
  {
    timer.enter_subsection("Update QPH data");
    std::cout << " UQPH " << std::flush;
  
    const BlockVector<double> solution_total(
      get_total_solution(solution_delta));
  
    const UpdateFlags uf_UQPH(update_values | update_gradients);
    PerTaskData_UQPH  per_task_data_UQPH;
    ScratchData_UQPH  scratch_data_UQPH(fe, qf_cell, uf_UQPH, solution_total);
    WorkStream::run(dof_handler_ref.active_cell_iterators(),
                  *this,
                  &Solid::update_qph_incremental_one_cell,
                  &Solid::copy_local_to_global_UQPH,
                  scratch_data_UQPH,
                  per_task_data_UQPH);
 
    timer.leave_subsection();
  } 

  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::update_qph_incremental_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_UQPH &                                    scratch,
    PerTaskData_UQPH & /*data*/)
  {
    const std::vector<std::shared_ptr<PointHistory<dim,NumberType>>> lqph =
      quadrature_point_history.get_data(cell);
    Assert(lqph.size() == n_q_points, ExcInternalError());
  
    Assert(scratch.solution_grads_u_total.size() == n_q_points,
          ExcInternalError());
  
    scratch.reset();
    scratch.fe_values.reinit(cell);
    scratch.fe_values[u_fe].get_function_gradients(
    scratch.solution_total, scratch.solution_grads_u_total);
 
  for (const unsigned int q_point :
       scratch.fe_values.quadrature_point_indices())
    lqph[q_point]->update_values(scratch.solution_grads_u_total[q_point],parameters,time);
  } 

// The next function is the driver method for the Newton-Raphson scheme. At
// its top we create a new vector to store the current Newton update step,
// reset the error storage objects and print solver header.
  template <int dim,typename NumberType>
  void
  Solid<dim,NumberType>::solve_nonlinear_timestep(BlockVector<double> &solution_delta)
  {
    //time.adjust_timestep_size(1);
    std::cout << std::endl << "Timestep " << time.get_timestep() << " @ "
              << time.current() << " s " << time.get_delta_t() << " delta t" << std::endl;
    //time.adjust_timestep_size(1);
    BlockVector<double> newton_update(dofs_per_block);

    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_conv_header();

    // We now perform a number of Newton iterations to iteratively solve the
    // nonlinear problem.  Since the problem is fully nonlinear and we are
    // using a full Newton method, the data stored in the tangent matrix and
    // right-hand side vector is not reusable and must be cleared at each
    // Newton step.  We then initially build the right-hand side vector to
    // check for convergence (and store this value in the first iteration).
    // The unconstrained DOFs of the rhs vector hold the out-of-balance
    // forces. The building is done before assembling the system matrix as the
    // latter is an expensive operation and we can potentially avoid an extra
    // assembly process by not assembling the tangent matrix when convergence
    // is attained.
    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.max_iterations_NR;
         ++newton_iteration)
      {
        std::cout << " " << std::setw(2) << newton_iteration << " " << std::flush;

        // If we have decided that we want to continue with the iteration, we
        // assemble the tangent, make and impose the Dirichlet constraints,
        // and do the solve of the linearized system:
        make_constraints(newton_iteration);      
        assemble_system(solution_delta);

        get_error_residual(error_residual);

        if (newton_iteration == 0)
          error_residual_0 = error_residual;

        // We can now determine the normalised residual error and check for
        // solution convergence:
        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);

        if (newton_iteration > 0 && error_update_norm.u <= parameters.tol_u
            && error_residual_norm.u <= parameters.tol_f)
          {
            std::cout << " CONVERGED! " << std::endl;
            print_conv_footer();

            break;
          }

        const std::pair<unsigned int, double>
        lin_solver_output = solve_linear_system(newton_update);

        get_error_update(newton_update, error_update);
        if (newton_iteration == 0)
          error_update_0 = error_update;

        // We can now determine the normalised Newton update error, and
        // perform the actual update of the solution increment for the current
        // time step, update all quadrature point information pertaining to
        // this new displacement and stress state and continue iterating:
        error_update_norm = error_update;
        error_update_norm.normalise(error_update_0);
        solution_delta += newton_update;

         // THIS part is commented but can be used if you want to plot after each iteration !!
/*              DataOut<dim> data_out;
             std::vector<DataComponentInterpretation::DataComponentInterpretation>
             data_component_interpretation(dim,
                                  DataComponentInterpretation::component_is_part_of_vector);

            std::vector<std::string> solution_name(dim, "displacement");

            data_out.attach_dof_handler(dof_handler_ref);
            data_out.add_data_vector(solution_delta,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
            Vector<double> soln(solution_delta.size());
            for (unsigned int i = 0; i < soln.size(); ++i)
              soln(i) = solution_delta(i);
            MappingQEulerian<dim> q_mapping(degree, dof_handler_ref, soln);
            data_out.build_patches(q_mapping, degree);
            const char *path = "/bigwork/nhgebaht/FEM/Betim_model/output/";
            std::ostringstream filename;
            filename << path <<  "solution-" << newton_iteration << ".vtk";

            std::ofstream output(filename.str().c_str());
            data_out.write_vtk(output);     */   

          // up to here          

        std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
                  << std::scientific << lin_solver_output.first << "  "
                  << lin_solver_output.second << "  " << error_residual_norm.norm
                  << "  " << error_residual_norm.u << "  "
                  << "  " << error_update_norm.norm << "  " << error_update_norm.u
                  << "  " << std::endl;
      }

    // At the end, if it turns out that we have in fact done more iterations
    // than the parameter file allowed, we raise an exception that can be
    // caught in the main() function. The call <code>AssertThrow(condition,
    // exc_object)</code> is in essence equivalent to <code>if (!cond) throw
    // exc_object;</code> but the former form fills certain fields in the
    // exception object that identify the location (filename and line number)
    // where the exception was raised to make it simpler to identify where the
    // problem happened.
    AssertThrow (newton_iteration <= parameters.max_iterations_NR,
                 ExcMessage("No convergence in nonlinear solver!"));
  }



// This program prints out data in a nice table that is updated
// on a per-iteration basis. The next two functions set up the table
// header and footer:
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::print_conv_header()
  {
    static const unsigned int l_width = 87;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;

    std::cout << "    SOLVER STEP    "
              << " |  LIN_IT   LIN_RES    RES_NORM    "
              << " RES_U     NU_NORM     "
              << " NU_U " << std::endl;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;
  }



  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::print_conv_footer()
  {
    static const unsigned int l_width = 87;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;

    std::cout << "Relative errors:" << std::endl
              << "Displacement:\t" << error_update.u / error_update_0.u << std::endl
              << "Force: \t\t" << error_residual.u / error_residual_0.u << std::endl
              << "v / V_0:\t" << vol_current << " / " << vol_reference
              << std::endl;
  }

// Determine the true residual error for the problem.  That is, determine the
// error in the residual for the unconstrained degrees of freedom.  Note that to
// do so, we need to ignore constrained DOFs by setting the residual in these
// vector components to zero.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::get_error_residual(Errors &error_residual)
  {
    BlockVector<double> error_res(dofs_per_block);

    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_res(i) = system_rhs(i);

    error_residual.norm = error_res.l2_norm();
    error_residual.u = error_res.block(u_dof).l2_norm();
  }


// @sect4{Solid::get_error_udpate}

// Determine the true Newton update error for the problem
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::get_error_update(const BlockVector<double> &newton_update,
                                               Errors &error_update)
  {
    BlockVector<double> error_ud(dofs_per_block);
    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_ud(i) = newton_update(i);

    error_update.norm = error_ud.l2_norm();
    error_update.u = error_ud.block(u_dof).l2_norm();
  }



// @sect4{Solid::get_total_solution}

// This function provides the total solution, which is valid at any Newton step.
// This is required as, to reduce computational error, the total solution is
// only updated at the end of the timestep.
  template <int dim,typename NumberType>
  BlockVector<double>
  Solid<dim,NumberType>::get_total_solution(const BlockVector<double> &solution_delta) const
  {
    BlockVector<double> solution_total(solution_n);
    solution_total += solution_delta;
    return solution_total;
  }


// @sect4{Solid::assemble_system}

  template <int dim,typename NumberType>
  struct Assembler_Base
  {
    virtual ~Assembler_Base() {}

    // Here we deal with the tangent matrix assembly structures. The
    // PerTaskData object stores local contributions.
    struct PerTaskData_ASM
    {
      const Solid<dim,NumberType>          *solid;
      FullMatrix<double>                   cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> local_dof_indices;

      PerTaskData_ASM(const Solid<dim,NumberType> *solid)
        :
        solid (solid),
        cell_matrix(solid->dofs_per_cell, solid->dofs_per_cell),
        cell_rhs(solid->dofs_per_cell),
        local_dof_indices(solid->dofs_per_cell)
      {}

      void reset()
      {
        cell_matrix = 0.0;
        cell_rhs = 0.0;
      }
    };

    // On the other hand, the ScratchData object stores the larger objects such as
    // the shape-function values array (<code>Nx</code>) and a shape function
    // gradient and symmetric gradient vector which we will use during the
    // assembly.
    struct ScratchData_ASM
    {
      const BlockVector<double>               &solution_total;
      std::vector<Tensor<2, dim,NumberType> >  solution_grads_u_total;

      FEValues<dim>                fe_values_ref;
      FEFaceValues<dim>            fe_face_values_ref;

      std::vector<std::vector<Tensor<2, dim,NumberType> > >         grad_Nx;
      std::vector<std::vector<SymmetricTensor<2,dim,NumberType> > > symm_grad_Nx;

      ScratchData_ASM(const FiniteElement<dim> &fe_cell,
                      const QGauss<dim> &qf_cell,
                      const UpdateFlags uf_cell,
                      const QGauss<dim-1> & qf_face,
                      const UpdateFlags uf_face,
                      const BlockVector<double> &solution_total)
        :
        solution_total(solution_total),
        solution_grads_u_total(qf_cell.size()),
        fe_values_ref(fe_cell, qf_cell, uf_cell),
        fe_face_values_ref(fe_cell, qf_face, uf_face),
        grad_Nx(qf_cell.size(),
                std::vector<Tensor<2,dim,NumberType> >(fe_cell.dofs_per_cell)),
        symm_grad_Nx(qf_cell.size(),
                     std::vector<SymmetricTensor<2,dim,NumberType> >
                     (fe_cell.dofs_per_cell))
      {}

      ScratchData_ASM(const ScratchData_ASM &rhs)
        :
        solution_total (rhs.solution_total),
        solution_grads_u_total(rhs.solution_grads_u_total),
        fe_values_ref(rhs.fe_values_ref.get_fe(),
                      rhs.fe_values_ref.get_quadrature(),
                      rhs.fe_values_ref.get_update_flags()),
        fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                           rhs.fe_face_values_ref.get_quadrature(),
                           rhs.fe_face_values_ref.get_update_flags()),
        grad_Nx(rhs.grad_Nx),
        symm_grad_Nx(rhs.symm_grad_Nx)
      {}

      void reset()
      {
        const unsigned int n_q_points = fe_values_ref.get_quadrature().size();
        const unsigned int n_dofs_per_cell = fe_values_ref.dofs_per_cell;
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            Assert( grad_Nx[q_point].size() == n_dofs_per_cell,
                    ExcInternalError());
            Assert( symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                    ExcInternalError());

            solution_grads_u_total[q_point] = Tensor<2,dim,NumberType>();
            for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
              {
                grad_Nx[q_point][k] = Tensor<2,dim,NumberType>();
                symm_grad_Nx[q_point][k] = SymmetricTensor<2,dim,NumberType>();
              }
          }
      }

    };

    // Of course, we still have to define how we assemble the tangent matrix
    // contribution for a single cell.
    void
    assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                             ScratchData_ASM &scratch,
                             PerTaskData_ASM &data)
    {
      // Due to the C++ specialization rules, we need one more
      // level of indirection in order to define the assembly
      // routine for all different number. The next function call
      // is specialized for each NumberType, but to prevent having
      // to specialize the whole class along with it we have inlined
      // the definition of the other functions that are common to
      // all implementations.
      assemble_system_tangent_residual_one_cell(cell, scratch, data);
      //assemble_neumann_contribution_one_cell(cell, scratch, data);
    }

    // This function adds the local contribution to the system matrix.
    void
    copy_local_to_global_ASM(const PerTaskData_ASM &data)
    {
      const AffineConstraints<double> &constraints = data.solid->constraints;
      BlockSparseMatrix<double> &tangent_matrix = const_cast<Solid<dim,NumberType> *>(data.solid)->tangent_matrix;
      BlockVector<double> &system_rhs =  const_cast<Solid<dim,NumberType> *>(data.solid)->system_rhs;

      constraints.distribute_local_to_global(
        data.cell_matrix, data.cell_rhs,
        data.local_dof_indices,
        tangent_matrix, system_rhs);
    }

  protected:

    // This function needs to exist in the base class for
    // Workstream to work with a reference to the base class.
    virtual void
    assemble_system_tangent_residual_one_cell(const typename DoFHandler<dim>::active_cell_iterator &/*cell*/,
                                              ScratchData_ASM &/*scratch*/,
                                              PerTaskData_ASM &/*data*/)
    {
      AssertThrow(false, ExcPureFunctionCalled());
    }

/*
    void
    assemble_neumann_contribution_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                           ScratchData_ASM &scratch,
                                           PerTaskData_ASM &data)
    {
    }
*/
  };
  template <int dim>
  struct Assembler<dim,double> : Assembler_Base<dim,double>
  {
    typedef double NumberType;
    using typename Assembler_Base<dim,NumberType>::ScratchData_ASM;
    using typename Assembler_Base<dim,NumberType>::PerTaskData_ASM;

    virtual ~Assembler() {}

    virtual void
    assemble_system_tangent_residual_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                              ScratchData_ASM &scratch,
                                              PerTaskData_ASM &data)
    {
      // Aliases for data referenced from the Solid class
      const unsigned int &n_q_points = data.solid->n_q_points;
      const unsigned int &dofs_per_cell = data.solid->dofs_per_cell;
      const FESystem<dim> &fe = data.solid->fe;
      const unsigned int &u_dof = data.solid->u_dof;
      const FEValuesExtractors::Vector &u_fe = data.solid->u_fe;
      const unsigned int &timestep = data.solid->time.get_timestep();
      const unsigned int &changetoML = data.solid->parameters.intToML;
      std::string switchtoML = data.solid->parameters.switchML;
      data.reset();
      scratch.reset();
      scratch.fe_values_ref.reinit(cell);
      cell->get_dof_indices(data.local_dof_indices);
      const std::vector<std::shared_ptr<const PointHistory<dim,NumberType> > > lqph =
        const_cast<const Solid<dim,NumberType> *>(data.solid)->quadrature_point_history.get_data(cell);
      Assert(lqph.size() == n_q_points, ExcInternalError());

      // We first need to find the solution gradients at quadrature points
      // inside the current cell and then we update each local QP using the
      // displacement gradient:
      scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total,
                                                         scratch.solution_grads_u_total);
      // Now we build the local cell stiffness matrix. Since the global and
      // local system matrices are symmetric, we can exploit this property by
      // building only the lower half of the local matrix and copying the values
      // to the upper half.
      //
      // In doing so, we first extract some configuration dependent variables
      // from our QPH history objects for the current quadrature point.
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          const Tensor<2,dim,NumberType> &grad_u = scratch.solution_grads_u_total[q_point];
          const Tensor<2,dim,NumberType> F = Physics::Elasticity::Kinematics::F(grad_u);
          const NumberType               det_F = determinant(F);
          const Tensor<2,dim,NumberType> Fbar = Physics::Elasticity::Kinematics::F_iso(F);
          const SymmetricTensor<2,dim,NumberType> bstar = Physics::Elasticity::Kinematics::b(Fbar);
          const SymmetricTensor<2,dim,NumberType> b = Physics::Elasticity::Kinematics::b(F);
          //const SymmetricTensor<2,dim,NumberType> b_bar = Physics::Elasticity::Kinematics::b(F_bar);
          const Tensor<2,dim,NumberType> F_inv = invert(F);
          PointHistory<dim,NumberType> *lqph_q_point_nc =
              const_cast<PointHistory<dim,NumberType>*>(lqph[q_point].get());
          SymmetricTensor<2,dim,NumberType> sigma;
          if (timestep < changetoML)  
           {
              lqph_q_point_nc->update_internal_equilibrium(Fbar,det_F,bstar,1);
              if(switchtoML == "On" )
                lqph_q_point_nc->lstm_forward(F,b,1);

              sigma = lqph[q_point]->get_cauchy_base();
            } 
          else if(timestep >= changetoML)
           {
              lqph_q_point_nc->lstm_forward(F,b,1);
              sigma = lqph[q_point]->get_cauchy_baseML();
            }   
          
          //const Tensor<2,dim,NumberType> F_bar = Physics::Elasticity::Kinematics::F_iso(F);
          //const SymmetricTensor<2,dim,NumberType> b_bar = Physics::Elasticity::Kinematics::b(F_bar);
          Assert(det_F > NumberType(0.0), ExcInternalError());

          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              const unsigned int k_group = fe.system_to_base_index(k).first.first;

              if (k_group == u_dof)
                {
                  scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
                  scratch.symm_grad_Nx[q_point][k] = symmetrize(scratch.grad_Nx[q_point][k]);
                }
              else
                Assert(k_group <= u_dof, ExcInternalError());
            }
          //const SymmetricTensor<2,dim,NumberType> tau = lqph[q_point]->get_tau(det_F,b_bar);
          //const SymmetricTensor<2,dim,NumberType> sigma = lqph[q_point]->get_cauchy_base(Fbar,det_F,bstar);
          //const SymmetricTensor<2,dim,NumberType> sigma = lqph[q_point]->get_cauchy_baseML();
          //const SymmetricTensor<2,dim,NumberType> S = lqph[q_point]->get_piola_base(F,det_F,C);
          //const SymmetricTensor<2,dim,NumberType> tau = lqph[q_point]->get_tau(det_F,b_bar);
          //const SymmetricTensor<4,dim,NumberType> Jc_const  = lqph[q_point]->get_Jc_const(det_F,b_bar,F);
          const SymmetricTensor<4,dim,NumberType> Jc  = lqph[q_point]->get_Jc(F,sigma,det_F);
          //std::cout<<"Jc: \n"<<Jc<<std::endl;

          const Tensor<2,dim,NumberType> tau_ns (sigma);
          // Next we define some aliases to make the assembly process easier to
          // follow
          const std::vector<SymmetricTensor<2, dim> > &symm_grad_Nx = scratch.symm_grad_Nx[q_point];
          
          const std::vector<Tensor<2, dim> > &grad_Nx = scratch.grad_Nx[q_point];
          const double JxW = scratch.fe_values_ref.JxW(q_point);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i = fe.system_to_component_index(i).first;
              const unsigned int i_group     = fe.system_to_base_index(i).first.first;

              if (i_group == u_dof)
                {//data.cell_rhs(i) -= (symm_grad_Nx[i] * sigma) * JxW;
                  data.cell_rhs(i) -= (symm_grad_Nx[i] * sigma) * JxW;
                }              
              else
                {
                  Assert(i_group <= u_dof, ExcInternalError());
                }
              for (unsigned int j = 0; j <= i; ++j)
                {
                  const unsigned int component_j = fe.system_to_component_index(j).first;
                  const unsigned int j_group     = fe.system_to_base_index(j).first.first;

                  // This is the $\mathsf{\mathbf{k}}_{\mathbf{u} \mathbf{u}}$
                  // contribution. It comprises a material contribution, and a
                  // geometrical stress contribution which is only added along
                  // the local matrix diagonals:
                  if ((i_group == j_group) && (i_group == u_dof))
                    {
                      data.cell_matrix(i, j) += symm_grad_Nx[i] * Jc // The material contribution:
                                                * symm_grad_Nx[j] * JxW;
                      if (component_i == component_j) // geometrical stress contribution
                        data.cell_matrix(i, j) += grad_Nx[i][component_i] * tau_ns
                                                  * grad_Nx[j][component_j] * JxW;
                    }
                  else
                    Assert((i_group <= u_dof) && (j_group <= u_dof),
                           ExcInternalError());
                }
            }
        }


      // Finally, we need to copy the lower half of the local matrix into the
      // upper half:
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
          data.cell_matrix(i, j) = data.cell_matrix(j, i);
    }

  };

    //Store the converged values of the internal variables
    template <int dim,typename NumberType>
    void Solid<dim,NumberType>::update_end_timestep()
    {
          FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
          cell (IteratorFilters::LocallyOwnedCell(),
                dof_handler_ref.begin_active()),
          endc (IteratorFilters::LocallyOwnedCell(),
                dof_handler_ref.end());
          for (; cell!=endc; ++cell)
          {
            Assert(cell->is_locally_owned(), ExcInternalError());
            Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());

            const std::vector<std::shared_ptr<PointHistory<dim,NumberType> > >
                lqph = quadrature_point_history.get_data(cell);
            Assert(lqph.size() == n_q_points, ExcInternalError());
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              lqph[q_point]->update_end_timestep();
          }
    }


// Since we use TBB for assembly, we simply setup a copy of the
// data structures required for the process and pass them, along
// with the memory addresses of the assembly functions to the
// WorkStream object for processing. Note that we must ensure that
// the matrix is reset before any assembly operations can occur.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::assemble_system(const BlockVector<double> &solution_delta)
  {
    timer.enter_subsection("Assemble linear system");
    std::cout << " ASM " << std::flush;

    tangent_matrix = 0.0;
    system_rhs = 0.0;

    const UpdateFlags uf_cell(update_gradients |
                              update_JxW_values);
    const UpdateFlags uf_face(update_values |
                              update_JxW_values);

    const BlockVector<double> solution_total(get_total_solution(solution_delta));
    typename Assembler_Base<dim,NumberType>::PerTaskData_ASM per_task_data(this);
    typename Assembler_Base<dim,NumberType>::ScratchData_ASM scratch_data(fe, qf_cell, uf_cell, qf_face, uf_face, solution_total);
    Assembler<dim,NumberType> assembler;

    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    static_cast<Assembler_Base<dim,NumberType>&>(assembler),
                    &Assembler_Base<dim,NumberType>::assemble_system_one_cell,
                    &Assembler_Base<dim,NumberType>::copy_local_to_global_ASM,
                    scratch_data,
                    per_task_data);

    timer.leave_subsection();
  }


// The constraints for this problem are simple to describe.
// However, since we are dealing with an iterative Newton method,
// it should be noted that any displacement constraints should only
// be specified at the zeroth iteration and subsequently no
// additional contributions are to be made since the constraints
// are already exactly satisfied.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::make_constraints(const int &it_nr)
  {
    std::cout << " CST " << std::flush;

    // After the first iteration, the constraints remain the same
    // and we can simply skip the rebuilding step if we do not clear it.
    if (it_nr > 1)
      return;
    const bool apply_dirichlet_bc = (it_nr == 0);

    // In the following, we will have to tell the function interpolation
    // boundary values which components of the solution vector should be
    // constrained (i.e., whether it's the x-, y-, z-displacements or
    // combinations thereof). This is done using ComponentMask objects (see
    // @ref GlossComponentMask) which we can get from the finite element if we
    // provide it with an extractor object for the component we wish to
    // select. To this end we first set up such extractor objects and later
    // use it when generating the relevant component masks:

    if (apply_dirichlet_bc)
    {
      constraints.clear();
      const FEValuesExtractors::Scalar x_displacement(0);
      const FEValuesExtractors::Scalar y_displacement(1);
      const FEValuesExtractors::Scalar z_displacement(2);
      // Fixed right hand side of the beam
      {
      const int boundary_id1 = 11;
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                boundary_id1,
                                                ZeroFunction<dim>(n_components),
                                                constraints,
                                                fe.component_mask(x_displacement));
      const int boundary_id2 = 13;
      VectorTools::interpolate_boundary_values(dof_handler_ref,
                                             boundary_id2,
                                             ZeroFunction<dim>(n_components),
                                             constraints,
                                             fe.component_mask(y_displacement));
      }
      const int boundary_id = 10; 
      if(parameters.load_type == "none")
      {
        const double delta_u_x = parameters.load_rate*time.get_delta_t();
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                boundary_id,
                                                ConstantFunction<dim>(-delta_u_x,n_components),
                                                constraints,
                                                fe.component_mask(x_displacement));     

        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                boundary_id,
                                                ZeroFunction<dim>(n_components),
                                                constraints,
                                                fe.component_mask(z_displacement));                                                                                      
        } 
      else if(parameters.load_type == "cyclic_to_zero")
        {
          std::vector<Point<dim> > tracked_vertices (1);  
          tracked_vertices[0][0] = 61.0;
          tracked_vertices[0][1] = 10.0;
          tracked_vertices[0][2] = 0.5;           
          double force, displacement;
          calculate_reaction_force(
                                  solution_n,
                                  tracked_vertices,
                                  &force, &displacement);
                                                
          const NumberType force_plus = -force;
          const NumberType displacement_plus = -displacement;      
          const double u_cycles[7] = {parameters.stretch1,parameters.stretch2,parameters.stretch3,
                                      parameters.stretch4,parameters.stretch5,parameters.stretch6,
                                      parameters.stretch7};
          //const double u_cycles[7] = {0.5,0.7,0.9,1.1,1.3,1.5,1.65};
          const double load_rate = parameters.load_rate;
          const double unload_rate = -parameters.load_rate*2; // In this specific model
            if (loading == 1)
            {
              const double delta_u_x = load_rate*time.get_delta_t();
              VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                    boundary_id,
                                                    ConstantFunction<dim>(-delta_u_x,n_components),
                                                    constraints,
                                                    fe.component_mask(x_displacement));

              VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                      boundary_id,
                                                      ZeroFunction<dim>(n_components),
                                                      constraints,
                                                      fe.component_mask(z_displacement));
                                                      
              if (displacement_plus > u_cycles[current_cycle])
                {
                  loading = 0;
                  unloading = 1;
                  if (current_cycle == cycles)
                    {
                      run_t = 0;
                    }
                }
              else
                {
                  loading = 1;
                  unloading = 0;
                }                                                   
            }
            else if(unloading == 1)
            {
              const double delta_u_x = unload_rate*time.get_delta_t();
              VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                    boundary_id,
                                                    ConstantFunction<dim>(-delta_u_x,n_components),
                                                    constraints,
                                                    fe.component_mask(x_displacement));

              VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                     boundary_id,
                                                     ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe.component_mask(z_displacement)); 
              if (force_plus < 1e-3)
                {
                  loading = 1;
                  unloading = 0;
                  current_cycle++;
                }
              else
                {
                  unloading = 1;
                  loading = 0;
                }               
            }
        } 

    }
      else
      {
      if (constraints.has_inhomogeneities())
      {
        AffineConstraints<double> homogeneous_constraints(constraints);
        for (unsigned int dof = 0; dof != dof_handler_ref.n_dofs(); ++dof)
          if (homogeneous_constraints.is_inhomogeneously_constrained(dof))
            homogeneous_constraints.set_inhomogeneity(dof, 0.0);
        constraints.clear();
        constraints.copy_from(homogeneous_constraints);
      }
      }
    constraints.close();
  }

// As the system is composed of a single block, defining a solution scheme
// for the linear problem is straight-forward.
  template <int dim,typename NumberType>
  std::pair<unsigned int, double>
  Solid<dim,NumberType>::solve_linear_system(BlockVector<double> &newton_update)
  {
    BlockVector<double> A(dofs_per_block);
    BlockVector<double> B(dofs_per_block);

    unsigned int lin_it = 0;
    double lin_res = 0.0;

    // We solve for the incremental displacement $d\mathbf{u}$.
    {
      timer.enter_subsection("Linear solver");
      std::cout << " SLV " << std::flush;
      if (parameters.type_lin == "CG")
        {
          const int solver_its = static_cast<unsigned int>(
                                    tangent_matrix.block(u_dof, u_dof).m()
                                    * parameters.max_iterations_lin);
          const double tol_sol = parameters.tol_lin
                                 * system_rhs.block(u_dof).l2_norm();

          SolverControl solver_control(solver_its, tol_sol);

          GrowingVectorMemory<Vector<double> > GVM;
          SolverCG<Vector<double> > solver_CG(solver_control, GVM);

          // We've chosen by default a SSOR preconditioner as it appears to
          // provide the fastest solver convergence characteristics for this
          // problem on a single-thread machine.  However, for multicore
          // computing, the Jacobi preconditioner which is multithreaded may
          // converge quicker for larger linear systems.
          PreconditionSelector<SparseMatrix<double>, Vector<double> >
          preconditioner (parameters.preconditioner_type,
                          parameters.preconditioner_relaxation);
          preconditioner.use_matrix(tangent_matrix.block(u_dof, u_dof));

          solver_CG.solve(tangent_matrix.block(u_dof, u_dof),
                          newton_update.block(u_dof),
                          system_rhs.block(u_dof),
                          preconditioner);

          lin_it = solver_control.last_step();
          lin_res = solver_control.last_value();
        }
      else if (parameters.type_lin == "Direct")
        {
          // Otherwise if the problem is small
          // enough, a direct solver can be
          // utilised.
          SparseDirectUMFPACK A_direct;
          A_direct.initialize(tangent_matrix.block(u_dof, u_dof));
          A_direct.vmult(newton_update.block(u_dof), system_rhs.block(u_dof));

          lin_it = 1;
          lin_res = 0.0;
        }
      else
        Assert (false, ExcMessage("Linear solver type not implemented"));

      timer.leave_subsection();
    }

    // Now that we have the displacement update, distribute the constraints
    // back to the Newton update:
    constraints.distribute(newton_update);

    return std::make_pair(lin_it, lin_res);
  }

// Here we present how the results are written to file to be viewed
// using ParaView or Visit. The method is similar to that shown in the
// tutorials so will not be discussed in detail.
  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::output_results(BlockVector<double> &solution_IN) const
  {
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,
                                  DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");

    data_out.attach_dof_handler(dof_handler_ref);
    data_out.add_data_vector(solution_n,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Postprocessor<dim> postprocessor;
    data_out.add_data_vector(solution_n, postprocessor);
    FEValues<dim> fe_values(fe, qf_cell, update_JxW_values);

    // Getting solutions 
    BlockVector<double> solution_total = solution_IN;
    std::vector<types::subdomain_id> partition_int(triangulation.n_active_cells());
    unsigned int num_comp_symm_tensor = 6;

    //Declare local vectors to store values
    // OUTPUT AVERAGED ON ELEMENTS -------------------------------------------
    std::vector<Vector<double>>cauchy_stresses_total_elements
                          (num_comp_symm_tensor,
                          Vector<double> (triangulation.n_active_cells()));
    std::vector<Vector<double>>stretches_elements
                          (dim,
                          Vector<double> (triangulation.n_active_cells()));
    // OUTPUT AVERAGED ON NODES ----------------------------------------------
    // We need to create a new FE space with a single dof per node to avoid
    // duplication of the output on nodes for our problem with dim dofs.                          
    FE_Q<dim> fe_vertex(1);
    DoFHandler<dim> vertex_handler_ref(triangulation);
    vertex_handler_ref.distribute_dofs(fe_vertex);
    AssertThrow(vertex_handler_ref.n_dofs() == triangulation.n_vertices(),
      ExcDimensionMismatch(vertex_handler_ref.n_dofs(),
                          triangulation.n_vertices()));

    Vector<double> counter_on_vertices_mpi
                    (vertex_handler_ref.n_dofs());
    Vector<double> sum_counter_on_vertices
                    (vertex_handler_ref.n_dofs());

    std::vector<Vector<double>>cauchy_stresses_total_vertex_mpi
                              (num_comp_symm_tensor,
                                Vector<double>(vertex_handler_ref.n_dofs()));
    std::vector<Vector<double>>sum_cauchy_stresses_total_vertex
                              (num_comp_symm_tensor,
                                Vector<double>(vertex_handler_ref.n_dofs()));

    std::vector<Vector<double>>green_langrage_total_vertex_mpi
                              (num_comp_symm_tensor,
                                Vector<double>(vertex_handler_ref.n_dofs()));
    std::vector<Vector<double>>sum_green_langrage_total_vertex
                              (num_comp_symm_tensor,
                                Vector<double>(vertex_handler_ref.n_dofs()));

    std::vector<Vector<double>>stretches_vertex_mpi
                              (dim,
                                Vector<double>(vertex_handler_ref.n_dofs()));
    std::vector<Vector<double>>sum_stretches_vertex
                              (dim,
                                Vector<double>(vertex_handler_ref.n_dofs()));
                            
    Vector<double> solid_vol_fraction_vertex_mpi(vertex_handler_ref.n_dofs());
    Vector<double> sum_solid_vol_fraction_vertex(vertex_handler_ref.n_dofs());


    // We need to create a new FE space with a dim dof per node to
    // be able to ouput data on nodes in vector form
    FESystem<dim> fe_vertex_vec(FE_Q<dim>(1),dim);
    DoFHandler<dim> vertex_vec_handler_ref(triangulation);
    vertex_vec_handler_ref.distribute_dofs(fe_vertex_vec);
    AssertThrow(vertex_vec_handler_ref.n_dofs() == (dim*triangulation.n_vertices()),
      ExcDimensionMismatch(vertex_vec_handler_ref.n_dofs(),
                            (dim*triangulation.n_vertices())));      
    Vector<double> counter_on_vertices_vec_mpi(vertex_vec_handler_ref.n_dofs());
    Vector<double> sum_counter_on_vertices_vec(vertex_vec_handler_ref.n_dofs());




    //Declare and initialize local unit vectors (to construct tensor basis)
    std::vector<Tensor<1,dim>> basis_vectors (dim, Tensor<1,dim>() );
    for (unsigned int i=0; i<dim; ++i){
      basis_vectors[i][i] = 1;
    }                       

    //Define a local instance of FEValues to compute updated values required
    //to calculate stresses
    const UpdateFlags uf_cell(update_values | update_gradients |
                              update_JxW_values);    

    FEValues<dim> fe_values_ref (fe, qf_cell, uf_cell);

    //Iterate through elements (cells) and Gauss Points
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
      cell(IteratorFilters::LocallyOwnedCell(),
            dof_handler_ref.begin_active()),
      endc(IteratorFilters::LocallyOwnedCell(),
            dof_handler_ref.end()),
      cell_v(IteratorFilters::LocallyOwnedCell(),
              vertex_handler_ref.begin_active()),
      cell_v_vec(IteratorFilters::LocallyOwnedCell(),
                  vertex_vec_handler_ref.begin_active());                              


    //start cell loop
    for (; cell!=endc; ++cell, ++cell_v, ++cell_v_vec)
    {
        Assert(cell->is_locally_owned(), ExcInternalError());
        Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());
        fe_values_ref.reinit(cell);
        std::vector<Tensor<2,dim>> solution_grads_u(n_q_points);
        fe_values_ref[u_fe].get_function_gradients(solution_total,
                                                  solution_grads_u);

        //start gauss point loop
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
          const Tensor<2,dim,NumberType> F = Physics::Elasticity::Kinematics::F(solution_grads_u[q_point]);
          const NumberType               det_F = determinant(F);
          const std::vector<std::shared_ptr<const PointHistory<dim,NumberType>>>
              lqph = quadrature_point_history.get_data(cell);      
          Assert(lqph.size() == n_q_points, ExcInternalError());

          //Cauchy stress
          static const SymmetricTensor<2,dim,double>
            I (Physics::Elasticity::StandardTensors<dim>::I);
          SymmetricTensor<2,dim> sigma; 
          if (time.get_timestep() < parameters.intToML)  
           {
              sigma  = lqph[q_point]->get_cauchy_base(); 
            } 
          else if(time.get_timestep() >= parameters.intToML)
           {
              sigma  = lqph[q_point]->get_cauchy_baseML();  
           }   
          //Green-Lagrange strain
          SymmetricTensor<2,dim,NumberType> E_strain = symmetrize(0.5*(transpose(F)*F - I));


          //Volumes:
          const double solid_vol_fraction = vol_reference/det_F;
          // Both average on elements and on nodes is NOT weighted with the
          // integration point volume, i.e., we assume equal contribution of each
          // integration point to the average. Ideally, it should be weighted,
          // but I haven't invested time in getting it to work properly.
          for (unsigned int v=0; v<(GeometryInfo<dim>::vertices_per_cell); ++v)
          {
            types::global_dof_index local_vertex_indices = cell_v->vertex_dof_index(v, 0);
            counter_on_vertices_mpi(local_vertex_indices) += 1;
            for (unsigned int k=0; k<dim; ++k)
            {
              cauchy_stresses_total_vertex_mpi[k](local_vertex_indices)
              += (sigma*basis_vectors[k])*basis_vectors[k];
              green_langrage_total_vertex_mpi[k](local_vertex_indices)
              += (E_strain*basis_vectors[k])*basis_vectors[k];              
              stretches_vertex_mpi[k](local_vertex_indices)
              += std::sqrt(1.0+2.0*Tensor<0,dim,double>(E_strain[k][k]));

            }

            solid_vol_fraction_vertex_mpi(local_vertex_indices)
              += solid_vol_fraction;       

            cauchy_stresses_total_vertex_mpi[3](local_vertex_indices)
              += (sigma*basis_vectors[0])*basis_vectors[1]; //sig_xy
            cauchy_stresses_total_vertex_mpi[4](local_vertex_indices)
              += (sigma*basis_vectors[0])*basis_vectors[2];//sig_xz
            cauchy_stresses_total_vertex_mpi[5](local_vertex_indices)
              += (sigma*basis_vectors[1])*basis_vectors[2]; //sig_yz

            green_langrage_total_vertex_mpi[3](local_vertex_indices)
              += (E_strain*basis_vectors[0])*basis_vectors[1]; //sig_xy
            green_langrage_total_vertex_mpi[4](local_vertex_indices)
              += (E_strain*basis_vectors[0])*basis_vectors[2];//sig_xz
            green_langrage_total_vertex_mpi[5](local_vertex_indices)
              += (E_strain*basis_vectors[1])*basis_vectors[2]; //sig_yz              
    

          }


        }
    }
    
    // Different nodes might have different amount of contributions, e.g.,
    // corner nodes have less integration points contributing to the averaged.
    // This is why we need a counter and divide at the end, outside the cell loop.
    for (unsigned int d=0; d<(vertex_handler_ref.n_dofs()); ++d)
    {
      sum_counter_on_vertices[d] =
      Utilities::MPI::sum(counter_on_vertices_mpi[d],
                          mpi_communicator);    

      sum_solid_vol_fraction_vertex[d] = Utilities::MPI::sum(solid_vol_fraction_vertex_mpi[d],
                                                              mpi_communicator);

      for (unsigned int k=0; k<num_comp_symm_tensor; ++k)
      {
        sum_cauchy_stresses_total_vertex[k][d] =
            Utilities::MPI::sum(cauchy_stresses_total_vertex_mpi[k][d],
                                mpi_communicator);
        sum_green_langrage_total_vertex[k][d] =
            Utilities::MPI::sum(green_langrage_total_vertex_mpi[k][d],
                                mpi_communicator);                                
      }
      for (unsigned int k=0; k<dim; ++k)
      {
        sum_stretches_vertex[k][d] =
            Utilities::MPI::sum(stretches_vertex_mpi[k][d],
                              mpi_communicator);
      }    
    }
    for (unsigned int d=0; d<(vertex_handler_ref.n_dofs()); ++d)
    {
      if (sum_counter_on_vertices[d]>0)
      {
        for (unsigned int i=0; i<num_comp_symm_tensor; ++i)
        {
            sum_cauchy_stresses_total_vertex[i][d] /= sum_counter_on_vertices[d];
            sum_green_langrage_total_vertex[i][d] /= sum_counter_on_vertices[d];
        }
        for (unsigned int i=0; i<dim; ++i)
        {
            sum_stretches_vertex[i][d] /= sum_counter_on_vertices[d];
        }
        sum_solid_vol_fraction_vertex[d] /= sum_counter_on_vertices[d];
      }
    }       

    // Add the results to the solution to create the output file for Paraview
    data_out.add_data_vector(vertex_handler_ref,
                            sum_cauchy_stresses_total_vertex[0],
                            "cauchy_xx");
    data_out.add_data_vector(vertex_handler_ref,
                            sum_cauchy_stresses_total_vertex[1],
                            "cauchy_yy");
    data_out.add_data_vector(vertex_handler_ref,
                            sum_cauchy_stresses_total_vertex[2],
                            "cauchy_zz");     
    data_out.add_data_vector(vertex_handler_ref,
                            sum_cauchy_stresses_total_vertex[3],
                            "cauchy_xy");                                                                               
    data_out.add_data_vector(vertex_handler_ref,
                            sum_cauchy_stresses_total_vertex[4],
                            "cauchy_xz");
    data_out.add_data_vector(vertex_handler_ref,
                            sum_cauchy_stresses_total_vertex[5],
                            "cauchy_yz");

    data_out.add_data_vector(vertex_handler_ref,
                            sum_green_langrage_total_vertex[0],
                            "E_xx");
    data_out.add_data_vector(vertex_handler_ref,
                            sum_green_langrage_total_vertex[1],
                            "E_yy");
    data_out.add_data_vector(vertex_handler_ref,
                            sum_green_langrage_total_vertex[2],
                            "E_zz");     
    data_out.add_data_vector(vertex_handler_ref,
                            sum_green_langrage_total_vertex[3],
                            "E_xy");                                                                               
    data_out.add_data_vector(vertex_handler_ref,
                            sum_green_langrage_total_vertex[4],
                            "E_xz");
    data_out.add_data_vector(vertex_handler_ref,
                            sum_green_langrage_total_vertex[5],
                            "E_yz");


    data_out.add_data_vector(vertex_handler_ref,
                              sum_stretches_vertex[0],
                              "stretch_xx");
    data_out.add_data_vector(vertex_handler_ref,
                              sum_stretches_vertex[1],
                              "stretch_yy");
    data_out.add_data_vector(vertex_handler_ref,
                              sum_stretches_vertex[2],
                              "stretch_zz");                                                        
    // Since we are dealing with a large deformation problem, it would be nice
    // to display the result on a displaced grid!  The MappingQEulerian class
    // linked with the DataOut class provides an interface through which this
    // can be achieved without physically moving the grid points in the
    // Triangulation object ourselves.  We first need to copy the solution to
    // a temporary vector and then create the Eulerian mapping. We also
    // specify the polynomial degree to the DataOut object in order to produce
    // a more refined output data set when higher order polynomials are used.


    // Uncoment this to get output on displaced grid for postprocessing.
    // Not used here since easier to follow a certain point on the grid during loading.
    // If you want to plot on displaced you have also to coment "data.out.build_patches()" below.

    Vector<double> soln(solution_n.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
      soln(i) = solution_n(i);
    MappingQEulerian<dim> q_mapping(degree, dof_handler_ref, soln);
    data_out.build_patches(q_mapping, degree);
    //data_out.build_patches();

    // The output path is here written manually !!! 
    const char *path = "./output/";
    std::ostringstream filename;
    filename << path <<  "solution-" << time.get_timestep() << ".vtk";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtk(output);

  }


  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::calculate_reaction_force(
                            BlockVector<double> &solution_IN,
                            std::vector<Point<dim> > &tracked_vertices_IN,
                            double* add_force, double* add_displacement) const
  {
    BlockVector<double> solution_total = solution_IN;
    Point<dim> reaction_force;
    Point<dim> reaction_force_extra;
    std::vector<Point<dim>> solution_vertices(tracked_vertices_IN.size());


    //Auxiliar variables needed for mpi processing
    Tensor<1,dim> sum_reaction_mpi;
    Tensor<1,dim> sum_reaction_extra_mpi;
    sum_reaction_mpi = 0.0;
    double sum_solid_vol_mpi = 0.0;
    double sum_vol_current_mpi = 0.0;
    double sum_vol_reference_mpi = 0.0;

    //Define a local instance of FEValues to compute updated values required
    //to calculate stresses
    const UpdateFlags uf_cell(update_values | update_gradients |
                              update_JxW_values);
    FEValues<dim> fe_values_ref (fe, qf_cell, uf_cell);    
    //Iterate through elements (cells) and Gauss Points
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
      cell(IteratorFilters::LocallyOwnedCell(),
        dof_handler_ref.begin_active()),
      endc(IteratorFilters::LocallyOwnedCell(),
        dof_handler_ref.end());

    //start cell loop
    for (; cell!=endc; ++cell)
    {
      Assert(cell->is_locally_owned(), ExcInternalError());
      Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());
      fe_values_ref.reinit(cell);
      std::vector<Tensor<2,dim>> solution_grads_u(n_q_points);
      fe_values_ref[u_fe].get_function_gradients(solution_total,
                                                  solution_grads_u);
       //start gauss point loop
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
        const Tensor<2,dim,NumberType> F = Physics::Elasticity::Kinematics::F(solution_grads_u[q_point]);
        const NumberType               det_F = determinant(F);
        const std::vector<std::shared_ptr<const PointHistory<dim,NumberType>>>
            lqph = quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());
        double JxW = fe_values_ref.JxW(q_point);
        //Volumes
        sum_vol_current_mpi  += det_F * JxW;
        sum_vol_reference_mpi += JxW;
        sum_solid_vol_mpi +=  JxW * det_F;
      } //end gauss point loop
        // Compute reaction force on load boundary
        // Define a local instance of FEFaceValues to compute values required
        // to calculate reaction force      
        const UpdateFlags uf_face( update_values | update_gradients |
                                  update_normal_vectors | update_JxW_values );
        FEFaceValues<dim> fe_face_values_ref(fe, qf_face, uf_face);
        //start face loop
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
          //Reaction force
          if (cell->face(face)->at_boundary() == true &&
              cell->face(face)->boundary_id() == 10)
              {
                fe_face_values_ref.reinit(cell, face);
                //Get displacement gradients for current face
                std::vector<Tensor<2,dim> > solution_grads_u_f(n_q_points_f);
                fe_face_values_ref[u_fe].get_function_gradients
                                                      (solution_total,
                                                      solution_grads_u_f);

                //start gauss points on faces loop
                for (unsigned int f_q_point=0; f_q_point<n_q_points_f; ++f_q_point)
                {
                  const Tensor<1,dim> &N = fe_face_values_ref.normal_vector(f_q_point);
                  const double JxW_f = fe_face_values_ref.JxW(f_q_point);

                  //(present configuration)
                  const std::vector<std::shared_ptr<const PointHistory<dim,NumberType>>>
                      lqph = quadrature_point_history.get_data(cell);
                  Assert(lqph.size() == n_q_points, ExcInternalError());
                  //Cauchy stress
                  SymmetricTensor<2,dim> sigma;
                  if (time.get_timestep() < parameters.intToML)  
                  {
                    sigma  = lqph[f_q_point]->get_cauchy_base(); 
                  } 
                  else if(time.get_timestep() >= parameters.intToML)
                  {
                    sigma  = lqph[f_q_point]->get_cauchy_baseML();  
                  } 
                  SymmetricTensor<2,dim> sigma_E;
                  for (unsigned int i=0; i<dim; ++i)
                    for (unsigned int j=0; j<dim; ++j)
                      sigma_E[i][j] = Tensor<0,dim,double>(sigma[i][j]);
                  sum_reaction_mpi += sigma * N * JxW_f;
                  sum_reaction_extra_mpi += sigma_E * N * JxW_f;
                }//end gauss points on faces loop                                      
              }              
        }//end face loop
    }//end cell loop
    //Sum the results from different MPI process and then add to the reaction_force vector
    //In theory, the solution on each surface (each cell) only exists in one MPI process
    //so, we add all MPI process, one will have the solution and the others will be zero 
   for (unsigned int d=0; d<dim; ++d)
   {
     reaction_force[d] = Utilities::MPI::sum(sum_reaction_mpi[d],
                                              mpi_communicator);
     reaction_force_extra[d] = Utilities::MPI::sum(sum_reaction_extra_mpi[d],
                                                   mpi_communicator);                              
   }
   
   //  Extract solution for tracked vectors
   // we copy each block of MPI::BlockVector into an MPI::Vector
   // And then we copy the MPI::Vector into "normal" Vectors
   //MPI::Vector solution_vector_u_MPI(solution_total.block(u_block));
   Vector<double> solution_vector_u_MPI(solution_total.block(u_dof));
   Vector<double> solution_u_vector(solution_vector_u_MPI);
   if (this_mpi_process == 0)
   {
     Vector<double> solution_vector(solution_u_vector.size());
     for (unsigned int d=0; d<(solution_u_vector.size()); ++d)
     {
          solution_vector[d] = solution_u_vector[d];
     }

     Functions::FEFieldFunction<dim,DoFHandler<dim>,Vector<double>>
     find_solution(dof_handler_ref, solution_vector);
     for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
     {
       Vector<double> update(dim);
       Point<dim> pt_ref;
       pt_ref[0]= tracked_vertices_IN[p][0];
       pt_ref[1]= tracked_vertices_IN[p][1];
       pt_ref[2]= tracked_vertices_IN[p][2];
       find_solution.vector_value(pt_ref, update);
       for (unsigned int d=0; d<(dim); ++d)
       {
         //For values close to zero, set to 0.0
         if (abs(update[d])<1e-4)
              update[d] = 0.0;
            solution_vertices[p][d] = update[d]; 
       }
     }    
    }
    *add_force = reaction_force[0];
    *add_displacement = solution_vertices[0][0];
  }  






  template <int dim,typename NumberType>
  void Solid<dim,NumberType>::output_results_to_plot(
                            const double current_time,
                            BlockVector<double> &solution_IN,
                            std::vector<Point<dim> > &tracked_vertices_IN,
                            std::ofstream &plotpointfile) const
  {
    BlockVector<double> solution_total = solution_IN;
    Point<dim> reaction_force;
    Point<dim> reaction_force_extra;
    double total_solid_vol = 0.0;
    double total_vol_current = 0.0;
    double total_vol_reference = 0.0;
    std::vector<Point<dim>> solution_vertices(tracked_vertices_IN.size());


    //Auxiliar variables needed for mpi processing
    Tensor<1,dim> sum_reaction_mpi;
    Tensor<1,dim> sum_reaction_extra_mpi;
    sum_reaction_mpi = 0.0;
    double sum_solid_vol_mpi = 0.0;
    double sum_vol_current_mpi = 0.0;
    double sum_vol_reference_mpi = 0.0;

    //Define a local instance of FEValues to compute updated values required
    //to calculate stresses
    const UpdateFlags uf_cell(update_values | update_gradients |
                              update_JxW_values);
    FEValues<dim> fe_values_ref (fe, qf_cell, uf_cell);    
    //Iterate through elements (cells) and Gauss Points
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
      cell(IteratorFilters::LocallyOwnedCell(),
        dof_handler_ref.begin_active()),
      endc(IteratorFilters::LocallyOwnedCell(),
        dof_handler_ref.end());

    //start cell loop
    for (; cell!=endc; ++cell)
    {
      Assert(cell->is_locally_owned(), ExcInternalError());
      Assert(cell->subdomain_id() == this_mpi_process, ExcInternalError());
      fe_values_ref.reinit(cell);
      std::vector<Tensor<2,dim>> solution_grads_u(n_q_points);
      fe_values_ref[u_fe].get_function_gradients(solution_total,
                                                  solution_grads_u);
       //start gauss point loop
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {
        const Tensor<2,dim,NumberType> F = Physics::Elasticity::Kinematics::F(solution_grads_u[q_point]);
        const NumberType               det_F = determinant(F);
        const std::vector<std::shared_ptr<const PointHistory<dim,NumberType>>>
            lqph = quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());
        double JxW = fe_values_ref.JxW(q_point);
        //Volumes
        sum_vol_current_mpi  += det_F * JxW;
        sum_vol_reference_mpi += JxW;
        sum_solid_vol_mpi +=  JxW * det_F;
      } //end gauss point loop
        // Compute reaction force on load boundary
        // Define a local instance of FEFaceValues to compute values required
        // to calculate reaction force      
        const UpdateFlags uf_face( update_values | update_gradients |
                                  update_normal_vectors | update_JxW_values );
        FEFaceValues<dim> fe_face_values_ref(fe, qf_face, uf_face);
        //start face loop
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
          //Reaction force
          if (cell->face(face)->at_boundary() == true &&
              cell->face(face)->boundary_id() == 10)
              {
                fe_face_values_ref.reinit(cell, face);
                //Get displacement gradients for current face
                std::vector<Tensor<2,dim> > solution_grads_u_f(n_q_points_f);
                fe_face_values_ref[u_fe].get_function_gradients
                                                      (solution_total,
                                                      solution_grads_u_f);

                //start gauss points on faces loop
                for (unsigned int f_q_point=0; f_q_point<n_q_points_f; ++f_q_point)
                {
                  const Tensor<1,dim> &N = fe_face_values_ref.normal_vector(f_q_point);
                  const double JxW_f = fe_face_values_ref.JxW(f_q_point);

                  //(present configuration)
                  const std::vector<std::shared_ptr<const PointHistory<dim,NumberType>>>
                      lqph = quadrature_point_history.get_data(cell);
                  Assert(lqph.size() == n_q_points, ExcInternalError());
                  //Cauchy stress
                  SymmetricTensor<2,dim> sigma;
                  if (time.get_timestep() < parameters.intToML)  
                  {
                    sigma  = lqph[f_q_point]->get_cauchy_base(); 
                  } 
                  else if(time.get_timestep() >= parameters.intToML)
                  {
                    sigma  = lqph[f_q_point]->get_cauchy_baseML();  
                  } 
                  SymmetricTensor<2,dim> sigma_E;
                  for (unsigned int i=0; i<dim; ++i)
                    for (unsigned int j=0; j<dim; ++j)
                      sigma_E[i][j] = Tensor<0,dim,double>(sigma[i][j]);
                  sum_reaction_mpi += sigma * N * JxW_f;
                  sum_reaction_extra_mpi += sigma_E * N * JxW_f;
                }//end gauss points on faces loop                                      
              }              
        }//end face loop
    }//end cell loop
    //Sum the results from different MPI process and then add to the reaction_force vector
    //In theory, the solution on each surface (each cell) only exists in one MPI process
    //so, we add all MPI process, one will have the solution and the others will be zero 
   for (unsigned int d=0; d<dim; ++d)
   {
     reaction_force[d] = Utilities::MPI::sum(sum_reaction_mpi[d],
                                              mpi_communicator);
     reaction_force_extra[d] = Utilities::MPI::sum(sum_reaction_extra_mpi[d],
                                                   mpi_communicator);                              
   }    
    total_solid_vol = Utilities::MPI::sum(sum_solid_vol_mpi,
                                          mpi_communicator);
    total_vol_current = Utilities::MPI::sum(sum_vol_current_mpi,
                                            mpi_communicator);
    total_vol_reference = Utilities::MPI::sum(sum_vol_reference_mpi,
                                              mpi_communicator);   
   
   //  Extract solution for tracked vectors
   // we copy each block of MPI::BlockVector into an MPI::Vector
   // And then we copy the MPI::Vector into "normal" Vectors
   //MPI::Vector solution_vector_u_MPI(solution_total.block(u_block));
   Vector<double> solution_vector_u_MPI(solution_total.block(u_dof));
   Vector<double> solution_u_vector(solution_vector_u_MPI);
   if (this_mpi_process == 0)
   {
     Vector<double> solution_vector(solution_u_vector.size());
     for (unsigned int d=0; d<(solution_u_vector.size()); ++d)
     {
          solution_vector[d] = solution_u_vector[d];
     }

     Functions::FEFieldFunction<dim,DoFHandler<dim>,Vector<double>>
     find_solution(dof_handler_ref, solution_vector);
     for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
     {
       Vector<double> update(dim);
       Point<dim> pt_ref;
       pt_ref[0]= tracked_vertices_IN[p][0];
       pt_ref[1]= tracked_vertices_IN[p][1];
       pt_ref[2]= tracked_vertices_IN[p][2];
       find_solution.vector_value(pt_ref, update);
       for (unsigned int d=0; d<(dim); ++d)
       {
         //For values close to zero, set to 0.0
         if (abs(update[d])<1e-4)
              update[d] = 0.0;
            solution_vertices[p][d] = update[d]; 
       }
     }  

     plotpointfile <<  std::setprecision(6) << std::scientific;  
     plotpointfile << std::setw(16) << current_time        << ","
                   << std::setw(15) << total_vol_reference << ","
                   << std::setw(15) << total_vol_current   << ","
                   << std::setw(15) << total_solid_vol     << ",";
     // Write the results to the plotting file.
     if (current_time == 0.0)
     {
       plotpointfile << std::endl; 
       for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
       {
         for (unsigned int d=0; d<dim; ++d)
              plotpointfile << std::setw(15) << 0.0 << ","; 
       }
       for (unsigned int d=0; d<(3*dim+2); ++d)
            plotpointfile << std::setw(15) << 0.0 << ",";

       plotpointfile << std::setw(15) << 0.0; 
     }   
     else{
        for (unsigned int p=0; p<tracked_vertices_IN.size(); ++p)
            for (unsigned int d=0; d<(dim); ++d)
            {
                plotpointfile << std::setw(15) << solution_vertices[p][d]<< ",";
            }
        for (unsigned int d=0; d<dim; ++d)
            plotpointfile << std::setw(15) << reaction_force[d] << ",";
        for (unsigned int d=0; d<dim; ++d)
            plotpointfile << std::setw(15) << reaction_force_extra[d] << ",";                
     }     
     plotpointfile << std::endl;                        
   }
  }

    //Header for plotting output file
    template <int dim,typename NumberType>
    void Solid<dim,NumberType>::print_plot_file_header(std::vector<Point<dim> > &tracked_vertices,
                                            std::ofstream &plotpointfile) const
    {
      plotpointfile << "#\n# *** Solution history for tracked vertices -- DOF: 0 = Ux,  1 = Uy,  2 = Uz ***"
                    << std::endl;

      for  (unsigned int p=0; p<tracked_vertices.size(); ++p)
      {
          plotpointfile << "#        Point " << p << " coordinates:  ";
          for (unsigned int d=0; d<dim; ++d)
            {
              plotpointfile << tracked_vertices[p][d];
              if (!( (p == tracked_vertices.size()-1) && (d == dim-1) ))
                  plotpointfile << ",        ";
            }
          plotpointfile << std::endl;
      }
      plotpointfile << "#    The reaction force is the integral over the loaded surfaces in the "
                    << "undeformed configuration of the Cauchy stress times the normal surface unit vector.\n"
                    << " and reac(E) corresponds to the extra part of the Cauchy stress due to the solid contribution."
                    << std::endl
                    << "# Column number:"
                    << std::endl
                    << "#";

    unsigned int columns = 24;
    for (unsigned int d=1; d<columns; ++d)
        plotpointfile << std::setw(15)<< d <<",";

      plotpointfile << std::setw(15)<< columns
                    << std::endl
                    << "#"
                    << std::right << std::setw(16) << "Time,"
                    << std::right << std::setw(16) << "ref vol,"
                    << std::right << std::setw(16) << "def vol,"
                    << std::right << std::setw(16) << "solid vol,";
      for (unsigned int p=0; p<tracked_vertices.size(); ++p)
          for (unsigned int d=0; d<(dim); ++d)
              plotpointfile << std::right<< std::setw(11)
                            <<"P" << p << "[" << d << "],";
      for (unsigned int d=0; d<dim; ++d)
          plotpointfile << std::right<< std::setw(13)
                        << "reaction [" << d << "],";

      for (unsigned int d=0; d<dim; ++d)
          plotpointfile << std::right<< std::setw(13)
                        << "reac(E) [" << d << "],";
    }
    //Footer for plotting output file
    template <int dim,typename NumberType>
    void Solid<dim,NumberType>::print_plot_file_footer(std::ofstream &plotpointfile) const
    {
           //Copy "parameters" file at end of output file.
           std::ifstream infile("parameters.prm");
           std::string content = "";
           int i;

           for(i=0 ; infile.eof()!=true ; i++)
           {
               char aux = infile.get();
               content += aux;
               if(aux=='\n') content += '#';
           }

           i--;
           content.erase(content.end()-1);
           infile.close();

           plotpointfile << "#"<< std::endl
                         << "#"<< std::endl
                         << "# PARAMETERS FILE USED IN THIS COMPUTATION:" << std::endl
                         << "#"<< std::endl
                         << content;
    }

}


// @sect3{Main function}
int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace vevpd_model;

  const unsigned int dim = 3;
  try
    {
      deallog.depth_console(0);
      Parameters::AllParameters parameters("parameters.prm");
      if (parameters.automatic_differentiation_order == 0)
        {
          std::cout << "Assembly method: Residual and linearisation are computed manually." << std::endl;

          // Allow multi-threading
          Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                              dealii::numbers::invalid_unsigned_int);

          typedef double NumberType;
          Solid<dim,NumberType> solid_3d(parameters);
          solid_3d.run();
        }
      else
        {
          AssertThrow(false,
                      ExcMessage("The selected assembly method is not supported. "
                                 "You need deal.II 9.0 and Trilinos with the Sacado package "
                                 "to enable assembly using automatic differentiation."));
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
