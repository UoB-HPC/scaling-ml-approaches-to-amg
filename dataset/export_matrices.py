from dolfin import *
import time
from mshr import *
import numpy as np
from tqdm.notebook import tqdm
from scipy.sparse import csr_matrix, save_npz
import matplotlib.pyplot as plt
import json
import time
import dolfin
import petsc4py
from pathlib import Path
import os
import logging
import sys
import re
import random



logger = logging.getLogger('GEN_MAT')
logger.setLevel(logging.INFO)


stdout_logger = logging.getLogger('GEN_MAT_STDOUT')
stdout_logger.setLevel(logging.INFO)


handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
stdout_logger.addHandler(handler)

fileHandler = logging.FileHandler("{0}/{1}-{2}.log".format(".", "generate-matrices", time.localtime()))
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)


logger.info("Installed FEniCS krylov precondtioners: {}".format(krylov_solver_preconditioners()))
logger.info("Current FEniCS Linear Algebra backend: {}".format(parameters.to_dict()['linear_algebra_backend']))


# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = True;
mpi_comm = MPI.comm_world
my_rank = MPI.rank(mpi_comm)

def walk_models_paper():
    for file in os.listdir("0_FIGURE_EXAMPLES"):
        if file.startswith("figure"):
            filename = "0_FIGURE_EXAMPLES/{}/output.msh.xdmf".format(file)
            if Path(filename).is_file():
                yield file, filename
            else:
                logger.warning("Looks like {} doesn't have a mesh file in the expected place".format(file))


def walk_models_real(max_models=10001):
    count = 0
    all_files = [x for x in os.listdir("ftetwild_output_msh") if x.endswith(".xdmf")]
    random.shuffle(all_files)
    for file in all_files:
        if file.endswith("xdmf"):
            count += 1
            if count > max_models:
                break
            filename = "ftetwild_output_msh/{}".format(file)
            if Path(filename).is_file():
                file = file.split('.')[0]
                yield file, filename
            else:
                logger.warning("Looks like {} doesn't have a mesh file in the expected place".format(file))
                
                
def load_mesh( filename) -> Mesh:
    mesh = Mesh()
    f = XDMFFile(mpi_comm, filename)
    f.read(mesh)
    return mesh

def matrix_shape(m):
     return as_backend_type(m).mat().size
    
def mkdir_if_not_exist(dirname):
    Path(dirname).mkdir(parents=True, exist_ok=True)


def assemble_matrices(mesh):
    # Define function spaces (P2-P1)
    "P tetrahedron 1"

    V = VectorFunctionSpace(mesh, "P", 2)
    Q = FunctionSpace(mesh, "P", 1)

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Set parameter values
    dt = 0.01
    T = 0.06
    nu = 0.01

    # Define time-dependent pressure boundary condition
    p_in = Expression("sin(3.0*t)", t=0.0, degree=2)

    # Define boundary conditions
    noslip  = DirichletBC(V, (0, 0, 0),
                          "on_boundary")
    inflow  = DirichletBC(Q, p_in, "x[0] < 0.1 - DOLFIN_EPS")
    outflow = DirichletBC(Q, 0, "x[0] > {} - DOLFIN_EPS".format(mesh.hmax()))
    bcu = [noslip]
    bcp = [inflow, outflow]

    # Create functions
    u0 = Function(V)
    u1 = Function(V)
    p1 = Function(Q)

    set_log_level(LogLevel.DEBUG)

    vertex_values = u0.compute_vertex_values(mesh)
    logger.info("Number of DoF: {}".format(len(vertex_values)))
    dof = len(vertex_values)


    # Define coefficients
    k = Constant(dt)
    f = Constant((0, 0, 0))

    # Tentative velocity step
    F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx +        nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Pressure update
    a2 = inner(grad(p), grad(q))*dx
    L2 = -(1/k)*div(u1)*q*dx

    # Velocity update
    a3 = inner(u, v)*dx
    L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Use amg preconditioner if available
    prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

    # Use nonzero guesses - essential for CG with non-symmetric BC
    parameters['krylov_solver']['nonzero_initial_guess'] = True

    logger.info("A1: {}".format(matrix_shape(A1)))
    logger.info("A2: {}".format(matrix_shape(A2)))
    logger.info("A3: {}".format(matrix_shape(A3)))

    # Time-stepping
    t = dt
    max_num_timesteps = 10
    num_timesteps = 0
    while num_timesteps < max_num_timesteps:
        num_timesteps += 1
        # Update pressure boundary condition
        p_in.t = t

        # Compute tentative velocity step
        begin("Computing tentative velocity")
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, u1.vector(), b1, "gmres", "hypre_amg")
        end()

        # Pressure correction
        begin("Computing pressure correction")
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        [bc.apply(p1.vector()) for bc in bcp]
        solve(A2, p1.vector(), b2, "gmres", "hypre_amg")
        end()

        # Velocity correction
        begin("Computing velocity correction")
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        solve(A3, u1.vector(), b3, "gmres", "hypre_amg")
        end()

        # Move to next time step
        u0.assign(u1)
        t += dt
    
    return A1, A2, A3, b1, b2, b3, dof
    

def save_matrix_plot(mat, name, filename):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.spy(mat)
    ax.set_title(name)
    plt.savefig(filename, dpi=150)


def write_matrices(A1, A2, A3, b1, b2, b3, dof, problem):
    json_dict = {}
    json_dict['problem'] = problem
    json_dict['matrices'] = []

    for obj, name in [(A1, "velocity_A"), (b1, "velocity_b"), 
                      (A2, "pressure_A"), (b2, "pressure_b"),
                      (A3, "velocity_correction_A"), (b3, "velocity_correction_b")]:
        if type(obj) is dolfin.cpp.la.Vector:
            v = as_backend_type(obj).vec()
            as_csr = csr_matrix(v.getArray())
        else:
            A_mat = as_backend_type(obj).mat()
            as_csr = csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size)

        save_npz("output/{}/{}-{}.npz".format(problem, problem, name), as_csr, compressed=True)
        this_json_dict={}
        this_json_dict['name'] = name
        this_json_dict['nnz'] = as_csr.nnz
        this_json_dict['shape'] = as_csr.shape
        this_json_dict['sparsity_percent'] = as_csr.nnz/(as_csr.shape[0]*as_csr.shape[1])*100
        json_dict['matrices'].append(this_json_dict)

    json_dict["degrees_of_freedom"] = dof

    with open("output/{}/{}.json".format(problem, problem), 'w') as outfile:    
        json.dump(json_dict, outfile)


if __name__=="__main__":
    mkdir_if_not_exist('output')
    for problem, mesh_filename in walk_models_real():
        logger.info("=======PROBLEM " + problem)
        stdout_logger.info("=======PROBLEM " + problem)
        mkdir_if_not_exist('output/{}'.format(problem))
        if Path("output/{}/{}.json".format(problem, problem)).is_file():
            logger.info("Skipping {}, an output file already exists for it".format(problem))
        else:
            try:
                mesh = load_mesh(mesh_filename)
                A1, A2, A3, b1, b2, b3, dof = assemble_matrices(mesh)
                write_matrices(A1, A2, A3, b1, b2, b3, dof, problem)
            except Exception as e:
                logger.error("Something went wrong processing {}".format(problem), exc_info=e)

