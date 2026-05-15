
# the space for the parameters of the HPR-SOCP algorithm
mutable struct HPRSOCP_parameters
    # the stopping tolerance, default is 1e-6
    stoptol::Float64

    # the maximum number of iterations, default is 1000
    max_iter::Int

    # the time limit in seconds, default is 3600.0
    time_limit::Float64

    # the check interval for the residuals, default is 150
    check_iter::Int

    # whether to use the Ruiz scaling, default is true
    use_Ruiz_scaling::Bool

    # whether to use the Pock-Chambolle scaling, default is true
    use_Pock_Chambolle_scaling::Bool

    # whether to use the scaling for b and c, default is true
    use_bc_scaling::Bool

    # use GPU or not, default is true
   

    # GPU device number, default is 0
   

    # whether do warm up, default is false
    warm_up::Bool

    # print frequency, print the log every print_frequency iterations, default is -1 (auto)
    print_frequency::Int

    # Default constructor
    HPRSOCP_parameters() = new(1e-4, typemax(Int32), 3600.0, 150, true, true, true, true, -1)
end


# Define the results will be returned
mutable struct HPRSOCP_results
    # Number of iterations
    iter::Int

    # Number of iterations for the 1e-4 accuracy
    iter_4::Int

    # Number of iterations for the 1e-6 accuracy
    iter_6::Int

    # Number of iterations for the 1e-8 accuracy
    iter_8::Int

    # Time in seconds
    time::Float64

    # Time in seconds for the 1e-4 accuracy
    time_4::Float64

    # Time in seconds for the 1e-6 accuracy
    time_6::Float64

    # Time in seconds for the 1e-8 accuracy
    time_8::Float64

    # Time used by power method
    power_time::Float64

    # Primal objective value
    primal_obj::Float64

    # Relative residuals of the primal feasibility, dual feasibility, and objective gap
    residuals::Float64

    # Objective gap
    gap::Float64


    # OPTIMAL, MAX_ITER or TIME_LIMIT
    # OPTIMAL: the algorithm finds the optimal solution
    # MAX_ITER: the algorithm reaches the maximum number of iterations
    # TIME_LIMIT: the algorithm reaches the time limit
    output_type::String

    # The vector x
    x::Vector{Float64}

    # The vector y
    y::Vector{Float64}

    # The vector z
    z::Vector{Float64}

    # Default constructor
    HPRSOCP_results() = new()
end


# Define the workspace for the HPR-SOCP algorithm


mutable struct HPRSOCP_workspace_cpu
    x::Vector{Float64}
    x_hat::Vector{Float64}
    x_bar::Vector{Float64}
    dx::Vector{Float64}
    y::Vector{Float64}
    y_hat::Vector{Float64}
    y_bar::Vector{Float64}
  
    dy::Vector{Float64}
    z_bar::Vector{Float64}
    A::SparseMatrixCSC{Float64,Int32}
    AT::SparseMatrixCSC{Float64,Int32}
    c::Vector{Float64}
    b::Vector{Float64}
    Rp::Vector{Float64}
    Rd::Vector{Float64}
    m::Int
    n::Int
    sigma::Float64
    lambda_max::Float64
    Ax::Vector{Float64}
    ATy::Vector{Float64}
    last_x::Vector{Float64}
    last_y::Vector{Float64}
    to_check::Bool
    HPRSOCP_workspace_cpu() = new()
end

# Define the variables related to the residuals of the HPR-SOCP
mutable struct HPRSOCP_residuals
    # The relative residuals of the primal feasibility evaluated at x_bar
    err_Rp_org_bar::Float64

    # The relative residuals of the dual feasibility evaluated at y_bar and z_bar
    err_Rd_org_bar::Float64

    # 原始锥残差 (x ∈ K)
    err_p_cone::Float64

    # 对偶锥残差 (z ∈ K)
    err_d_cone::Float64

    # The primal objective value evaluated at x_bar
    primal_obj_bar::Float64

    # The dual objective value evaluated at y_bar and z_bar
    dual_obj_bar::Float64

    # The relative gap evaluated at x_bar, y_bar, and z_bar
    rel_gap_bar::Float64

    # The maximum of the primal feasibility, dual feasibility, and duality gap
    KKTx_and_gap_org_bar::Float64

    # Define a default constructor
    HPRSOCP_residuals() = new()
end

# Define the variables related to the restart of the HPR-SOCP
mutable struct HPRSOCP_restart
    # indicate which restart condition is satisfied, 1: sufficient, 2: necessary, 3: long
    restart_flag::Int

    # indicate whether it is the first restart
    first_restart::Bool

    # the value \tilde{R}_{r,0}
    last_gap::Float64

    # the value \tilde{R}_{r,t}
    current_gap::Float64

    # the value \tilde{R}_{r,t-1}
    save_gap::Float64

    # the best value \tilde{R}_{best}
    best_gap::Float64

    # the  value of sigma at the best_gap
    best_sigma::Float64

    # the number of inner iterations, t in the paper
    inner::Int

    # the number of restart step length for fixed step restart
    step::Int

    # the number of restart triggered by sufficient decrease
    sufficient::Int

    # the number of restart triggered by necessary decrease
    necessary::Int

    # the number of restart triggered by long iterations
    long::Int

    # the ratio of ||Δx|| and ||Δy||
    ratio::Int

    # the number of restart
    times::Int

    # the value of M-norm 
    weighted_norm::Float64

    # Default constructor
    HPRSOCP_restart() = new()
end

# the space for the SOCP information on the CPU
mutable struct SOCP_info_cpu
    A::SparseMatrixCSC{Float64,Int32}
    AT::SparseMatrixCSC{Float64,Int32}
    b::Vector{Float64}
    c::Vector{Float64}
   
    obj_constant::Float64
end

# the space for the scaling information on the CPU
mutable struct Scaling_info_cpu
    # the original vector l
    l_org::Vector{Float64}

    # the original vector u
    u_org::Vector{Float64}

    # the row norm of the matrix A
    row_norm::Vector{Float64}

    # the column norm of the matrix A
    col_norm::Vector{Float64}

    # the scaling factor for the vector b
    b_scale::Float64

    # the scaling factor for the vector c
    c_scale::Float64

    # the norm of the vector b
    norm_b::Float64

    # the norm of the vector c
    norm_c::Float64

    # the norm of the original vector b
    norm_b_org::Float64

    # the norm of the original vector c
    norm_c_org::Float64
end

