
# the function to compute the M norm 
function compute_weighted_norm_cpu!(ws::HPRSOCP_workspace_cpu)
    mul!(ws.Ax, ws.A, ws.dx)
    dot_prod = 2 * dot(ws.Ax, ws.dy)
    dy_squarenorm = dot(ws.dy, ws.dy)
    dx_squarenorm = dot(ws.dx, ws.dx)
    weighted_norm = ws.sigma * (ws.lambda_max * dy_squarenorm) + (dx_squarenorm) / ws.sigma + dot_prod
    if weighted_norm < 0
        println("The estimated value of lambda_max is too small! Please increase params.lambda_factor!")
        ws.lambda_max = -(dot_prod + (dx_squarenorm) / ws.sigma) / (ws.sigma * (dy_squarenorm)) * 1.05
        weighted_norm = sqrt(-(dot_prod + (dx_squarenorm) / ws.sigma) * 0.05)
    else
        weighted_norm = sqrt(weighted_norm)
    end
    return weighted_norm
end

# the Halpern iteration, Step 10 in Algorithm 2P

function Halpern_update_cpu!(ws::HPRSOCP_workspace_cpu, restart_info::HPRSOCP_restart)
    fact1 = 1.0 / (restart_info.inner + 2.0)
    fact2 = (restart_info.inner + 1.0) / (restart_info.inner + 2.0)
    ws.x .= fact1 .* ws.last_x .+ fact2 .* ws.x_hat
    ws.y .= fact1 .* ws.last_y .+ fact2 .* ws.y_hat
    restart_info.inner += 1
end

# the function to compute the residuals for the original SOCP problem
function compute_residuals_cpu!(ws::HPRSOCP_workspace_cpu,
    socp::SOCP_info_cpu,
    sc::Scaling_info_cpu,
    res::HPRSOCP_residuals,
    iter::Int
)
    ### obj
   
    res.primal_obj_bar = sc.c_scale * dot(ws.c, ws.x_bar) +socp.obj_constant
    res.dual_obj_bar = sc.b_scale * dot(ws.b, ws.y_bar) + socp.obj_constant
    res.rel_gap_bar = abs(res.primal_obj_bar - res.dual_obj_bar) / (1.0 + abs(res.primal_obj_bar) + abs(res.dual_obj_bar))

    ### Rd
    compute_err_Rd_cpu!(ws, sc)
    res.err_Rd_org_bar = sc.c_scale * norm(ws.Rd) / sc.norm_c_org

    ### Rp
    compute_err_Rp_cpu!(ws, sc)
    res.err_Rp_org_bar = sc.b_scale * norm(ws.Rp) / sc.norm_b_org

    #原始锥残差
    x_proj = similar(ws.x_bar)
    compute_soc_projection_cpu!(x_proj, ws.x_bar, length(ws.x_bar))
    res.err_p_cone = norm(ws.x_bar - x_proj) / (1.0 + norm(ws.x_bar))

    #对偶锥残差
    z_proj = similar(ws.z_bar)
    compute_soc_projection_cpu!(z_proj, ws.z_bar, length(ws.z_bar))
    res.err_d_cone = norm(ws.z_bar - z_proj) / (1.0 + norm(ws.z_bar))
    

    res.KKTx_and_gap_org_bar = max(res.err_Rp_org_bar, res.err_Rd_org_bar, res.rel_gap_bar,res.err_p_cone, res.err_d_cone )
end


# the function to update the value of sigma

function update_sigma_cpu!(
    restart_info::HPRSOCP_restart,
    ws::HPRSOCP_workspace_cpu,
    residuals::HPRSOCP_residuals,
)
    if restart_info.restart_flag >= 1 && restart_info.restart_flag <= 3
        ws.dx .= ws.x_bar .- ws.last_x
        ws.dy .= ws.y_bar .- ws.last_y
        primal_move = norm(ws.dx)
        dual_move = norm(ws.dy)
        if primal_move > 1e-16 && dual_move > 1e-16 &&
           primal_move < 1e12 && dual_move < 1e12
            pm_over_dm = primal_move / dual_move
            sqrtλ = sqrt(ws.lambda_max)
            ratio = pm_over_dm / sqrtλ
            fact = exp(-0.05 * (restart_info.current_gap / restart_info.best_gap))
            temp_1 = max(min(residuals.err_Rd_org_bar, residuals.err_Rp_org_bar), min(residuals.rel_gap_bar, restart_info.current_gap))
            sigma_cand = exp(fact * log(ratio) + (1 - fact) * log(restart_info.best_sigma))
            if temp_1 > 9e-10
                κ = 1.0
            elseif temp_1 > 5e-10
                ratio_infeas_org = residuals.err_Rd_org_bar / residuals.err_Rp_org_bar
                κ = clamp(sqrt(ratio_infeas_org), 1e-2, 100.0)
            else
                ratio_infeas_org = residuals.err_Rd_org_bar / residuals.err_Rp_org_bar
                κ = clamp((ratio_infeas_org), 1e-2, 100.0)
            end
            ws.sigma = κ * sigma_cand
        else
            ws.sigma = 1.0
        end
    end
end


# the function to check whether to restart the algorithm
function check_restart(restart_info::HPRSOCP_restart,
    iter::Int,
    check_iter::Int, sigma::Float64,
)

    restart_info.restart_flag = 0
    # adaptive restart
    if restart_info.first_restart
        if iter == check_iter
            restart_info.first_restart = false
            restart_info.restart_flag = 1
            restart_info.best_gap = restart_info.current_gap
            restart_info.best_sigma = sigma
        end
    else
        if rem(iter, check_iter) == 0
            if restart_info.current_gap < 0
                restart_info.current_gap = 1e-6
                println("current_gap < 0")
            end

            # sufficient decrease
            if restart_info.current_gap <= 0.2 * restart_info.last_gap
                restart_info.sufficient += 1
                restart_info.restart_flag = 1
            end

            # necessary decrease
            if (restart_info.current_gap <= 0.6 * restart_info.last_gap) && (restart_info.current_gap > 1.00 * restart_info.save_gap)
                restart_info.necessary += 1
                restart_info.restart_flag = 2
            end

            # long iterations
            if restart_info.inner >= 0.2 * iter
                restart_info.long += 1
                restart_info.restart_flag = 3
            end

            if restart_info.best_gap > restart_info.current_gap
                restart_info.best_gap = restart_info.current_gap
                restart_info.best_sigma = sigma
            end

            restart_info.save_gap = restart_info.current_gap
        end
    end
end

# the function to do the restart
function do_restart!(restart_info::HPRSOCP_restart, ws::HPRSOCP_workspace_cpu)
    if restart_info.restart_flag > 0
        ws.x .= ws.x_bar
        ws.y .= ws.y_bar
        ws.last_x .= ws.x_bar
        ws.last_y .= ws.y_bar
        restart_info.times += 1
        restart_info.inner = 0
        restart_info.save_gap = Inf
    end
end

# the function to check whether to stop the algorithm
function check_break(residuals::HPRSOCP_residuals,
    iter::Int,
    t_start_alg::Float64,
    params::HPRSOCP_parameters,
)
    if residuals.KKTx_and_gap_org_bar < params.stoptol
        return "OPTIMAL"
    end

    if iter == params.max_iter
        return "MAX_ITER"
    end

    if time() - t_start_alg > params.time_limit
        return "TIME_LIMIT"
    end

    return "CONTINUE"
end

# the function to collect the results

function collect_results_cpu!(
    ws::HPRSOCP_workspace_cpu,
    residuals::HPRSOCP_residuals,
    sc::Scaling_info_cpu,
    iter::Int,
    t_start_alg::Float64,
    power_time::Float64,
    status::String,
    tolerance_times::Vector{Float64},
    tolerance_iters::Vector{Int}
)
    results = HPRSOCP_results()
    results.x = Vector{Float64}(undef, ws.n)
    results.y = Vector{Float64}(undef, ws.m)
    results.z = Vector{Float64}(undef, ws.n)
    results.iter = iter
    results.time = time() - t_start_alg
    results.power_time = power_time
    results.residuals = residuals.KKTx_and_gap_org_bar
    results.primal_obj = residuals.primal_obj_bar
    results.gap = residuals.rel_gap_bar
    results.x .= sc.b_scale * (ws.x_bar ./ sc.col_norm)
    results.y .= sc.c_scale * (ws.y_bar ./ sc.row_norm)
    results.z .= sc.c_scale * (ws.z_bar .* sc.col_norm)

    results.output_type = status
    # Set tolerance results, using final values if threshold not reached
    results.time_4 = tolerance_times[1] == 0.0 ? results.time : tolerance_times[1]
    results.iter_4 = tolerance_iters[1] == 0 ? iter : tolerance_iters[1]
    results.time_6 = tolerance_times[2] == 0.0 ? results.time : tolerance_times[2]
    results.iter_6 = tolerance_iters[2] == 0 ? iter : tolerance_iters[2]
    results.time_8 = tolerance_times[3] == 0.0 ? results.time : tolerance_times[3]
    results.iter_8 = tolerance_iters[3] == 0 ? iter : tolerance_iters[3]
    return results
end


# the function to prepare the spmv for a given sparse matrix A
#gpu

# the function to allocate the workspace for the HPR-SOCP algorithm

function allocate_workspace_cpu(socp::SOCP_info_cpu, scaling_info::Scaling_info_cpu)
    ws = HPRSOCP_workspace_cpu()
    m, n = size(socp.A)
    ws.m = m
    ws.n = n
    ws.x = Vector(zeros(n))    # 当前迭代点 x
    ws.x_hat = Vector(zeros(n))    # 外推点 x̂
    ws.x_bar = Vector(zeros(n))      # 投影点 x̄
    ws.dx = Vector(zeros(n))      # x̄ - x̂，用于残差
    ws.y = Vector(zeros(m))       # 当前对偶点 y
    ws.y_hat = Vector(zeros(m))        # 外推点 ŷ
    ws.y_bar = Vector(zeros(m))       # 投影点 ȳ

    ws.dy = Vector(zeros(m))        # ȳ - ŷ，用于残差
    ws.z_bar = Vector(zeros(n))     # 对偶辅助变量 z̄
    ws.A = socp.A
    ws.AT = socp.AT
    ws.b = copy(socp.b)
    ws.c = socp.c
 
    ws.Rp = Vector(zeros(m))
    ws.Rd = Vector(zeros(n))
    ws.ATy = Vector(zeros(n))
    ws.Ax = Vector(zeros(m))
    ws.last_x = Vector(zeros(n))
    ws.last_y = Vector(zeros(m))
    ws.to_check = false
    if scaling_info.norm_b > 1e-8 && scaling_info.norm_c > 1e-8
        ws.sigma = scaling_info.norm_b / scaling_info.norm_c
    else
        ws.sigma = 1.0
    end
    return ws
end

# the function to initialize the restart information
function initialize_restart(sigma::Float64)
    restart_info = HPRSOCP_restart()
    restart_info.first_restart = true
    restart_info.save_gap = Inf
    restart_info.current_gap = Inf
    restart_info.last_gap = Inf
    restart_info.best_gap = Inf
    restart_info.best_sigma = sigma
    restart_info.inner = 0
    restart_info.times = 0
    restart_info.sufficient = 0
    restart_info.necessary = 0
    restart_info.long = 0
    restart_info.ratio = 0
    restart_info.restart_flag = 0
    restart_info.weighted_norm = Inf
    return restart_info
end

function print_step(iter::Int)
    return max(10^floor(log10(iter)) / 10, 10)
end

function compute_maximum_eigenvalue!(socp::SOCP_info_cpu,
    ws::HPRSOCP_workspace_cpu,
    params::HPRSOCP_parameters)
    t_start_power = time()
    println("ESTIMATING MAXIMUM EIGENVALUE ...")
    
    lambda_max = power_iteration_cpu(socp.A, socp.AT) * 1.01
    
    power_time = time() - t_start_power
    println(@sprintf("ESTIMATING MAXIMUM EIGENVALUE time = %.2f seconds", power_time))
    println(@sprintf("estimated maximum eigenvalue of AAT = %.2e", lambda_max))
    ws.lambda_max = lambda_max

    return power_time
end

# The main function for the HPR-SOCP algorithm
function solve(socp::SOCP_info_cpu,
    scaling_info::Scaling_info_cpu,
    params::HPRSOCP_parameters)

    println("HPR-LP version v0.1.3")
    t_start_alg = time()

    ### Initialization ###
    residuals = HPRSOCP_residuals()
    ws =allocate_workspace_cpu(socp, scaling_info)
    restart_info = initialize_restart(ws.sigma)

    ### power iteration to estimate lambda_max ###
    power_time = compute_maximum_eigenvalue!(socp, ws, params)

    println(" iter     errRp        errRd         p_obj            d_obj          gap         sigma       time")

    # Track when tolerance thresholds are reached
    tolerance_levels = [1e-4, 1e-6, 1e-8]
    tolerance_times = zeros(Float64, length(tolerance_levels))
    tolerance_iters = zeros(Int, length(tolerance_levels))
    tolerance_reached = falses(length(tolerance_levels))


    for iter = 0:params.max_iter
        ### whether to print the log ###
        if params.print_frequency == -1
            print_yes = ((rem(iter, print_step(iter)) == 0) || (iter == params.max_iter) ||
                         (time() - t_start_alg > params.time_limit))
        elseif params.print_frequency > 0
            print_yes = ((rem(iter, params.print_frequency) == 0) || (iter == params.max_iter) ||
                         (time() - t_start_alg > params.time_limit))
        else
            error("Invalid print_frequency: ", params.print_frequency, ". It should be a positive integer or -1 for automatic printing.")
        end

        ### compute residuals ###
        if rem(iter, params.check_iter) == 0 || print_yes
            compute_residuals_cpu!(ws, socp, scaling_info, residuals, iter)
        end

        ### check break ###
        status = check_break(residuals, iter, t_start_alg, params)

        ### check restart ###
        check_restart(restart_info, iter, params.check_iter, ws.sigma)

        ### print the log ###
        if print_yes || (status != "CONTINUE")
            println(@sprintf("%5.0f    %3.2e    %3.2e    %+7.6e    %+7.6e    %3.2e    %3.2e    %6.2f",
                iter,
                residuals.err_Rp_org_bar,
                residuals.err_Rd_org_bar,
                residuals.primal_obj_bar,
                residuals.dual_obj_bar,
                residuals.rel_gap_bar,
                ws.sigma,
                time() - t_start_alg))
        end

        ### collect results and return ###
        # Check tolerance thresholds
        for i in eachindex(tolerance_levels)
            if !tolerance_reached[i] && residuals.KKTx_and_gap_org_bar < tolerance_levels[i]
                tolerance_times[i] = time() - t_start_alg
                tolerance_iters[i] = iter
                tolerance_reached[i] = true
                println("KKT < ", tolerance_levels[i], " at iter = ", iter)
            end
        end

        if status != "CONTINUE"
            println("Termination reason: ", status, ", accuracy = ", residuals.KKTx_and_gap_org_bar)
            results = collect_results_cpu!(ws, residuals, scaling_info, iter, t_start_alg, power_time, status, tolerance_times, tolerance_iters)
            return results
        end

        ### update sigma ###
        update_sigma_cpu!(restart_info, ws, residuals)

        ### restart if needed ###
        do_restart!(restart_info, ws)

        ## whether to compute bar points for residuals ##
        ws.to_check = (rem(iter + 1, params.check_iter) == 0) || (restart_info.restart_flag > 0)
        if params.print_frequency == -1
            ws.to_check = ws.to_check || (rem(iter + 1, print_step(iter + 1)) == 0)
        elseif params.print_frequency > 0
            ws.to_check = ws.to_check || (rem(iter + 1, params.print_frequency) == 0)
        end

        ### update x, y,  and z ###
        fact1 = 1.0 / (restart_info.inner + 2.0)
        fact2 = 1.0 - fact1
        update_x_z_cpu!(ws, fact1, fact2)
        update_y_cpu!(ws, fact1, fact2)
        restart_info.inner += 1 

        ### compute weighted norm ###
        if rem(iter + 1, params.check_iter) == 0
            restart_info.current_gap = compute_weighted_norm_cpu!(ws)
        end
        if restart_info.restart_flag > 0
            restart_info.last_gap = compute_weighted_norm_cpu!(ws)
        end
    end
end
