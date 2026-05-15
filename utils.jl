# The function to read the SOCP problem from the file and formulate the SOCP problem
  

# the scaling function for the SOCP problem
function scaling!(socp::SOCP_info_cpu, use_Ruiz_scaling::Bool, use_Pock_Chambolle_scaling::Bool, use_bc_scaling::Bool)
    m, n = size(socp.A)
    row_norm = ones(m)
    col_norm = ones(n)

    # Preallocate temporary arrays
    # 初始化行、列缩放因子（列缩放因子保持为1，不进行列缩放）
    temp_norm1 = zeros(m)          # 仅用于行缩放
    DA = spdiagm(temp_norm1)       # 行缩放对角矩阵，稍后重新赋值

    # 计算原始右端项和目标系数的范数（用于终止条件分母）
    norm_b_org = 1 + norm(socp.b)
    norm_c_org = 1 + norm(socp.c)

    # 创建缩放信息结构体
    # 注意：l_org 和 u_org 在SOCP中不需要，传入空向量占位
    scaling_info = Scaling_info_cpu(
        Vector{Float64}(),          # l_org（空）
        Vector{Float64}(),          # u_org（空）
        row_norm,                    # 初始行缩放因子（全1）
        col_norm,                    # 初始列缩放因子（全1）
        1.0, 1.0, 1.0, 1.0,         # b_scale, c_scale, norm_b, norm_c 暂时为1
        norm_b_org, norm_c_org       # 原始范数
    )


    # Ruiz scaling
    if use_Ruiz_scaling
        for _ in 1:10
            temp_norm1 .= sqrt.(maximum(abs, socp.A, dims=2)[:, 1])
            temp_norm1[iszero.(temp_norm1)] .= 1.0
            row_norm .*= temp_norm1
            DA .= spdiagm(1.0 ./ temp_norm1)
            
            # 对矩阵A进行行缩放
            socp.A .= DA * socp.A 
            # 对右端项b进行同步行缩放，保持 Ax = b 不变
            socp.b .= DA * socp.b
        end
    end

    # Pock-Chambolle scaling
    if use_Pock_Chambolle_scaling
        # 行缩放因子：每行绝对值和平方根
        temp_norm1 .= sqrt.(sum(abs, socp.A, dims=2)[:, 1])
        temp_norm1[iszero.(temp_norm1)] .= 1.0
        row_norm .*= temp_norm1
        DA = spdiagm(1.0 ./ temp_norm1)
        
        # 对矩阵 A 进行行缩放
        socp.A .= DA * socp.A 
        # 同步缩放右端项 b，保持 Ax = b 不变
        socp.b .= DA * socp.b
    end

    # scaling for b and c
    if use_bc_scaling
        # 计算右端项的整体缩放因子：1 + ||b||
        b_scale = 1 + norm(socp.b)
        # 计算目标系数的整体缩放因子：1 + ||c||
        c_scale = 1 + norm(socp.c)

        # 应用缩放
        socp.b ./= b_scale
        socp.c ./= c_scale

        # 记录缩放因子
        scaling_info.b_scale = b_scale
        scaling_info.c_scale = c_scale
    else
        scaling_info.b_scale = 1.0
        scaling_info.c_scale = 1.0
    end

    scaling_info.norm_b = norm(socp.b)
    scaling_info.norm_c = norm(socp.c)
    socp.AT = transpose(socp.A)
    scaling_info.row_norm = row_norm
    scaling_info.col_norm = col_norm
    return scaling_info
end



function power_iteration_cpu(A::SparseMatrixCSC, AT::SparseMatrixCSC,
    max_iterations::Int=5000, tolerance::Float64=1e-4)
    seed = 1
    m, n = size(A)
    z = Vector(randn(Random.MersenneTwister(seed), m)) .+ 1e-8 # Initial random vector
    q = zeros(Float64, m)
    ATq = zeros(Float64, n)
    lambda_max = 0.0   # 初始化，避免未定义
  
    for i in 1:max_iterations
        q .= z
        q ./= norm(q)
        mul!(ATq, AT, q)
        mul!(z, A, ATq)
        lambda_max = dot(q, z)
        q .= z .- lambda_max .* q
        if norm(q) < tolerance
            return lambda_max
        end
    end
    println("Power iteration did not converge within the specified tolerance.")
    println("The maximum iteration is ", max_iterations, " and the error is ", norm(q))
    return lambda_max
end

# the function to run the HPR-SOCP algorithm on a single file



# it's used in demo_Abc.jl
function run_socp(A::SparseMatrixCSC,
    b::Vector{Float64},

    c::Vector{Float64},
 
    obj_constant::Float64,
    params::HPRSOCP_parameters)
    
    if params.warm_up
        println("warm up starts: ---------------------------------------------------------------------------------------------------------- ")
        t_start_all = time()
        max_iter = params.max_iter
        params.max_iter = 200
        results = run_socp_core(A, b, c, obj_constant, params)
        params.max_iter = max_iter
        all_time = time() - t_start_all
        println("warm up time: ", all_time)
        println("warm up ends ----------------------------------------------------------------------------------------------------------")
    end
    println("main run starts: ----------------------------------------------------------------------------------------------------------")
    results = run_socp_core(A, b, c, obj_constant, params)
    println("main run ends----------------------------------------------------------------------------------------------------------")
    return results
end

# the function to run the HPR-SOCP algorithm on a single SOCP problem 
function run_socp_core(A::SparseMatrixCSC,
    b::Vector{Float64},
    c::Vector{Float64},
    
    obj_constant::Float64,
    params::HPRSOCP_parameters)
    A = copy(A)
    b = copy(b)
    c = copy(c)
   
    setup_start = time()
    standard_socp = SOCP_info_cpu(A, transpose(A), b, c, obj_constant)
    
        t_start = time()
        println("SCALING SOCP ...")
        scaling_info = scaling!(standard_socp, params.use_Ruiz_scaling, params.use_Pock_Chambolle_scaling, params.use_bc_scaling)
        println(@sprintf("SCALING SOCP time: %.2f seconds", time() - t_start))

        
       
    setup_time = time() - setup_start

    
        results = solve(standard_socp, scaling_info, params)
    
    println(@sprintf("Total time: %.2fs", setup_time + results.time),
        @sprintf("  setup time = %.2fs", setup_time),
        @sprintf("  solve time = %.2fs", results.time))
    return results
end

