function compute_soc_projection_cpu!(proj, r, n)
    if n < 2
        error("SOC dimension must be at least 2")
    end
    
    t = r[1]
    u_norm = 0.0
    @inbounds for i in 2:n
        u_norm += r[i]^2
    end
    u_norm = sqrt(u_norm)
    
    if u_norm <= t
        # 在锥内
        @inbounds for i in 1:n
            proj[i] = r[i]
        end
    elseif u_norm <= -t
        # 在锥的极锥内
        @inbounds for i in 1:n
            proj[i] = 0.0
        end
    else
        # 在锥外
        alpha = (t + u_norm) / 2
        proj[1] = alpha
        
        safe_u_norm = max(u_norm, 1e-12)
        @inbounds for i in 2:n
            proj[i] = (alpha / safe_u_norm) * r[i]
        end
    end
    
    return proj
end



function update_x_z_cpu!(ws::HPRSOCP_workspace_cpu, fact1::Float64, fact2::Float64)
    mul!(ws.ATy, ws.AT, ws.y)
    x = ws.x
    x_bar = ws.x_bar
    z_bar = ws.z_bar
    x_hat = ws.x_hat
    x0 = ws.last_x
    sigma = ws.sigma
    ATy = ws.ATy
    c = ws.c
    dx = ws.dx
    n = length(x)

    # 计算临时向量 r = x + σ·(Aᵀy - c)
    r = similar(x)          # 可考虑预分配以节省内存
    @. r = x + sigma * (ATy - c)


    if ws.to_check
        # 将 r 投影到二阶锥，结果存入 x_bar
        compute_soc_projection_cpu!(x_bar, r, n)

        @simd for i in eachindex(x)
            @inbounds begin
               x_hat[i] = 2 * x_bar[i] - x[i]
               dx[i] = x_bar[i] - x_hat[i]
               x[i] = fact1 * x0[i] + fact2 * x_hat[i]
               z_bar[i] = (x_bar[i] - r[i]) / sigma 
            end
        end
    else
        compute_soc_projection_cpu!(x_bar, r, n)

        @simd for i in eachindex(x)
            @inbounds begin
              
               x_hat[i] = 2 * x_bar[i] - x[i]
               x[i] = fact1 * x0[i] + fact2 * x_hat[i]
            end
        end
    end
    return
end


function update_y_cpu!(ws::HPRSOCP_workspace_cpu, Halpern_fact1::Float64, Halpern_fact2::Float64)
    mul!(ws.Ax, ws.A, ws.x_hat)
    fact = 1.0 / (ws.lambda_max * ws.sigma)   # 1/(λσ)
    y = ws.y
    y0 = ws.last_y
    y_bar = ws.y_bar
    y_hat = ws.y_hat
    Ax = ws.Ax
    dy = ws.dy
    b = ws.b
    @simd for i in eachindex(y)
        @inbounds begin
            # 直接计算投影点（等式约束）
            y_bar[i] = y[i] + fact * (b[i] - Ax[i])
            # 外推
            y_hat[i] = 2 * y_bar[i] - y[i]
            # 计算用于加权范数的差分
            dy[i] = y_bar[i] - y_hat[i]        # 对应原代码中的 dy
            # Halpern update
            y[i] = Halpern_fact1 * y0[i] + Halpern_fact2 * y_hat[i]
        end
    end
    return
end

# the kernel function to compute the dual residuals, ||c - A^T y - z||

function compute_err_Rd_cpu!(ws::HPRSOCP_workspace_cpu, sc::Scaling_info_cpu)
    mul!(ws.Rd, ws.AT, ws.y_bar)
    c = ws.c
    Rd = ws.Rd
    z_bar = ws.z_bar
    @simd for i in eachindex(Rd)
        @inbounds Rd[i] = Rd[i] + z_bar[i] - c[i]
   
    end
end


# the kernel function to compute the primal residuals, 
#||Ax-b||
function compute_err_Rp_cpu!(ws::HPRSOCP_workspace_cpu, sc::Scaling_info_cpu)
    mul!(ws.Ax, ws.A, ws.x_bar)
    b = ws.b
    Ax = ws.Ax
    Rp = ws.Rp
    row_norm = sc.row_norm

    # Parallelize and eliminate bounds checks & branching
    @simd for i in eachindex(Rp)
        @inbounds Rp[i] = Ax[i] - b[i]    # 缩放后的线性残差
        @inbounds Rp[i] *= row_norm[i]    # 乘以行范数以还原原始尺度
    end
end

