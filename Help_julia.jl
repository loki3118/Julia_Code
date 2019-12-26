using LinearAlgebra, Plots

#--------------------------------------------------------------------------
# Function Declarations
function Sim1d_mat(Xpos, num_pts, resol, b1, grad, offset, M)

    gam = 2*pi*4258
    Mout = zeros(3, length(Xpos))
    B1 = [0.0; 0.0; 0.0]

    @progress for n = 1:length(Xpos)
        for tt = 1:num_pts-1
            Δω = gam * Xpos[n] * grad[tt] + offset[n]
            B1 = [real(b1[tt]); imag(b1[tt]); Δω]

            θ = norm(B1) * resol

            if abs(θ) > 1e-9
                B1 = B1 / norm(B1)
                M[:, tt+1]  = M[:, tt] * cos(θ) + cross(B1, M[:, tt]) * sin(θ) +
                       B1 * dot(B1, M[:, tt]) * (1 - cos(θ))
            else
                M[:, tt+1] = M[:, tt]
            end
        end
        Mout[:, n] = M[:, end]
    end
    return Mout
end

function Sim1d_vec(Xpos, num_pts, resol, b1, grad, offset, Mx, My, Mz)

    gam = 2*pi*4258
    Mxout_1d = zeros(length(Xpos))
    Myout_1d = zeros(length(Xpos))
    Mzout_1d = zeros(length(Xpos))

    @progress for h = 1:length(Xpos)
        crs_prd = zeros(3)
        Weff = zeros(3)
        Mxyz = zeros(6)

        Mxyz[1] = Mx[h]
        Mxyz[3] = My[h]
        Mxyz[5] = Mz[h]

        for tt = 1:num_pts
            b1x_tt = real(b1[tt])
            b1y_tt = imag(b1[tt])

            Weff[1] = b1x_tt
            Weff[2] = b1y_tt
            Weff[3] = gam * Xpos[h] * grad[tt] + offset[h]

            abs_weff = sqrt(Weff[1]^2 + Weff[2]^2 + Weff[3]^2)
            phi = -abs_weff * resol

            if abs_weff > 1e-3
                Weff[1] = Weff[1] / abs_weff
                Weff[2] = Weff[2] / abs_weff
                Weff[3] = Weff[3] / abs_weff

                crs_prd[1] = Weff[2] * Mxyz[5] - Weff[3] * Mxyz[3]
                crs_prd[2] = Weff[3] * Mxyz[1] - Weff[1] * Mxyz[5]
                crs_prd[3] = Weff[1] * Mxyz[3] - Weff[2] * Mxyz[1]

                dot = Weff[1] * Mxyz[1] + Weff[2] * Mxyz[3] + Weff[3] * Mxyz[5]

                Mxyz[2] = (cos(phi) * Mxyz[1] + sin(phi) * crs_prd[1] +
                           (1 - cos(phi)) * dot * Weff[1])
                Mxyz[4] = (cos(phi) * Mxyz[3] + sin(phi) * crs_prd[2] +
                           (1 - cos(phi)) * dot * Weff[2])
                Mxyz[6] = (cos(phi) * Mxyz[5] + sin(phi) * crs_prd[3] +
                           (1 - cos(phi)) * dot * Weff[3])
            else
                Mxyz[2] = Mxyz[1]
                Mxyz[4] = Mxyz[3]
                Mxyz[6] = Mxyz[5]
            end
            if tt == num_pts
                Mxout_1d[h] += Mxyz[2]
                Myout_1d[h] += Mxyz[4]
                Mzout_1d[h] += Mxyz[6]
            end
            Mxyz[1] = Mxyz[2]
            Mxyz[3] = Mxyz[4]
            Mxyz[5] = Mxyz[6]
        end
    end
    return Mxout_1d, Myout_1d, Mzout_1d
end
#--------------------------------------------------------------------------
# Main
gam = 2 * pi * 4258
resol = 1e-6
tp = 0.005
num_pts = Int(round(tp / resol))
ts = range(-1, 1, length = num_pts)
#--------------------------------------------------------------------------
bw1 = 4.5 / tp
bw2 = 9 / tp
FOV = 10

scale = (2 * pi * bw1) / (gam * FOV)
grad = scale * ones(1, num_pts)
#--------------------------------------------------------------------------
ω_rf = 2*pi*(bw2 / 2).*range(-1,1, length = num_pts)
phase = cumsum(ω_rf,dims=1)* resol

b1 = zeros(ComplexF64, num_pts)
b1 = ones(1,num_pts) .* exp.(1.0im * phase)

ωmax = 2 * pi * 200
#--------------------------------------------------------------------------
x_pts = 1000

offset = 2 * pi * range(-2e3, 2e3, length = x_pts)
Xpos = range(-FOV, FOV, length = x_pts)
#--------------------------------------------------------------------------
# Calling Slow code
M = zeros(3, num_pts)
M[:, 1] = [0.0; 0.0; 1.0]
Mout = @timev Sim1d_mat(Xpos,num_pts,resol,ωmax.*b1,grad,offset,M)
#--------------------------------------------------------------------------
# Calling Fast Code
Mx = zeros(length(Xpos))
My = zeros(length(Xpos))
Mz = ones(length(Xpos))
Mxout_1d,Myout_1d,Mzout_1d = @timev Sim1d_vec(Xpos,num_pts,resol,ωmax.*b1,
                                                grad,offset,Mx,My,Mz)
#--------------------------------------------------------------------------
# Plotting
plot(abs.(Mout[1, :] + 1im * Mout[2, :]))
plot!(abs.(Mxout_1d+1im*Myout_1d))
