# Δmᵢ is mᵢ - M₀; i.e. relative to cutoff magnitude choice
κ(Δmᵢ, K, α) = K*exp(α*Δmᵢ)
de(Δmᵢ, D, ɣ) = (D^2)*10^(2*ɣ*Δmᵢ)
cnorm(q,Δmᵢ, D, ɣ)=(q-1)/(pi*de(Δmᵢ, D, ɣ))
h(t, tⱼ, c, p) = 1 / (t-tⱼ+c)^p
H(t, tⱼ, c, p) = (1 / (-p + 1) )* ((c + (t - tⱼ))^(-p + 1) - (c)^(-p + 1))
v(x, xj, y, yj, Δmᵢ, q, D, ɣ)=cnorm(q,Δmᵢ, D, ɣ)*(1+((x-xj)^2+(y-yj)^2)/(de(Δmᵢ, D, ɣ)))^(-q)


