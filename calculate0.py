import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import solve
# 预设定的常数

palette = {
    "G1": "blue",        # 深蓝
    "G2": "green",        # 暗红
    "Nonrec":  "red",   # 深绿
    "Gamma": "purple",     # 深紫
    "grid": "gray"
}

# ===================== 【全局输入参数 无量纲 hbar=m=1】 =====================
Eres0    = 1.0
Ebot     = 0.0
B        = 0.5
Er_up    = Ebot + Eres0 - B      # s=+1 准束缚共振能
Er_down  = Ebot + Eres0 + B      # s=-1
EF       = 2.0
      # 费米能级，固定参数
T        = 0.01      # 温度
U        = 5      #
alphaR   = 1.5    #
V_barrier= 5 # delta势垒强度 2m V /hbar^2 附录A
a        = 1     # 腔长度无量纲
deltaV   = 1     # 外加小偏压 δV 
mmRatio  = 1

# global variable matrix for passing solution of master equation
matP_var = [0, 0, 0, 0]

def getGlobPara():
    headStr  = ""
    headStr += f"Eres0    = {Eres0:.6f}\n"
    headStr += f"Ebot     = {Ebot:.6f}\n"
    headStr += f"B        = {B:.6f}\n"
    headStr += f"Er_up    = {Er_up:.6f}\n"
    headStr += f"Er_down  = {Er_down:.6f}\n"
    headStr += f"Ef       = {EF:.6f}\n"
    headStr += f"T        = {T:.6f}\n"
    headStr += f"U        = {U:.6f}\n"
    headStr += f"alphaR   = {alphaR:.6f}\n"
    headStr += f"V_barri  = {V_barrier:.6f}\n"
    headStr += f"a        = {a:.6f}\n"
    headStr += f"deltaV   = {deltaV:.6f}\n"
    headStr += f"mmRatio  = {mmRatio:.6f}\n"
    return headStr

   



# 扫描变量 in 在主循环设置


# ===================== 1. 基础函数：费米 f0，洛伦兹谱函数 =====================
def f0(eps):
    """费米函数 f0(eps-EF)，任何电压不进这里"""
    x = eps - EF
    x = x / T
    if x > 20.0:
        return 0.0
    elif x < -20.0:
        return 1.0
    res = 1.0/(1.0 + np.exp(x))
 #   print("Fermi distribution: ", res, "\n")
    return res






def k_cav_s(E, s):
    """
    腔内 k_plus^{(s)} 右行，k_minus^{(s)} 左行；附录A
    H_cav^{(s)} = Ebot + p^2/2 + s*alphaR*p - s*B
    E = k^2/2 + s*alphaR*k - s*B
    解：k = -s*alphaR ± sqrt(alphaR**2 +2*s*B +2*(E-Ebot) )
    """
    
    disc = alphaR**2 + 2.0*s*B + 2.0*(E - Ebot)
    if disc < 0:
        print("Equation Delta < 0")
        return 0.0, 0.0
 #   print("spin = ", s, "   disc=", disc)
    sqrt_disc = np.sqrt(disc)
    kp = -s*alphaR + sqrt_disc   # k_+^{(s)}>0 右行
    km = -s*alphaR - sqrt_disc   # k_-^{(s)} 左行
    km = -km                     # use convention k_-^{(s)} > 0 左行
    
  #  print(kp, km)
  #  print("For spin= ", s, "\n")
    return kp, km

 
def k_lead_s(E, s, eVg):    # bias eV in this lead
    """
    Lead k_ext_plus^{(s)} 右行，k_ext_minus^{(s)} 左行；附录A
    H_cav^{(s)} = p^2/2 + s*alphaR*p
    E = k^2/2 + s*alphaR*k
    解：k = -s*alphaR ± sqrt(alphaR**2 + 2*(E-eV) )
    """
 #   alphaR_leads = alphaR
    
    alphaR_leads = 0
    disc = alphaR_leads**2 + 2.0*(E-eVg)
    if disc <= 0:
        print("Equation Delta < 0")
        return 1e-6, 1e-6
 #   print("spin = ", s, "   disc=", disc)
    sqrt_disc = np.sqrt(disc)
    kextp = -s*alphaR_leads + sqrt_disc   # k_+^{(s)}>0 右行
    kextm = -s*alphaR_leads - sqrt_disc   # k_-^{(s)} 左行
    kextm = -kextm                     # use convention k_-^{(s)} > 0 左行
    
  #  print(kp, km)
  #  print("For spin= ", s, "\n")
    return np.sqrt(mmRatio) * kextp, np.sqrt(mmRatio) * kextm


def rt_amplitudes(k_incid, k_trans, k_refl):
    Vb = V_barrier
    denom = 1j * (k_trans + k_refl) - Vb 
    r_ampl = (1j * (k_incid - k_trans) + Vb) / denom
    t_ampl = (1j * (k_incid + k_refl)) / denom
    return r_ampl, t_ampl



#checked correct
def scatter_8amplitude(E, s, eVg_L):      #  # eVg_L: bias applied to left electrode, -eVg_R for right
    """
    附录A δ势垒边界匹配，计算全部8个复散射振幅 (A5‑A12)
    kp = k_+^{(s)} 腔内右行波矢
    km = k_-^{(s)} 腔内左行波矢
    Vb = 2*m*V / hbar**2

    Returns
    -------
    rL_s_left  : r_L^{s, ←}   (A5) 左势垒：腔→左引线 反射振幅
    tL_s_right : t_L^{s, →}   (A6) 左势垒：左引线→腔 透射振幅
    rL_s_right : r_L^{s, →}   (A7) 左势垒：腔→腔向右反射振幅
    tL_s_left  : t_L^{s, ←}   (A8) 左势垒：腔→左引线 透射振幅
    rR_s_right : r_R^{s, →}   (A9) 右势垒：腔→腔向右反射振幅
    tR_s_left  : t_R^{s, ←}   (A10)右势垒：右引线→腔 透射振幅
    rR_s_left  : r_R^{s, ←}   (A11)右势垒：腔→腔向左反射振幅
    tR_s_right : t_R^{s, →}   (A12)右势垒：腔→右引线 透射振幅
    """
    kp, km = k_cav_s(E, s)
    k_extL_p, k_extL_m = k_lead_s(E, s, eVg_L)
    k_extR_p, k_extR_m = k_lead_s(E, s, -eVg_L)
    Vb = V_barrier


    
#    print("8amliude spin = ", s, "  k+, k-", kp, km, "\n")

    # ========= 左势垒 x=0 =========
    # A5: r_L^{s, ←}
    rL_s_left, tL_s_right = rt_amplitudes(k_extL_p, kp, k_extL_m)

    # A7: r_L^{s, →}
    rL_s_right, tL_s_left = rt_amplitudes(km, k_extL_m, kp)

    # ========= 右势垒 x=a =========
    # A9: r_R^{s, →}
    rR_s_right, tR_s_left = rt_amplitudes(k_extR_m, km, k_extR_p)

    # A11: r_R^{s, ←}
    rR_s_left, tR_s_right = rt_amplitudes(kp, k_extR_p, km)
  
#     print("8-scatter amplitudes: ")
#     print( rL_s_left, tL_s_right, rL_s_right, tL_s_left, \
#     rR_s_right, tR_s_left, rR_s_left, tR_s_right)
# #    print("\n")






    return (rL_s_left, tL_s_right,
            rL_s_right, tL_s_left,
            rR_s_right, tR_s_left,
            rR_s_left, tR_s_right)



#checked correct
def compute_bare_gamma(E, s, eVg_L):     #value of bias applied to left lead, the negative for right lead
    """
    由8个散射振幅，计算裸耦合：Gamma_Lin, Gamma_Rin, Gamma_Lout, Gamma_Rout, Gamma_tot
    附录A公式
    """
    rL_s_left, tL_s_right, rL_s_right, tL_s_left, \
    rR_s_right, tR_s_left, rR_s_left, tR_s_right = scatter_8amplitude(E, s, eVg_L)
    k_extL_p, k_extL_m = k_lead_s(E, s, eVg_L)
    k_extR_p, k_extR_m = k_lead_s(E, s, -eVg_L)
    kp, km = k_cav_s(E, s)
 #   print("K in leads: ", k_ext_p, k_ext_m)
    
#    print("4 bare gamma, spin = ", s, "  k+, k-", kp, km, "\n")
#    print( rL_s_left, tL_s_right, rL_s_right, tL_s_left, \
#    rR_s_right, tR_s_left, rR_s_left, tR_s_right, "\n")

   
    alphaR_leads = 0
   # alphaR_leads = alphaR
    v_extL_p = k_extL_p / mmRatio + s * alphaR_leads          # hbar=m=1  v_ext = hbar*k_ext/m + s * alphaR >= 0 
    v_extL_m = k_extL_m / mmRatio - s * alphaR_leads           # - s * alphaR >= 0; use convention "groud speed" in major energy region, i.e., manitude of velocity
    v_extR_p = k_extR_p / mmRatio + s * alphaR_leads          # hbar=m=1  v_ext = hbar*k_ext/m + s * alphaR >= 0 
    v_extR_m = k_extR_m / mmRatio - s * alphaR_leads           # - s * alphaR >= 0; use convention "groud speed" in major energy region, i.e., manitude of velocity
    v_plus = kp + s * alphaR        # v_+^{(s)} = hbar kp/m + s * alphaR >= 0
    v_minus = km - s * alphaR       # v_-^{(s)} = hbar km/m - s * alphaR >= 0; use convention "groud speed" in major energy region, i.e., manitude of velocity

 
 
#    print("V in leads: ", v_ext_p, v_ext_m)
#    print("Test conservation law: ", 1 - (v_minus / v_plus) * np.square(np.abs(rR_s_left)) - (v_ext_p / v_plus) * np.square(np.abs(tR_s_right)))
#    print("Test conservation law: ", (k_ext_p - np.square(np.abs(rL_s_left))*k_ext_m - np.square(np.abs(tL_s_right)) * kp))
#    print("Test conservation law: ", (k_ext_m - np.square(np.abs(rR_s_right))*k_ext_p - np.square(np.abs(tR_s_left)) * km))
#    print("Test conservation law: ", (km - np.square(np.abs(rL_s_right))*kp) - np.square(np.abs(tL_s_left)) * k_ext_m)
# reflected to a different k, the conservation breaks down
    denom = v_plus + v_minus
    if abs(denom) < 1e-12:
        return 0.0,0.0,0.0,0.0,0.0
    gamma0 = v_plus * v_minus / denom

    eps = 1e-12

#    print("a = ", a, " v_plus, v_minus = ", v_plus, ", ", v_minus, "\n")
#    print("v_ext = ", v_ext, "\n")
 
    Gamma_Rout = gamma0 * ((v_extR_p / v_plus) * np.abs(tR_s_right)**2) if np.abs(v_plus) > eps else 0
    Gamma_Lout = gamma0 * ((v_extL_m / v_minus) * np.abs(tL_s_left)**2) if np.abs(v_minus) > eps else 0
    Gamma_Rin  = gamma0 * ((v_minus / v_extR_m) * np.abs(tR_s_left)**2) if np.abs(v_extR_m) > eps else 0
    Gamma_Lin  = gamma0 * ((v_plus / v_extL_p) * np.abs(tL_s_right)**2) if np.abs(v_extL_p) > eps else 0
    Gamma_tot = Gamma_Lout + Gamma_Rout
#    print("bare gammas Lin, Rin, Lout, Rout: ", Gamma_Lin, Gamma_Rin, Gamma_Lout, Gamma_Rout, Gamma_tot, "\n")
    return Gamma_Lin, Gamma_Rin, Gamma_Lout, Gamma_Rout, Gamma_tot


# ===================== 3. 组态能量、ΔE_add ΔE_rmv =====================
def config_energy(n_up, n_down):
    return n_up*Er_up + n_down*Er_down + U * n_up * n_down 


def deltaE_add(n_up, n_down, s):
    """ΔE_add(n,s)=E(n+δ_s)-E(n)"""
    if s == +1:
        return config_energy(n_up+1, n_down) - config_energy(n_up, n_down)
    else:
        return config_energy(n_up, n_down+1) - config_energy(n_up, n_down)


def deltaE_rmv(n_up, n_down, s):
    """ΔE_rmv(n,s)=E(n)-E(n−δ_s)"""
    if s == +1:
        return config_energy(n_up, n_down) - config_energy(n_up-1, n_down)
    else:
        return config_energy(n_up, n_down) - config_energy(n_up, n_down-1)



def W_s(x, Gamma_tot):
    """Lorentz 谱函数 W^{(s)}(x, Gamma_tot)"""
    return (1.0/np.pi) * Gamma_tot / (x**2 + Gamma_tot**2)



# ===================== 4. 卷积积分求组态相关隧穿速率 =====================
def gamma_integral(Gamma_bare, dE_tilde, Gamma_tot, is_out=False):
    """
    通用卷积
    is_out=False: 输入 ∫ f0(ε) W(ε-dE_tilde) dε
    is_out=True:  输出 ∫ (1‑f0(ε)) W(ε‑dE_tilde) dε
    """
    def integrand(eps):
        w = W_s(eps - dE_tilde, Gamma_tot)                                     
        if not is_out:
            return f0(eps) * w
        else:     
            return (1.0 - f0(eps)) * w

   # print("30 * gamma_tot = ", 30 * Gamma_tot, "\n")
    minEps = min(30 * Gamma_tot, 5)
    a0 = max(0, dE_tilde - minEps)
    b0 = max(0, dE_tilde + minEps)
    res, err = quad(integrand, a=a0, b=b0, epsabs=1e-7, epsrel=1e-7)
    return Gamma_bare * res


# ===================== 5. 给定V_gate, sign_dV(+1正向偏压/-1反向偏压),求解总电流I =====================
def solve_total_current(V_gate, sign_dV):
    """
    sign_dV = +1 → δV； sign_dV = -1 → -δV
    全部电势偏移加到点侧ΔE，f0完全不动；无VL VR
    4组态：(0,0), (1,0), (0,1), (1,1) → P0,Pup,Pdown,Pboth
    """

    global matP_var
    dV_eff = sign_dV * deltaV           # bias is applied from left to right, or vice versa
    # 取共振能量处的裸Gamma（准束缚态，在Er处计算散射）
    GammaL_in_up, GammaR_in_up, GammaL_out_up, GammaR_out_up, GammaTot_up = compute_bare_gamma(Er_up, s=+1,   eVg_L=0) #eVg_L=-dV_eff/2.0)    # electron charge e = -1
    GammaL_in_dn, GammaR_in_dn, GammaL_out_dn, GammaR_out_dn, GammaTot_dn = compute_bare_gamma(Er_down, s=-1, eVg_L=0) #eVg_L=-dV_eff/2.0)
    # 4个组态编号
    # idx0: n=(0,0)
    # idx1: n=(1,0) up
    # idx2: n=(0,1) down
    # idx3: n=(1,1) both
    M = np.zeros((4,4))
    b = np.zeros(4)
    # ---------- 组态 (0,0) idx0 ----------
    n0u,n0d = 0,0
    # add spin up from L/R
    dEA_L_up = deltaE_add(n0u,n0d,+1) - V_gate + dV_eff/2.0         # electron charge e = -1
    dEA_R_up = deltaE_add(n0u,n0d,+1) - V_gate - dV_eff/2.0
    g_inL_up_0 = gamma_integral(GammaL_in_up, dEA_L_up, GammaTot_up, is_out=False)
    g_inR_up_0 = gamma_integral(GammaR_in_up, dEA_R_up, GammaTot_up, is_out=False)
    # add spin down from L/R
    dEA_L_dn = deltaE_add(n0u,n0d,-1) - V_gate + dV_eff/2.0
    dEA_R_dn = deltaE_add(n0u,n0d,-1) - V_gate - dV_eff/2.0
    g_inL_dn_0 = gamma_integral(GammaL_in_dn, dEA_L_dn, GammaTot_dn, is_out=False)
    g_inR_dn_0 = gamma_integral(GammaR_in_dn, dEA_R_dn, GammaTot_dn, is_out=False)
    rate_out_0 = 0.0
    rate_in_0 = g_inL_up_0 + g_inR_up_0 + g_inL_dn_0 + g_inR_dn_0
    # from (1,0) remove up
    dER_L_up_1 = deltaE_rmv(1,0,+1) - V_gate + dV_eff/2.0
    dER_R_up_1 = deltaE_rmv(1,0,+1) - V_gate - dV_eff/2.0
    g_outL_up_1 = gamma_integral(GammaL_out_up, dER_L_up_1, GammaTot_up, is_out=True)
    g_outR_up_1 = gamma_integral(GammaR_out_up, dER_R_up_1, GammaTot_up, is_out=True)
    # from (0,1) remove down
    dER_L_dn_2 = deltaE_rmv(0,1,-1) - V_gate + dV_eff/2.0
    dER_R_dn_2 = deltaE_rmv(0,1,-1) - V_gate - dV_eff/2.0
    g_outL_dn_2 = gamma_integral(GammaL_out_dn, dER_L_dn_2, GammaTot_dn, is_out=True)
    g_outR_dn_2 = gamma_integral(GammaR_out_dn, dER_R_dn_2, GammaTot_dn, is_out=True)
    M[0,0] = -rate_in_0
    M[0,1] = g_outL_up_1 + g_outR_up_1
    M[0,2] = g_outL_dn_2 + g_outR_dn_2
    M[0,3] = 0.0
    # ----------组态 (1,0) idx1  n_up=1,n_down=0 ----------
    n1u,n1d = 1,0
    # in from (0,0)
    M[1,0] = g_inL_up_0 + g_inR_up_0
    # out remove up
    dER_L_up_1b = deltaE_rmv(n1u,n1d,+1) - V_gate + dV_eff/2.0
    dER_R_up_1b = deltaE_rmv(n1u,n1d,+1) - V_gate - dV_eff/2.0
    g_outL_up_1b = gamma_integral(GammaL_out_up, dER_L_up_1b, GammaTot_up, is_out=True)
    g_outR_up_1b = gamma_integral(GammaR_out_up, dER_R_up_1b, GammaTot_up, is_out=True)
    # add down to (1,1)
    dEA_L_dn_1 = deltaE_add(n1u,n1d,-1) - V_gate + dV_eff/2.0
    dEA_R_dn_1 = deltaE_add(n1u,n1d,-1) - V_gate - dV_eff/2.0
    g_inL_dn_1 = gamma_integral(GammaL_in_dn, dEA_L_dn_1, GammaTot_dn, is_out=False)
    g_inR_dn_1 = gamma_integral(GammaR_in_dn, dEA_R_dn_1, GammaTot_dn, is_out=False)
    rate_out_1 = g_outL_up_1b + g_outR_up_1b
    rate_in_1  = g_inL_dn_1 + g_inR_dn_1
    M[1,1] = -(rate_out_1 + rate_in_1)
    M[1,2] = 0.0
    # M[1,3]移至idx3伽马计算之后

    # ----------组态 (0,1) idx2 n_up=0,n_down=1 ----------
    n2u,n2d = 0,1
    M[2,0] = g_inL_dn_0 + g_inR_dn_0
    dER_L_dn_2b = deltaE_rmv(n2u,n2d,-1) - V_gate + dV_eff/2.0
    dER_R_dn_2b = deltaE_rmv(n2u,n2d,-1) - V_gate - dV_eff/2.0
    g_outL_dn_2b = gamma_integral(GammaL_out_dn, dER_L_dn_2b, GammaTot_dn, is_out=True)
    g_outR_dn_2b = gamma_integral(GammaR_out_dn, dER_R_dn_2b, GammaTot_dn, is_out=True)
    dEA_L_up_2 = deltaE_add(n2u,n2d,+1) - V_gate + dV_eff/2.0
    dEA_R_up_2 = deltaE_add(n2u,n2d,+1) - V_gate - dV_eff/2.0
    g_inL_up_2 = gamma_integral(GammaL_in_up, dEA_L_up_2, GammaTot_up, is_out=False)
    g_inR_up_2 = gamma_integral(GammaR_in_up, dEA_R_up_2, GammaTot_up, is_out=False)
    rate_out_2 = g_outL_dn_2b + g_outR_dn_2b
    rate_in_2  = g_inL_up_2 + g_inR_up_2
    M[2,2] = -(rate_out_2 + rate_in_2)
    M[2,1] = 0.0
    # M[2,3]移至idx3伽马计算之后
    # ----------组态 (1,1) idx3 n_up=1,n_down=1 ----------
    n3u,n3d =1,1
    M[3,1] = g_inL_dn_1 + g_inR_dn_1
    M[3,2] = g_inL_up_2 + g_inR_up_2
    dER_L_up_3 = deltaE_rmv(n3u,n3d,+1) - V_gate + dV_eff/2.0
    dER_R_up_3 = deltaE_rmv(n3u,n3d,+1) - V_gate - dV_eff/2.0
    g_outL_up_3 = gamma_integral(GammaL_out_up, dER_L_up_3, GammaTot_up, is_out=True)
    g_outR_up_3 = gamma_integral(GammaR_out_up, dER_R_up_3, GammaTot_up, is_out=True)
    dER_L_dn_3 = deltaE_rmv(n3u,n3d,-1) - V_gate + dV_eff/2.0
    dER_R_dn_3 = deltaE_rmv(n3u,n3d,-1) - V_gate - dV_eff/2.0
    g_outL_dn_3 = gamma_integral(GammaL_out_dn, dER_L_dn_3, GammaTot_dn, is_out=True)
    g_outR_dn_3 = gamma_integral(GammaR_out_dn, dER_R_dn_3, GammaTot_dn, is_out=True)

    # 双占据态3向外跃迁矩阵元，此时全部 *_3变量已经定义完毕
    M[1,3] = g_outL_dn_3 + g_outR_dn_3   # 3(idx3) →1(idx1) 抛掉down电子
    M[2,3] = g_outL_up_3 + g_outR_up_3   # 3(idx3) →2(idx2) 抛掉up电子

    # M33：负的态3全部迁出速率之和（up迁出 + down迁出）
    M[3,3] = -(g_outL_up_3+g_outR_up_3 + g_outL_dn_3+g_outR_dn_3)

    # 归一化条件替换最后一行，保持原样
    # print("Matrix0 of M-equation before normalization\n")
    # print(M)
    # print("\n")
     
    M[3,:] = np.array([1.0,1.0,1.0,1.0])
    # print("Matrix of M-equation\n")
    # print(M)
    b[:] = 0.0
    b[3] = 1.0
    P = solve(M,b)
    P0,Pup,Pdn,Pboth = P
    matP_var[0] = P0
    matP_var[1] = Pup
    matP_var[2] = Pdn
    matP_var[3] = Pboth
    # print("Solution of M-equation\n")
    # print(P, "\n")
    # # ========= 按论文公式求左电极电流IL（电子电荷e=-1无量纲） =========

 #   e = -1.0
 #   # up自旋贡献
 #   term_up = (g_inL_up_0 * P0) - (g_outL_up_1b * Pup)
 #   # down自旋贡献
 #   term_dn = (g_inL_dn_0 * P0) - (g_outL_dn_2b * Pdn)
 #   I_L = e*(term_up + term_dn)
 #   return I_L

 #========= 按论文公式求左电极电流IL（电子电荷e=-1无量纲） =========
    e = -1.0
 # up：空态注入 - 单占据up抛出 - 双占据态抛出up
    term_up = (g_inL_up_0 * P0) - (g_outL_up_1b * Pup) - (g_outL_up_3 * Pboth)+g_inL_dn_1*Pup
 # down：空态注入 - 单占据down抛出 - 双占据态抛出down
    term_dn = (g_inL_dn_0 * P0) - (g_outL_dn_2b * Pdn) - (g_outL_dn_3 * Pboth)+g_inL_up_2*Pdn
    I_L = e*(term_up + term_dn)
 # 右电极，严格同结构
    term_up_R = (g_inR_up_0 * P0) - (g_outR_up_1b * Pup) - (g_outR_up_3 * Pboth)+g_inR_dn_1 * Pup
    term_dn_R = (g_inR_dn_0 * P0) - (g_outR_dn_2b * Pdn) - (g_outR_dn_3 * Pboth)+g_inR_up_2 * Pdn
    I_R = e*(term_up_R + term_dn_R)
 #   print(f"I_L={I_L:.6e} I_R={I_R:.6e} I_L+I_R={I_L+I_R:.6e}")
    return I_L









def calcG_Efcurve():
    global EF
# 获取当前 py 文件所在目录
    print("enter G_Ef curve function")
    py_path = os.path.abspath(__file__)
    py_dir = os.path.dirname(py_path)
# 输出文件名，和 py 同目录
    out_file  = os.path.join(py_dir, "Nonrec_Efcurve.txt")
    out_file1 = os.path.join(py_dir, "masterMatrix_Efcurve.txt")
# 模拟你的 X,Y 数据收集
    headStr  = getGlobPara()
    headStr0 = headStr
    headStr1 = headStr
    headStr0 += f"i,     Ef,     I_plus,     I_minus,    G1,     G2,     nonRecip\n" 
    headStr1 += f"i,     Ef,     P0+V, P1+V, P2+V, P3+V,  P0-V, P1-V, P2-V, P3-V,   Pup*Pdown_avg\n" 
    Vg_test = 0.0
    EF_array = np.linspace(0.001, 10, 200)
    data_list = []
    data_list1 = []
    for i, Ef in enumerate(EF_array):
        EF=Ef
        I_plus  = solve_total_current(Vg_test, sign_dV=+1)
        matP0   = matP_var[:]
        I_minus = solve_total_current(Vg_test, sign_dV=-1)
        matP1   = matP_var[:]
        G1 = (I_plus - I_minus)/(2.0*deltaV)
        G2 = (I_plus + I_minus)/(2.0*deltaV) * 10
        nonRecip = (I_plus + I_minus) / (I_plus - I_minus)
        data_list.append([i, Ef, I_plus, I_minus, G1, G2, nonRecip * 100])
     #   pt = 0.5 * (matP0[1] * matP0[2] + matP1[1] * matP1[2] )
        data_list1.append([i, Ef, matP0[0], matP0[1], matP0[2], matP0[3], matP1[0], matP1[1], matP1[2], matP1[3]])
    data_array = np.array(data_list)
    data_array1 = np.array(data_list1)

# ========== 双Y轴绘图部分【修改重点】 ========== # ========== 双Y轴绘图部分 ==========
# 提取变量
    x = data_array[:, 1]        # Ef
    G1 = data_array[:, 4]        # G1
    G2 = data_array[:, 5]      # G2
    nonRecip_pct = data_array[:, 6] # nonRecip，已经 *100

    fig, ax1 = plt.subplots()

    # 左轴：G1、10*G2
    ax1.plot(x, G1, marker='o', linestyle='-', label=r'$G_1$', color=palette["G1"])
    ax1.plot(x, G2, marker='o', linestyle='-', label=r'$10\,G_2$', color=palette["G2"])
    ax1.set_xlabel(r"$E_F$")
    ax1.set_ylabel(r'$G_1,\;10\,G_2$')
    ax1.grid(True)

    # 右轴：非互易因子
    ax2 = ax1.twinx()
#    ax2.plot(x, nonRecip_pct, marker='s', linestyle='-', label=r'$\mathrm{Nonrec}$', color=palette["Nonrec"])
    ax2.plot(x, nonRecip_pct, marker='s', linestyle='-', label=r'$\mathrm{Nonrec}$', color=palette["Nonrec"], zorder=0)
    ax2.set_ylabel(r'$\mathrm{Nonrec}\ (\%)$')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    # ====== 最后画 inset，并且置顶 ======
    ax_inset = ax1.inset_axes([0.7, 0.43, 0.34, 0.34]) 

    p_up_plus   = data_array1[:,3]
    p_down_plus = data_array1[:,4]
    p_up_minus  = data_array1[:,7]
    p_down_minus= data_array1[:,8]

    ax_inset.plot(x, p_up_plus,   color='#1f77b4', linestyle='-',  label=r'$p_\uparrow, +V$')
    ax_inset.plot(x, p_down_plus, color='#ff7f0e', linestyle='-',  label=r'$p_\downarrow, +V$')
    ax_inset.plot(x, p_up_minus,  color='#1f77b4', linestyle='--', label=r'$p_\uparrow, -V$')
    ax_inset.plot(x, p_down_minus,color='#ff7f0e', linestyle='--', label=r'$p_\downarrow, -V$')

    ax_inset.set_xlabel(r'$E_F$', fontsize=9)
    ax_inset.set_ylabel(r'Population', fontsize=9)
    ax_inset.legend(fontsize=7)
    ax_inset.grid(True, alpha=0.4)
    ax_inset.set_zorder(10) # 强制顶层，盖住红线
    plt.tight_layout()
    plt.show()

    # ==============================================  
    with open(out_file, 'w') as f:
         f.write(headStr0)
         np.savetxt(f, data_array, fmt="%.6f")
    print(f"Data print to: \n{out_file}", flush=True)   

    with open(out_file1, 'w') as f1:
         f1.write(headStr1)
         np.savetxt(f1, data_array1, fmt="%.6f")
    print(f"Data print to: \n{out_file1}", flush=True)    


# def calcG_Efcurve():
#     global EF
# # 获取当前 py 文件所在目录
#     print("enter G_Ef curve function")
#     py_path = os.path.abspath(__file__)
#     py_dir = os.path.dirname(py_path)
# # 输出文件名，和 py 同目录
#     out_file  = os.path.join(py_dir, "Nonrec_Efcurve.txt")
#     out_file1 = os.path.join(py_dir, "masterMatrix_Efcurve.txt")
# # 模拟你的 X,Y 数据收集
#     headStr  = getGlobPara()
#     headStr0 = headStr
#     headStr1 = headStr
#     headStr0 += f"i,     Ef,     I_plus,     I_minus,    G1,     G2,     nonRecip\n" 
#     headStr1 += f"i,     Ef,     P0+V, P1+V, P2+V, P3+V,  P0-V, P1-V, P2-V, P3-V,\n" 
#     Vg_test = 0.0
#     EF_array = np.linspace(0.001, 10, 200)
#     data_list = []
#     data_list1 = []
#     for i, Ef in enumerate(EF_array):
#         EF=Ef
#         I_plus  = solve_total_current(Vg_test, sign_dV=+1)
#         matP0   = matP_var[:]
#         I_minus = solve_total_current(Vg_test, sign_dV=-1)
#         matP1   = matP_var[:]
#         G1 = (I_plus - I_minus)/(2.0*deltaV)
#         G2 = (I_plus + I_minus)/(2.0*deltaV) * 10
#         nonRecip = (I_plus + I_minus) / (I_plus - I_minus)
#         data_list.append([i, Ef, I_plus, I_minus, G1, G2, nonRecip * 100])
#         data_list1.append([i, Ef, matP0[0], matP0[1], matP0[2], matP0[3], matP1[0], matP1[1], matP1[2], matP1[3]])
#     data_array = np.array(data_list)
#     data_array1 = np.array(data_list1)

# # ========== 双Y轴绘图部分【修改重点】 ==========
#     # 提取变量
#     x = data_array[:, 1]        # Ef
#     G1 = data_array[:, 4]        # G1
#     G2 = data_array[:, 5]      # G2
#     nonRecip_pct = data_array[:, 6] # nonRecip，已经 *100

#     fig, ax1 = plt.subplots()

#     # 左轴：G1
#     ax1.plot(x, G1, marker='o', linestyle='-', label=r'$G_1$', color=palette["G1"])
#     ax1.plot(x, G2, marker='o', linestyle='-', label=r'$10*G_2$', color=palette["G2"])
#     ax1.set_xlabel(r"$E_F$")
#     ax1.set_ylabel(r'$G_1,\;10\,G_2$')
#     ax1.grid(True)

#     # 右轴：非互易因子
#     ax2 = ax1.twinx()
#     ax2.plot(x, nonRecip_pct, marker='s', linestyle='-', label=r'$\mathrm{Nonrec}$', color=palette["Nonrec"])
#     ax2.set_ylabel(r'$\mathrm{Nonrec}\ (\%)$')

#     # 合并图例
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2)

#     plt.tight_layout()
#     plt.show()
#     # ==============================================  
#     with open(out_file, 'w') as f:
#          f.write(headStr0)
#          np.savetxt(f, data_array, fmt="%.6f")
#     print(f"Data print to: \n{out_file}", flush=True)   

#     with open(out_file1, 'w') as f1:
#          f1.write(headStr1)
#          np.savetxt(f1, data_array1, fmt="%.6f")
#     print(f"Data print to: \n{out_file1}", flush=True)    
  

def calcG_alphaRcurve():
#    global EF
    global alphaR
# 获取当前 py 文件所在目录
    print("enter G_alphaR curve function")
    py_path = os.path.abspath(__file__)
    py_dir = os.path.dirname(py_path)
# 输出文件名，和 py 同目录
    out_file = os.path.join(py_dir, "Nonrec_alphaRCurve.txt")
# 模拟你的 X,Y 数据收集
    headStr  = getGlobPara()
    headStr += f"i,     alphaR,     I_plus,     I_minus,    G1,     G2,     nonRecip\n" 
    Vg_test = 0.0
    alphaR_array = np.linspace(0.0, 2.5, 100)
    data_list = []
    for i, alpha in enumerate(alphaR_array):
        alphaR = alpha
        I_plus  = solve_total_current(Vg_test, sign_dV=+1)
        I_minus = solve_total_current(Vg_test, sign_dV=-1)
        G1 = (I_plus - I_minus)/(2.0*deltaV)
        G2 = (I_plus + I_minus)/(2.0*deltaV) * 10
        nonRecip = (I_plus + I_minus) / (I_plus - I_minus)
        data_list.append([i, alphaR, I_plus, I_minus, G1, G2, nonRecip * 100])
    data_array = np.array(data_list)

# ========== 双Y轴绘图部分【修改重点】 ==========
    # 提取变量
    x = data_array[:, 1]        # Ef
    G1 = data_array[:, 4]        # G1
    G2 = data_array[:, 5]      # G2
    nonRecip_pct = data_array[:, 6] # nonRecip，已经 *100

    fig, ax1 = plt.subplots()

    # 左轴：G1
    ax1.plot(x, G1, marker='o', linestyle='-', label=r'$G_1$', color=palette["G1"])
    ax1.plot(x, G2, marker='o', linestyle='-', label=r'$10*G_2$', color=palette["G2"])
    ax1.set_xlabel(r"$alphaR$")
    ax1.set_ylabel(r'$G_1,\;10\,G_2$')
    ax1.grid(True)

    # 右轴：非互易因子
    ax2 = ax1.twinx()
    ax2.plot(x, nonRecip_pct, marker='s', linestyle='-', label=r'$\mathrm{Nonrec}$', color=palette["Nonrec"])
    ax2.set_ylabel(r'$\mathrm{Nonrec}\ (\%)$')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.show()
    # ==============================================  
    with open(out_file, 'w') as f:
        f.write(headStr)
        np.savetxt(f, data_array, fmt="%.6f")
    print(f"Data print to: \n{out_file}", flush=True)    



def calcG_Ucurve():
    global U
# 获取当前 py 文件所在目录
    print("enter G_Coulomb curve function")
    py_path = os.path.abspath(__file__)
    py_dir = os.path.dirname(py_path)
# 输出文件名，和 py 同目录
    out_file = os.path.join(py_dir, "Nonrec_CoulombCurve.txt")
# 模拟你的 X,Y 数据收集
    headStr  = getGlobPara()
    headStr += f"i,     Ef,     I_plus,     I_minus,    G1,     G2,     nonRecip\n" 
    Vg_test = 0.0
    U_array = np.linspace(0.0, 12, 200)
    data_list = []
    for i, U1 in enumerate(U_array):
        U = U1
        I_plus  = solve_total_current(Vg_test, sign_dV=+1)
        I_minus = solve_total_current(Vg_test, sign_dV=-1)
        G1 = (I_plus - I_minus)/(2.0*deltaV)
        G2 = (I_plus + I_minus)/(2.0*deltaV) * 10
        nonRecip = (I_plus + I_minus) / (I_plus - I_minus)
        data_list.append([i, U1, I_plus, I_minus, G1, G2, nonRecip * 100])
    data_array = np.array(data_list)

# ========== 双Y轴绘图部分【修改重点】 ==========
    # 提取变量
    x = data_array[:, 1]        # Ef
    G1 = data_array[:, 4]        # G1
    G2 = data_array[:, 5]      # G2
    nonRecip_pct = data_array[:, 6] # nonRecip，已经 *100

    fig, ax1 = plt.subplots()

    # 左轴：G1
    ax1.plot(x, G1, marker='o', linestyle='-', label=r'$G_1$', color=palette["G1"])
    ax1.plot(x, G2, marker='o', linestyle='-', label=r'$10*G_2$', color=palette["G2"])
    ax1.set_xlabel(r"$Coulomb$")
    ax1.set_ylabel(r'$G_1,\;10\,G_2$')
    ax1.grid(True)

    # 右轴：非互易因子
    ax2 = ax1.twinx()
    ax2.plot(x, nonRecip_pct, marker='s', linestyle='-', label=r'$\mathrm{Nonrec}$', color=palette["Nonrec"])
    ax2.set_ylabel(r'$\mathrm{Nonrec}\ (\%)$')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.show()
    # ==============================================  
    with open(out_file, 'w') as f:
        f.write(headStr)
        np.savetxt(f, data_array, fmt="%.6f")
    print(f"Data print to: \n{out_file}", flush=True)    




#Calculate I versus delV curve data
def calcI_delVcurve():
    global deltaV
# 获取当前 py 文件所在目录
    print("enter I_delV curve function")
    py_path = os.path.abspath(__file__)
    py_dir = os.path.dirname(py_path)
# 输出文件名，和 py 同目录
    out_file = os.path.join(py_dir, "Nonrec_IdelVcurve.txt")
# 模拟你的 X,Y 数据收集
    headStr  = getGlobPara()
    headStr += f"i,     delV,     I_plus,     I_minus,    G1,     G2,     nonRecip\n" 
    Vg_test = 0.0
    delV_array = np.linspace(0.01, 3, 200)
    data_list = []
 #   data_list.append([0, 0, 0, 0, 0, 0, 0])
    for i, delV in enumerate(delV_array):
        deltaV = delV
        I_plus  = solve_total_current(Vg_test, sign_dV=+1)
        I_minus = solve_total_current(Vg_test, sign_dV=-1)
        G1 = (I_plus - I_minus)/(2.0*deltaV)
        G2 = (I_plus + I_minus)/(2.0*deltaV) * 10
        nonRecip = (I_plus + I_minus) / (I_plus - I_minus)
        data_list.append([i, deltaV, I_plus, I_minus, G1, G2, nonRecip * 100])
    data_array = np.array(data_list)

 # ========== 双Y轴绘图部分【修改重点】 ==========
    # 提取变量
    x = data_array[:, 1]        # Ef
    G1 = data_array[:, 4]        # G1
    G2 = data_array[:, 5]       # G2
    nonRecip_pct = data_array[:, 6] # nonRecip，已经 *100

    fig, ax1 = plt.subplots()

    # 左轴：G1
    ax1.plot(x, G1, marker='o', linestyle='-', label=r'$G_1$', color=palette["G1"])
    ax1.plot(x, G2, marker='o', linestyle='-', label=r'$10*G_2$', color=palette["G2"])
    ax1.set_xlabel("delta V")
    ax1.set_ylabel(r'$G_1, 10*G_2$')
    ax1.grid(True)

    # 右轴：非互易因子
    ax2 = ax1.twinx()
    ax2.plot(x, nonRecip_pct, marker='s', linestyle='-', label=r'$Nonrec', color=palette["Nonrec"])
    ax2.set_ylabel('Nonrec(%)')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.show()
    # ==============================================

# 写入文件
    with open(out_file, 'w') as f:
        f.write(headStr)
        np.savetxt(f, data_array, fmt="%.6f")
    print(f"Data print to: \n{out_file}", flush=True)    









# ===================== 单点测试示例 =====================
if __name__ == "__main__":

#    calcG_Ucurve()
#    calcG_alphaRcurve()
     calcG_Efcurve()
#    calcI_delVcurve()
        
#    Vg_test = 0.0
#    I_plus  = solve_total_current(Vg_test, sign_dV=+1)
#    I_minus = solve_total_current(Vg_test, sign_dV=-1)

#    G1 = (I_plus - I_minus)/(2.0*deltaV)
#    G2 = (I_plus + I_minus)/(2.0*deltaV)

#    print(f"V_gate = {Vg_test:.3f}")
#    print(f"DeltaV  = {deltaV:.6e}")
#    print(f"I(+δV)  = {I_plus:.6e}")
#    print(f"I(-δV)  = {I_minus:.6e}")
#    print(f"G1 = {G1:.6e}")
#    print(f"G2 = {G2:.6e}")
