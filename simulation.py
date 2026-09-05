import numpy as np
import matplotlib.pyplot as plt

class InterFace:
    def __init__(self, dielectric_left, dielectric_right):
        self.eps_left = dielectric_left
        self.eps_right = dielectric_right

    def matrix(self, wavelength):
        n0 = np.sqrt(self.eps_left)
        n1 = np.sqrt(self.eps_right)

        I = (1/(2*n0)) * np.array([[n0 + n1, n1 - n0],
                                   [n1 - n0, n0 + n1]], dtype=complex)
        return I

    def __matmul__(self, other):
        return Component([self]) @ other

class Layer:
    def __init__(self, dielectric, length):
        self.eps = dielectric
        self.length = length

    def matrix(self, wavelength):
        n = np.sqrt(self.eps)
        c = 2.99792458e8

        phase = 2 * np.pi * self.length * n / wavelength

        P = np.array([[np.exp(-1j * phase), 0],
                      [0, np.exp(1j * phase)]], dtype=complex)
        return P

    def __matmul__(self, other):
        return Component([self]) @ other

class Component:
    def __init__(self, elements=None):
        self.elements = [] if elements is None else list(elements)

    def __matmul__(self, other):
        if isinstance(other, Component):
            return Component(self.elements + other.elements)
        if isinstance(other, (InterFace, Layer)):
            return Component(self.elements + [other])
        return NotImplemented

    def matrix(self, wavelength):
        M = np.eye(2, dtype=complex)
        for e in self.elements:
            M = M @ e.matrix(wavelength)
        return M

    def rt(self, wavelength):
        M = self.matrix(wavelength)
        r = M[1, 0] / M[0, 0]
        t = 1 / M[0, 0]
        return r, t

    def RTA(self, wavelength, eps_in, eps_out):
        r, t = self.rt(wavelength)

        n0 = np.sqrt(eps_in)
        ns = np.sqrt(eps_out)

        R = abs(r)**2
        T = (ns.real / n0.real) * abs(t)**2
        A = 1 - R - T
        return R, T, A
    
    def print_matrices(self, wavelength):
        print(f"Component contains {len(self.elements)} elements")
        print("-" * 50)

        for i, e in enumerate(self.elements):
            print(f"[{i}] {e.__class__.__name__}")
            M_e = e.matrix(wavelength)
            print(M_e)
            print("-" * 50)

def Certain_Wavelength(eps_in, eps_out, eps_d_list, re=0):
    if len(eps_d_list) % 2 != 0:
        raise ValueError("eps_d_list length must be even: [eps1,d1,eps2,d2,...].")

    pairs = []
    for i in range(0, len(eps_d_list), 2):
        eps = eps_d_list[i]
        d = eps_d_list[i+1]
        pairs.append((eps, d))

    comp = Component([])
    prev_eps = eps_in

    for eps, d in pairs:
        comp =  comp @ (InterFace(prev_eps, eps) @ Layer(eps, d))
        prev_eps = eps

    comp = comp @ InterFace(prev_eps, eps_out)
    if re:
        R,T,A = comp.RTA(400e-9, eps_in, eps_out)
        print(R, T, A)
    return comp

def plot_T_vs_wavelength(eps_in, eps_out, eps_d_list, wavelength_low, wavelength_high):
    component = Certain_Wavelength(eps_in, eps_out, eps_d_list)
    wavelengths = np.arange(wavelength_low, wavelength_high + 1e-12, 1e-9)
    RR = ((np.sqrt(1) - np.sqrt(11.7))/ (1 + np.sqrt(11.7)))**2

    T_list = []

    for lam in wavelengths:
        _, T, _ = component.RTA(lam, eps_in, eps_out)
        T_list.append(T)

    T_list = np.array(T_list)
    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths , T_list, linewidth=2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission T")
    plt.title("Transmission vs Wavelength")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def TT(eps_in, eps_out, eps_d_list, wavelength_low, wavelength_high):
    component = Certain_Wavelength(eps_in, eps_out, eps_d_list)
    wavelengths = np.arange(wavelength_low, wavelength_high + 1e-12, 1e-9)
    RR = ((np.sqrt(1) - np.sqrt(11.7))/ (1 + np.sqrt(11.7)))**2

    T_list = []

    for lam in wavelengths:
        _, T, _ = component.RTA(lam, eps_in, eps_out)
        T_list.append(T)

    T_list = np.array(T_list)
    
    return T_list

def plot_comparison(eps_in, eps_out, eps_d_list1, eps_d_list2, wavelength_low, wavelength_high):
    """
    比較兩種不同結構的穿透率光譜（自動區分 Structure 1 與 Structure 2）
    """
    # 建立波長陣列
    wavelengths = np.arange(wavelength_low, wavelength_high + 1e-12, 1e-9)
    
    # 計算第一種結構的穿透率
    component1 = Certain_Wavelength(eps_in, eps_out, eps_d_list1)
    T_list1 = []
    for lam in wavelengths:
        _, T, _ = component1.RTA(lam, eps_in, eps_out)
        T_list1.append(T)
        
    # 計算第二種結構的穿透率
    component2 = Certain_Wavelength(eps_in, eps_out, eps_d_list2)
    T_list2 = []
    for lam in wavelengths:
        _, T, _ = component2.RTA(lam, eps_in, eps_out)
        T_list2.append(T)

    # 轉換為 nm 單位供畫圖使用
    wavelengths_nm = wavelengths * 1e9

    # --- 繪圖 ---
    plt.figure(figsize=(9, 6))
    # 使用內建的 "Structure 1" 與 "Structure 2" 作為圖例標籤
    plt.plot(wavelengths_nm, T_list1, linewidth=2, label="Structure 1", color='blue')
    plt.plot(wavelengths_nm, T_list2, linewidth=2, label="Structure 2", color='red', linestyle='--')
    
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission T")
    plt.title("Transmission Spectrum Comparison")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')  # 顯示圖例
    plt.tight_layout()
    plt.show()

"""
component = Certain_Wavelength(1, 1, [11.7, 120e-9, 3.9, 120e-9, 11.7, 120e-9, 3.9, 120e-9, 11.7, 120e-9, 3.9, 120e-9, 11.7, 120e-9, 3.9, 120e-9, 11.7, 120e-9, 3.9, 120e-9])
component.print_matrices(500e-9)

"""



"""
eps_in = 1
eps_out = 1

eps_d_list = [11.7, 120e-9, 3.9, 120e-9, 11.7, 120e-9, 3.9, 120e-9, 11.7, 120e-9, 3.9, 120e-9, 11.7, 120e-9, 3.9, 120e-9, 11.7, 120e-9, 3.9, 120e-9]

plot_T_vs_wavelength(
    eps_in=eps_in,
    eps_out=eps_out,
    eps_d_list=eps_d_list,
    wavelength_low=300e-9,
    wavelength_high=800e-9
)

eps_in = 1
eps_out = 1

eps_d_list = [11.7, 50e-9, 3.9, 50e-9, 11.7, 50e-9, 3.9, 50e-9, 11.7, 50e-9, 3.9, 50e-9, 11.7, 50e-9, 3.9, 50e-9, 11.7, 50e-9, 3.9, 50e-9]

plot_T_vs_wavelength(
    eps_in=eps_in,
    eps_out=eps_out,
    eps_d_list=eps_d_list,
    wavelength_low=200e-9,
    wavelength_high=800e-9
)
"""