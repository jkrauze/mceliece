from mceliece.mathutils import *
from goppa.goppacodegenerator import *
import logging
import timeit

log = logging.getLogger("ntrucipher")


class McElieceCipher:

    def __init__(self, m, n, t):
        self.m = m
        self.n = n
        self.t = t
        self.q = 2
        log.info(f"McEliece(m={self.m},n={self.n},t={self.t},q={self.q},q^m={self.q ** self.m}) initiated")
        self.G = None
        self.H = None
        self.k = None
        self.P = None
        self.P_inv = None
        self.S = None
        self.S_inv = None
        self.Gp = None
        self.g_poly = None

    def generate_random_keys(self):
        global a, b
        ring = GF2mRing(4, GF2Poly.from_list([1, 1, 0, 0, 1]))
        a = GF2mPoly.from_list(
            [GF2m(GF2Poly.from_list([]), ring), GF2m(GF2Poly.from_list([0, 1]), ring)])
        b = GF2mPoly.from_list(
            [GF2m(GF2Poly.from_list([0, 1, 0, 1]), ring), GF2m(GF2Poly.from_list([1]), ring),
             GF2m(GF2Poly.from_list([1, 0, 1]), ring), GF2m(GF2Poly.from_list([0, 1, 1]), ring)])
        print(a)
        print(b)
        print(ext_euclid_poly(a, b, ring))
        print(timeit.timeit('divmod(b,a)', number=10000, globals=globals()))
        self.G, self.H, self.g_poly, self.irr_poly = GoppaCodeGenerator(self.m, self.n, self.t).gen()
        self.g_poly = np.array([(Poly(e, alpha) % self.irr_poly).trunc(2).all_coeffs()[::-1] for e in
                                self.g_poly.all_coeffs()[::-1]])
        self.irr_poly = np.array(self.irr_poly.all_coeffs()[::-1])
        self.k = self.G.arr.shape[0]
        self.P = GF2Matrix.from_list(random_perm_matrix(self.n))
        self.P_inv = self.P.inv()
        self.S = GF2Matrix.from_list(random_inv_matrix(self.k))
        self.S_inv = self.S.inv()
        self.Gp = self.S * self.G * self.P

    def encrypt(self, msg_arr):
        if len(msg_arr) != self.Gp.shape[0]:
            raise Exception(f"Wrong message length. Should be {self.Gp.shape[0]} bits.")
        log.debug(f"msg: {msg_arr}")
        Cp = GF2Matrix.from_list(msg_arr) * GF2Matrix.from_list(self.Gp)
        log.debug(f"C': {Cp}")
        bits_to_flip = np.random.choice(len(Cp), size=self.t, replace=False)
        log.debug(f"bits_to_flip: {bits_to_flip}")
        for b in bits_to_flip:
            Cp[b] = Cp[b].flip()
        log.debug(f"C': {Cp}")
        return Cp

    def decode(self, msg_arr):
        syndrome = msg_arr * GF2Matrix.from_list(self.H.T)
        log.info(f'syndrome:\n{syndrome}')
        self.irr_poly = GF2Poly.from_numpy(self.irr_poly)
        ring = GF2mRing(4, self.irr_poly)
        self.g_poly = GF2mPoly.from_list(
            [GF2m(GF2Poly.from_list([int(e) for e in coeff]), ring) for coeff in self.g_poly])
        print(self.irr_poly)
        print(self.g_poly)
        global a, b
        ring = GF2mRing(4, GF2Poly.from_list([1, 1, 0, 0, 1]))
        a = GF2mPoly.from_list(
            [GF2m(GF2Poly.from_list([0, 1]), ring), GF2m(GF2Poly.from_list([1]), ring)])
        print(f"AAAAA:{a}")
        print((a * a.inv_mod(self.g_poly)) % self.g_poly)

        S_poly = GF2mPoly.from_list(
            [GF2m(GF2Poly.from_list([int(e) for e in syndrome[i * self.m:(i + 1) * self.m].flat]), ring) for i in
             range(len(syndrome) // self.m)])
        log.debug(f'S_poly={S_poly}')
        S_inv_poly = S_poly.inv_mod(self.g_poly)
        log.debug(f'S_inv_poly={S_inv_poly}')
        log.debug(f'S_poly*S_inv_poly (mod g_poly: {self.g_poly})={(S_poly*S_inv_poly)%self.g_poly}')

        H_poly = S_inv_poly + GF2mPoly.from_list(
            [GF2m(GF2Poly.from_list([]), ring), GF2m(GF2Poly.from_list([1]), ring)])
        log.debug(f'H_poly={H_poly}')

        tau_poly = (H_poly ** (1/2))%self.g_poly
        log.debug(f'tau_poly={tau_poly}')


    def decrypt(self, msg_arr):
        if len(msg_arr) != self.H.shape[1]:
            raise Exception(f"Wrong message length. Should be {self.H.shape[1]} bits.")
        log.debug(f"msg: {msg_arr}")
        Cp = GF2Matrix.from_list(msg_arr) * GF2Matrix.from_list(self.P_inv)
        log.debug(f"C': {Cp}")
        Mp = self.decode(Cp)
        log.debug(f"m': {Mp}")
        M = Mp * self.S_inv
        log.debug(f"msg: {M}")
        return M
