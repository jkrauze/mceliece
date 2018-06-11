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
        self.irr_poly = None

    def generate_random_keys(self):
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

    def repair_errors(self, msg_arr, syndrome):
        if type(self.irr_poly) != GF2Poly:
            self.irr_poly = GF2Poly.from_numpy(self.irr_poly)
        ring = GF2mRing(self.m, self.irr_poly)
        if type(self.g_poly) != GF2mPoly:
            self.g_poly = GF2mPoly.from_list(
            [GF2m(GF2Poly.from_list([int(e) for e in coeff]), ring) for coeff in self.g_poly])
        log.debug(f'irr_poly:{self.irr_poly}')
        log.debug(f'g_poly:{self.g_poly}')

        S_poly = GF2mPoly.from_list(
            [GF2m(GF2Poly.from_list([int(e) for e in syndrome[i * self.m:(i + 1) * self.m].flat]), ring) for i in
             range(len(syndrome) // self.m)])
        log.debug(f'S_poly={S_poly}')
        S_inv_poly = S_poly.inv_mod(self.g_poly)
        log.debug(f'S_inv_poly={S_inv_poly}')
        log.debug(f'S_poly*S_inv_poly (mod g_poly: {self.g_poly})={(S_poly*S_inv_poly)%self.g_poly}')

        if S_inv_poly.degree() == 1 and S_inv_poly[1].n.degree() == 0 and S_inv_poly[1].n.poly.coeffs()[0] == 1:
            tau_poly = S_inv_poly
        else:
            g0, g1 = self.g_poly.split()

            log.debug(f"g0:{g0};g1:{g1}")
            log.debug(f"g0^2 + z*g1^2 :{g0**2 + GF2mPoly.x(ring)*g1**2}")
            log.debug(f'g1_inv:{g1.inv_mod(self.g_poly)}')

            w = g0 * g1.inv_mod(self.g_poly)
            log.debug(f"w:{w}")

            H_poly = S_inv_poly + GF2mPoly.from_list(
                [GF2m(GF2Poly.from_list([0]), ring), GF2m(GF2Poly.from_list([1]), ring)])
            log.debug(f'H_poly={H_poly}')

            H0, H1 = H_poly.split()
            log.debug(f'H0={H0};H1={H1}')

            R = H0 + w * H1

            log.debug(f'R:{R}')
            log.debug(f'R^2 mod g:{(R**2)%self.g_poly}')

            b, _, a = ext_euclid_poly_alt(R, self.g_poly, ring, self.t)

            log.debug(f'a:{a};b:{b}')
            log.debug(f'a**2:{a**2};b**2:{b**2};z*b**2:{GF2mPoly.x(ring)*b**2}')
            log.debug(f'b*R mod g:{(b*R)%self.g_poly}')

            tau_poly = (a ** 2 + GF2mPoly.x(ring) * b ** 2)

        log.debug(f'tau_poly={tau_poly}')
        test_elem = ring.one()
        for i in range(len(msg_arr)):
            value = tau_poly.eval(test_elem)
            log.debug(f't(alpha^{i})={value}')
            if value == 0:
                msg_arr[i] = msg_arr[i].flip()
                log.info(f"REPAIRED ERROR ON {i}th POSITION")
            test_elem = test_elem * ring.alpha()

        return msg_arr

    def decode(self, msg_arr):
        if type(msg_arr) != GF2Matrix:
            msg_arr = GF2Matrix.from_list(msg_arr)
        log.debug(f'msg_len:{len(msg_arr)}')
        syndrome = msg_arr * GF2Matrix.from_list(self.H.T)
        log.info(f'syndrome:\n{syndrome}')
        if not all(syndrome.arr == 0):
            msg_arr = self.repair_errors(msg_arr, syndrome)

        D = GF2Matrix.from_list(np.append(self.G.T, msg_arr.arr.reshape(len(msg_arr), 1), axis=1))
        log.debug(f'G^T|c=')
        D_rref = rref(D, steps=self.G.shape[0])
        log.debug(f'I|m=\n{D_rref}')

        return GF2Matrix.from_list(D_rref[:self.G.shape[0], self.G.shape[0]:].flatten())

    def decrypt(self, msg_arr):
        if len(msg_arr) != self.H.shape[1]:
            raise Exception(f"Wrong message length. Should be {self.H.shape[1]} bits.")
        log.debug(f"msg: {msg_arr}")
        Cp = GF2Matrix.from_list(msg_arr) * GF2Matrix.from_list(self.P_inv)
        log.debug(f"C': {Cp}")
        Mp = self.decode(Cp)
        log.debug(f"m': {Mp}")
        M = Mp * GF2Matrix.from_list(self.S_inv)
        log.debug(f"msg: {M}")
        return M.to_numpy()
