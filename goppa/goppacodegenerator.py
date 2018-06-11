from mceliece.mathutils import *
from sympy.polys.galoistools import gf_irreducible, gf_irreducible_p
from sympy import lcm, ZZ
from sympy.abc import x, alpha
from sympy import Matrix
import logging

log = logging.getLogger("goppacodegenerator")


class GoppaCodeGenerator:

    def __init__(self, m, n, t):
        self.m = m
        self.n = n
        self.t = t
        self.q = 2
        log.info(f"GoppaCodeGenerator(m={self.m},n={self.n},t={self.t},q={self.q},q^m={self.q ** self.m}) initiated")

    def gen(self):
        irr_poly = Poly(alpha ** self.m + alpha + 1, alpha).set_domain(GF(self.q))
        if is_irreducible_poly(irr_poly, self.q):
            ring = power_dict(self.q ** self.m, irr_poly, self.q)
        else:
            ring = []
        log.info("irr(q_size: {}): {}".format(len(ring), irr_poly))
        while len(ring) < self.q ** self.m - 1:
            irr_poly = irreducible_poly(self.m, self.q, alpha)
            ring = power_dict(self.q ** self.m, irr_poly, self.q)
            log.info("irr(q_size: {}): {}".format(len(ring), irr_poly))

        log.debug(f"ring={ring}")
        g_poly = Poly(1, x)

        roots_num = max(0, self.q ** self.m - self.n - self.t)

        #g_roots = np.random.choice(range(self.q ** self.m - 1), roots_num, replace=False)
        g_roots = set()
        g_non_roots = list(set(range(self.q ** self.m - 1)) - set(g_roots))

        log.debug(f"g_roots({len(g_roots)})={g_roots}")
        log.debug(f"g_non_roots({len(g_non_roots)})={g_non_roots}")

        for i in g_roots:
            g_poly = (g_poly * Poly(x + alpha ** i, x)).trunc(self.q)

        if g_poly.degree() < self.t:
            small_irr = None
            for i in range(100):
                small_irr = irreducible_poly_ext_candidate(self.t - g_poly.degree(), irr_poly, self.q, x, non_roots=g_non_roots)
                log.debug(f"irr_part_of_g={small_irr}")
                if small_irr.eval(0).is_zero or small_irr.eval(1).is_zero:
                    log.debug(f'roots in trivial case 0:{small_irr.eval(0)} 1:{small_irr.eval(1)}')
                    continue
                first_root = first_alpha_power_root(small_irr, irr_poly, self.q)
                if first_root > 0:
                    log.debug(f"alpha^{first_root} is a root of g(x)={small_irr}")
                    continue
                break
            else:
                raise Exception("irr poly not found")
            g_poly = (g_poly * small_irr).trunc(self.q)

        g_poly = reduce_to_alpha_power(g_poly, irr_poly, ring, self.q)
        log.info(f"g(x)={g_poly}")
        coeffs = g_poly.all_coeffs()

        first_root = first_alpha_power_root(g_poly, irr_poly, self.q, elements_to_check=g_non_roots)
        if first_root > 0:
            raise Exception(f"alpha^{first_root} is a root of g(x)={g_poly}")

        C = Matrix(self.t, self.t, lambda i, j: coeffs[j - i] if 0 <= j - i < self.t else 0)
        log.debug(f"C={C}")
        X = Matrix(self.t, self.n, lambda i, j: (alpha ** ((j * (self.t - i - 1)) % self.n)))
        log.debug(f"X={X}")
        Y = Matrix(self.n, self.n,
                   lambda i, j: get_alpha_power(g_poly.eval(alpha ** g_non_roots[i]), irr_poly, ring, self.q, neg=True)
                   if i == j else 0)
        log.debug(f"Y={Y}")
        H = C * X * Y
        H = Matrix(self.t, self.n, lambda i, j: get_alpha_power(H[i, j], irr_poly, ring, self.q))
        log.debug(f"H=\n{H}")
        H_bin = np.array(
            [np.column_stack([get_binary_from_alpha(e, irr_poly, self.q) for e in line]) for line in
             H.tolist()]).astype(GF2)
        H_bin = GF2Matrix.from_list(H_bin.reshape(-1, H.shape[1]))
        log.info(f"H_bin=\n{H_bin}")
        H_nullspace, nullity = H_bin.nullspace()
        log.debug(f"H_nullspace({nullity})=\n{H_nullspace}")
        G = GF2Matrix(H_nullspace.T()[:nullity])
        log.info(f"G=\n{G}")
        log.debug(f"G*H^T=\n{G * H_bin.T()}")
        return G, H_bin, g_poly, irr_poly
