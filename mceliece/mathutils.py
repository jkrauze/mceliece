import numpy as np
import numpy.polynomial.polynomial
from sympy.abc import x, alpha
from sympy import GF, Poly, Pow, Add, Symbol
from sympy.polys.galoistools import gf_irreducible, gf_irreducible_p
from sympy import lcm, ZZ
import math
import logging
import flint

log = logging.getLogger("mathutils")


def irreducible_poly_ext_candidate(m, irr_poly, p, var, non_roots=None):
    elems = [0, 1, alpha]
    for e in non_roots:
        elems.append(alpha ** e)
    return Poly(np.concatenate([np.random.choice(elems[1:], size=1, replace=True),
                                np.random.choice(elems, size=m, replace=True)], axis=0), var)


def irreducible_poly(m, p, var):
    return Poly([int(c.numerator) for c in gf_irreducible(m, p, ZZ)], var)


def is_irreducible_poly(poly, p):
    return gf_irreducible_p([int(c) for c in poly.all_coeffs()], p, ZZ)


def first_alpha_power_root(poly, irr_poly, p, elements_to_check=None):
    poly = Poly([(Poly(coeff, alpha) % irr_poly).trunc(p).as_expr() for coeff in poly.all_coeffs()], x)
    test_poly = Poly(1, alpha)
    log.debug(f"testing f:{poly}")
    for i in range(1, p ** irr_poly.degree()):
        test_poly = (Poly(Poly(alpha, alpha) * test_poly, alpha) % irr_poly).set_domain(GF(p))
        if elements_to_check is not None and i not in elements_to_check:
            continue
        value = Poly((Poly(poly.eval(test_poly.as_expr()), alpha) % irr_poly), alpha).trunc(p)
        log.debug(f"testing alpha^{i} f({test_poly})={value}")
        if value.is_zero:
            return i
    return -1


def is_alpha_power_root(poly, i, irr_poly, p):
    return Poly((Poly(poly.eval(alpha ** i), alpha) % irr_poly), alpha).trunc(p).is_zero


def random_perm_matrix(n):
    return np.array([[1 if i == x else 0 for i in range(n)] for x in np.random.permutation(n)])


def random_inv_matrix(n):
    for i in range(1, 1000):
        try:
            candidate = np.random.randint(2, size=(n, n))
            det = int(round(np.linalg.det(candidate)))
            log.debug("Generating random matrix (det={}). Try {}...".format(det, i))
            if det % 2 == 1:
                return candidate
        except ValueError:
            pass
    return None


def order(x, p):
    tx = int(x)
    for i in range(p):
        if tx == 1:
            return i + 1
        tx = (tx * x) % p
    return None


def minimal_poly(i, n, q, irr_poly):
    ti = int(i)
    checked = np.zeros(n, dtype=bool)
    checked[ti] = True
    poly = Poly(x - alpha ** ti, x)
    for k in range(n):
        ti = (ti * q) % n
        if checked[ti]:
            polys = [(Poly(c, alpha) % irr_poly).trunc(q) for c in poly.all_coeffs()]
            for p in polys:
                if p.degree() > 0:
                    raise Exception("Couldn't find minimal polynomial")
            coeffs = [p.nth(0) for p in polys]
            return Poly(coeffs, x)
        checked[ti] = True
        poly = poly * Poly(x - alpha ** ti, x)
    return None


def power_dict(n, irr, p):
    result = {(1,): 0}
    test_poly = Poly(1, alpha)
    for i in range(1, n - 1):
        test_poly = (Poly(Poly(alpha, alpha) * test_poly, alpha) % irr).set_domain(GF(p))
        if tuple(test_poly.all_coeffs()) in result:
            return result
        result[tuple(test_poly.all_coeffs())] = i
    return result


def ring_generate(irr, p):
    power_dict = {(1,): 0}
    element_dict = {0: [1]}
    test_poly = flint.nmod_poly([1], 2)
    alpha_poly = flint.nmod_poly([0, 1], 2)
    for i in range(1, p ** irr.degree() - 1):
        test_poly = (test_poly * alpha_poly) % irr.poly
        coeffs = tuple(int(e) for e in test_poly.coeffs())
        if coeffs in power_dict:
            return power_dict, element_dict
        power_dict[coeffs] = i
        element_dict[i] = list(coeffs)
    return power_dict, element_dict


def get_alpha_power(poly, irr_poly, quotient, p, neg=False):
    poly = (Poly(poly, alpha) % irr_poly).trunc(p)
    if poly.is_zero:
        return 0
    power = quotient[tuple(poly.all_coeffs())]
    if neg:
        power = len(quotient) - power
    return alpha ** power


def get_binary_from_alpha(poly, irr_poly, p):
    poly = (Poly(poly, alpha) % irr_poly).trunc(p)
    result = np.full((irr_poly.degree(),), GF2(0))
    result[:len(poly.all_coeffs())] = [GF2(e) for e in poly.all_coeffs()[::-1]]
    return result


def reduce_to_alpha_power(poly, irr_poly, quotient, p):
    return Poly([get_alpha_power(coeff, irr_poly, quotient, p) for coeff in poly.all_coeffs()], x)


def flatten_frac(muls, m, p, pow_dict):
    if len(muls.args) == 0:
        return (Poly(muls, alpha) % m).set_domain(GF(p))
    log.debug("Dividing: {}".format(muls))
    if len(muls.args) != 2:
        raise Exception("Wrong case")
    inv = muls.args[0]
    add = muls.args[1]
    if type(add) == Pow:
        inv, add = add, inv
    log.debug("num: {}; denum: {}".format(add, inv))
    if type(add) != Add and type(add) != Symbol and type(add) != Pow:
        log.debug(type(add))
        add = int(add)
        if add < 0:
            inv_poly = (Poly(muls.args[0] ** -add) % m).set_domain(GF(p))
            if inv_poly.is_zero:
                raise Exception("Dividing by 0")
            result_pow = pow_dict[tuple(inv_poly.all_coeffs())]
            if result_pow < 0:
                result_pow += len(pow_dict)
            result = Poly(alpha ** result_pow, alpha).set_domain(GF(p))
            log.debug("Dividing result: {}".format(result))
            return (result % m).set_domain(GF(p))
    if (inv.args[1] > 0):
        print(inv.args)
        raise Exception("Wrong case")
    add_poly = (Poly(add) % m).set_domain(GF(p))
    if add_poly.is_zero:
        return add_poly
    i = pow_dict[tuple(add_poly.all_coeffs())]
    inv_poly = (Poly(inv.args[0] ** -inv.args[1]) % m).set_domain(GF(p))
    if inv_poly.is_zero:
        raise Exception("Dividing by 0")
    j = pow_dict[tuple(inv_poly.all_coeffs())]
    result_pow = i - j
    if result_pow < 0:
        result_pow += len(pow_dict)
    result = Poly(alpha ** result_pow, alpha).set_domain(GF(p))
    log.debug("Dividing result: {}".format(result))
    return (result % m).set_domain(GF(p))


def gcd(a, b):
    if abs(a) < abs(b):
        return gcd(b, a)

    while abs(b) > 0:
        q, r = divmod(a, b)
        a, b = b, r

    return a


def ext_euclid(a, b):
    if abs(b) > abs(a):
        (x, y, d) = ext_euclid(b, a)
        return (y, x, d)

    if abs(b) == 0:
        return (1, 0, a)

    x1, x2, y1, y2 = 0, 1, 1, 0
    while abs(b) > 0:
        q, r = divmod(a, b)
        x = x2 - q * x1
        y = y2 - q * y1
        a, b, x2, x1, y2, y1 = b, r, x1, x, y1, y

    return (x2, y2, a)


def ext_euclid_poly(a, b, ring):
    if b.degree() > a.degree():
        (x, y, d) = ext_euclid_poly(b, a, ring)
        return (y, x, d)

    if b.is_zero():
        return (GF2mPoly.from_elem(ring.one()),
                GF2mPoly.from_elem(ring.zero()),
                a)

    x1, x2, y1, y2 = GF2mPoly.from_elem(ring.zero()), \
                     GF2mPoly.from_elem(ring.one()), \
                     GF2mPoly.from_elem(ring.one()), \
                     GF2mPoly.from_elem(ring.zero())
    while not b.is_zero():
        q, r = divmod(a, b)
        x = x2 - q * x1
        y = y2 - q * y1
        a, b, x2, x1, y2, y1 = b, r, x1, x, y1, y

    return (x2, y2, a)


def ext_euclid_poly_alt(a, b, ring, t):
    if b.degree() > a.degree():
        (x, y, d) = ext_euclid_poly_alt(b, a, ring, t)
        return (y, x, d)
    if b.is_zero():
        return (GF2mPoly.from_elem(ring.one()),
                GF2mPoly.from_elem(ring.zero()),
                a)

    x1, x2, y1, y2 = GF2mPoly.from_elem(ring.zero()), \
                     GF2mPoly.from_elem(ring.one()), \
                     GF2mPoly.from_elem(ring.one()), \
                     GF2mPoly.from_elem(ring.zero())
    while not a.degree() <= t // 2 or not x2.degree() <= (t - 1) // 2:
        q, r = divmod(a, b)
        x = x2 - q * x1
        y = y2 - q * y1
        a, b, x2, x1, y2, y1 = b, r, x1, x, y1, y
        log.debug(f"ext euclid: {(x2, y2, a)} B:{b}")

    return (x2, y2, a)


def rref(arr, steps=None):
    m = len(arr)
    n = len(arr[0])
    assert all([len(row) == n for row in arr[1:]]), "Matrix rows have non-uniform length"
    log.debug("rref start:\n{}".format(arr))

    if steps is None:
        steps = min(m, n)

    for k in range(steps):
        i_max = -1
        for i in range(k, m):
            if arr[i, k].n > 0:
                i_max = i
                break
        if i_max == -1:
            continue

        arr[[k, i_max]] = arr[[i_max, k]]

        for i in range(k + 1, m):
            if arr[i, k].n != 0:
                for j in range(k, n):
                    arr[i, j] -= arr[k, j]

    log.debug("rref triangle:\n{}".format(arr))
    m = len(arr)
    n = len(arr[0])
    for k in range(steps - 1, -1, -1):
        for i in range(k - 1, -1, -1):
            if arr[i, k].n != 0:
                for l in range(k, n):
                    arr[i, l] -= arr[k, l]

    log.debug("rref final:\n{}".format(arr))
    return arr


class GF2():

    def __init__(self, n):
        self.n = int(n)

    def __add__(self, other):
        return GF2(self.n ^ other.n)

    def __sub__(self, other):
        return GF2(self.n ^ other.n)

    def __mul__(self, other):
        return GF2(self.n & other.n)

    def __truediv__(self, other):
        return self * other.inv()

    def __neg__(self):
        return self

    def __eq__(self, other):
        if isinstance(other, GF2):
            return self.n == other.n
        if self.n == int(other):
            return True
        return False

    def __abs__(self):
        return abs(self.n)

    def __str__(self):
        return str(self.n)

    def __repr__(self):
        return self.__str__()

    def __int__(self):
        return self.n

    def __divmod__(self, divisor):
        q, r = divmod(self.n, divisor.n)
        return (GF2(q), GF2(r))

    def flip(self):
        return GF2(1 if self.n == 0 else 0)

    def inv(self):
        x, y, d = ext_euclid(self.n, 2)
        return GF2(x)


class GF2mRing:

    def __init__(self, m, irr):
        self.m = m
        self.size = 2 ** m - 1
        self.irr = irr
        self.power_dict, self.element_dict = ring_generate(irr, 2)
        if len(self.power_dict) != self.size:
            raise Exception("Root of given polynomial is not a GF(2^m) ring generator.")

    def pow(self, elem):
        return self.power_dict[tuple(int(e) for e in elem.poly.coeffs())]

    def elem(self, pow):
        return self.element_dict[pow]

    def zero(self):
        return GF2m(GF2Poly.from_list([0]), self)

    def one(self):
        return GF2m(GF2Poly.from_list([1]), self)

    def alpha(self):
        return GF2m(GF2Poly.from_list([0, 1]), self)


class GF2m:

    def __init__(self, n, ring):
        self.n = n
        self.ring = ring

    def __add__(self, other):
        return GF2m(self.n + other.n, self.ring)

    def __sub__(self, other):
        return GF2m(self.n + other.n, self.ring)

    def __mul__(self, other):
        if self.n.is_zero():
            return self
        if type(other) is int and other == 0:
            return GF2m(GF2Poly.from_list([0]), self.ring)
        if other.n.is_zero():
            return other
        return GF2m(GF2Poly.from_list(
            self.ring.elem((self.ring.pow(self.n) + self.ring.pow(other.n)) % self.ring.size)), self.ring)

    def __truediv__(self, other):
        if self.n.is_zero():
            return self
        if (type(other) is int and other == 0) or other.n.is_zero():
            raise Exception("Dividing by zero!")
        return GF2m(GF2Poly.from_list(
            self.ring.elem((self.ring.size + self.ring.pow(self.n) - self.ring.pow(other.n)) % self.ring.size)),
            self.ring)

    def sqrt(self):
        if self.n.is_zero():
            return self
        alpha_pow = self.ring.pow(self.n)
        if alpha_pow % 2 == 1:
            alpha_pow = alpha_pow + self.ring.size
        alpha_pow = alpha_pow // 2
        return GF2m(GF2Poly.from_list(self.ring.elem(alpha_pow)), self.ring)

    def __neg__(self):
        return self

    def __eq__(self, other):
        if isinstance(other, GF2m):
            return self.n == other.n
        if self.n.is_zero() and other == 0:
            return True
        return False

    def __abs__(self):
        return abs(self.n)

    def __str__(self):
        return str(self.n)

    def __repr__(self):
        return self.__str__()


class GF2Matrix:

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=GF2)

    @staticmethod
    def from_list(list):
        if len(list.shape) == 1:
            return GF2Matrix(np.array([GF2(x) for x in list]))
        return GF2Matrix(np.array([[GF2(x) for x in line] for line in list]))

    @staticmethod
    def from_flint(list):
        return GF2Matrix(np.array([GF2(e) for e in list.entries()]).reshape(list.nrows(), list.ncols()))

    def to_numpy(self):
        return np.array([int(x) for x in self.arr.flat])

    def __add__(self, other):
        return GF2Matrix(self.arr + other.arr)

    def __sub__(self, other):
        return GF2Matrix(self.arr + other.arr)

    def __mul__(self, other):
        return GF2Matrix(np.dot(self.arr, other.arr))

    def __truediv__(self, other):
        return self * other.inv()

    def __neg__(self):
        return self

    def __eq__(self, other):
        return isinstance(other, GF2Matrix) and self.arr == other.arr

    def __abs__(self):
        return abs(self.arr)

    def __str__(self):
        return str(self.arr)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, item):
        return self.arr[item]

    def __setitem__(self, key, value):
        self.arr[key] = value

    def T(self):
        return GF2Matrix(self.arr.T)

    def to_flint(self):
        return flint.nmod_mat(self.arr.shape[0], self.arr.shape[1], [int(e) for e in self.arr.flatten()], 2)

    def nullspace(self):
        X, nullity = self.to_flint().nullspace()
        return GF2Matrix.from_flint(X), nullity

    def rref(self):
        X, rank = self.to_flint().rref()
        return GF2Matrix.from_flint(X), rank

    def inv(self):
        eye = flint.nmod_mat(self.arr.shape[0], self.arr.shape[1],
                             [int(e) for e in np.eye(self.arr.shape[0], self.arr.shape[1]).flatten()], 2)
        return GF2Matrix.from_flint(self.to_flint().solve(eye))

    def inv2(self):
        if int(round(np.linalg.det(self.arr.astype(int)))) % 2 != 1:
            raise Exception("Matrix not inversible.")
        return GF2Matrix(
            rref(
                GF2Matrix(
                    np.append(
                        self.arr,
                        GF2Matrix.from_list(np.eye(self.arr.shape[0]).astype(int)).arr, axis=1)
                )
            )[:, self.arr.shape[0]:])


class GF2Poly:

    def __init__(self, poly):
        self.poly = poly

    @staticmethod
    def from_list(arr):
        return GF2Poly(flint.nmod_poly(arr, 2))

    @staticmethod
    def from_numpy(arr):
        return GF2Poly(flint.nmod_poly([int(e) for e in arr.flat], 2))

    def __add__(self, other):
        return GF2Poly(self.poly + other.poly)

    def __sub__(self, other):
        return GF2Poly(self.poly + other.poly)

    def __mul__(self, other):
        return GF2Poly(self.poly * other.poly)

    def __truediv__(self, other):
        return GF2Poly(self.poly / other.poly)

    def __neg__(self):
        return self

    def __eq__(self, other):
        return isinstance(other, GF2Poly) and self.poly == other.poly

    def __abs__(self):
        return abs(self.poly)

    def __str__(self):
        return str(self.poly)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.poly)

    def __divmod__(self, other):
        return divmod(self.poly, other.poly)

    def __mod__(self, other):
        return self.poly % other.poly

    def __getitem__(self, item):
        return self.poly[item]

    def __setitem__(self, key, value):
        self.poly[key] = value

    def is_zero(self):
        return len(self.poly.coeffs()) == 0

    def degree(self):
        return self.poly.degree()


class GF2mPoly:

    def __init__(self, poly):
        self.poly = poly

    @staticmethod
    def from_list(list):
        return GF2mPoly(np.array(list))

    @staticmethod
    def from_elem(elem):
        return GF2mPoly.from_list([elem])

    @staticmethod
    def x(ring):
        return GF2mPoly.from_list([GF2m(GF2Poly.from_list([0]), ring), GF2m(GF2Poly.from_list([1]), ring)])

    def _trim_zeros(self, val):
        return np.trim_zeros(val, 'b') if len(val) > 1 else val

    def __add__(self, other):
        return GF2mPoly(self._trim_zeros(numpy.polynomial.polynomial.polyadd(self.poly, other.poly)))

    def __sub__(self, other):
        return GF2mPoly(self._trim_zeros(numpy.polynomial.polynomial.polyadd(self.poly, other.poly)))

    def __mul__(self, other):
        if other == 0:
            return GF2mPoly(self.poly[0].ring.zero())
        return GF2mPoly(self._trim_zeros(numpy.polynomial.polynomial.polymul(self.poly, other.poly)))

    def __divmod__(self, other):
        result = numpy.polynomial.polynomial.polydiv(self.poly, other.poly)
        return GF2mPoly(self._trim_zeros(result[0])), GF2mPoly(self._trim_zeros(result[1]))

    def __truediv__(self, other):
        return self.__divmod__(other)[0]

    def __mod__(self, other):
        return self.__divmod__(other)[1]

    def __pow__(self, power, modulo=None):
        return GF2mPoly(self._trim_zeros(numpy.polynomial.polynomial.polypow(self.poly, power)))

    def __neg__(self):
        return self

    def __eq__(self, other):
        return isinstance(other, GF2mPoly) and self.poly == other.poly

    def __abs__(self):
        return abs(self.poly)

    def __str__(self):
        return str(self.poly)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.poly)

    def __getitem__(self, item):
        return self.poly[item]

    def __setitem__(self, key, value):
        self.poly[key] = value

    def eval(self, point):
        return numpy.polynomial.polynomial.polyval(point, self.poly)

    def roots(self):
        return numpy.polynomial.polynomial.polyroots(self.poly)

    def degree(self):
        return max(0, len(self.poly) - 1)

    def is_zero(self):
        return len(self.poly) == 0 or all(e.n.is_zero() for e in self.poly.flat)

    def inv_mod(self, other):
        if self.degree() == 0:
            return GF2mPoly.from_list([self.poly[0].ring.one() / self.poly[0]])
        a, b, c = ext_euclid_poly(self, other, self.poly[0].ring)
        return a / c

    def split(self):
        p0_arr = np.sqrt(self.poly[0::2])
        p1_arr = np.sqrt(self.poly[1::2])
        if len(p1_arr) == 0:
            p1_arr = np.array([self.poly[0].ring.zero()])
        p0 = GF2mPoly(p0_arr)
        p1 = GF2mPoly(p1_arr)
        return p0, p1
