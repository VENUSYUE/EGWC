"""Edge group walk centrality: initialization, exact greedy, and Schur-MinEGWC.

One loader for every experiment: directions ignored, self loops removed,
duplicate/reciprocal edges merged into one unit-weight edge, largest
connected component kept.

Schur-MinEGWC:
  * ApproxSingleton: electrical JL sketch, q shared solves on one grounded system.
  * per round: b = ceil(sqrt n) highest-degree endpoints of unselected edges as
    terminals; Wilson forests with roots T + sink, adaptive stopping on the
    empirical variance of the sampled Schur correction; Schur recovery of the
    terminal block and potentials; JL sketch of the nonterminal block
    L_UU^{-1} with q shared solves; unit solves refine the short-listed endpoints.
Exact-MinEGWC: sparse LU, streamed inverse columns, exact local gains.
"""
import os
import time
import math
import hashlib
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from numba import njit, prange, set_num_threads

# ----------------------------------------------------------------------------
# graph loading


class Graph:
    def __init__(self, n, u, v):
        self.n = int(n)
        self.u = np.asarray(u, np.int64)
        self.v = np.asarray(v, np.int64)
        self.m = len(self.u)
        self.w = np.ones(self.m)
        self.A = sp.coo_matrix((np.r_[self.w, self.w], (np.r_[self.u, self.v], np.r_[self.v, self.u])),
                               shape=(self.n, self.n)).tocsr()
        self.d = np.asarray(self.A.sum(1)).ravel()
        self.vol = float(self.d.sum())

    def grounded(self, sel_mask):
        """L_F = D - W restricted to nonselected edges."""
        keep = ~sel_mask
        uu, vv, ww = self.u[keep], self.v[keep], self.w[keep]
        W = sp.coo_matrix((np.r_[ww, ww], (np.r_[uu, vv], np.r_[vv, uu])), shape=(self.n, self.n))
        return (sp.diags(self.d) - W).tocsr()


def load(path):
    with open(path, 'rb') as stream:
        raw = stream.read()
    sha = hashlib.sha256(raw).hexdigest()
    txt = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig', errors='ignore')
    is_mtx = txt.startswith('%%MatrixMarket')
    us, vs = [], []
    header_done = not is_mtx
    for line in txt.splitlines():
        s = line.strip()
        if not s or s[0] in '#%':
            continue
        if not header_done:          # MatrixMarket size line
            header_done = True
            continue
        p = s.replace(',', ' ').split()
        if len(p) < 2:
            continue
        try:
            us.append(int(float(p[0]))); vs.append(int(float(p[1])))
        except ValueError:
            continue
    a = np.asarray(us, np.int64); b = np.asarray(vs, np.int64)
    raw_lines = len(a)
    loop = a == b
    a, b = a[~loop], b[~loop]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    nodes, inv = np.unique(np.r_[lo, hi], return_inverse=True)
    lo, hi = inv[:len(lo)], inv[len(lo):]
    key = np.unique(lo * len(nodes) + hi)
    lo, hi = key // len(nodes), key % len(nodes)
    n0 = len(nodes)
    M = sp.coo_matrix((np.ones(len(lo)), (lo, hi)), shape=(n0, n0))
    nc, lab = connected_components(M, directed=False)
    big = np.argmax(np.bincount(lab))
    keepv = np.flatnonzero(lab == big)
    remap = -np.ones(n0, np.int64); remap[keepv] = np.arange(len(keepv))
    ke = (remap[lo] >= 0) & (remap[hi] >= 0)
    g = Graph(len(keepv), remap[lo[ke]], remap[hi[ke]])
    info = dict(file=os.path.basename(path), sha256=sha, raw_edge_lines=int(raw_lines),
                self_loops=int(loop.sum()), raw_n=int(n0), unique_edges=int(len(key)),
                components=int(nc), n=g.n, m=g.m)
    return g, info

# ----------------------------------------------------------------------------
# block Jacobi-preconditioned CG (independent CG per column, vectorized)


@njit(parallel=True, cache=True)
def _pcg_cols(indptr, indices, data, dinv, B, rtol, maxiter):
    n, k = B.shape
    X = np.zeros((n, k)); its = np.zeros(k, np.int64); res = np.zeros(k)
    for j in prange(k):
        x = np.zeros(n); r = B[:, j].copy(); z = r * dinv; p = z.copy()
        rz = 0.0; bn = 0.0
        for i in range(n):
            rz += r[i] * z[i]; bn += r[i] * r[i]
        bn = math.sqrt(bn)
        if bn == 0.0:
            continue
        ap = np.empty(n)
        it = 0
        rn = bn
        while it < maxiter and rn > rtol * bn:
            pap = 0.0
            for i in range(n):
                acc = 0.0
                for e in range(indptr[i], indptr[i + 1]):
                    acc += data[e] * p[indices[e]]
                ap[i] = acc; pap += p[i] * acc
            al = rz / pap
            rn2 = 0.0; rzn = 0.0
            for i in range(n):
                x[i] += al * p[i]; r[i] -= al * ap[i]; z[i] = r[i] * dinv[i]
                rn2 += r[i] * r[i]; rzn += r[i] * z[i]
            be = rzn / rz; rz = rzn
            for i in range(n):
                p[i] = z[i] + be * p[i]
            rn = math.sqrt(rn2); it += 1
        X[:, j] = x; its[j] = it; res[j] = rn / bn
    return X, its, res


def pcg(L, B, rtol=1e-8, maxiter=20000):
    """Jacobi-preconditioned CG, one independent CG per right-hand side."""
    L = L.tocsr()
    single = B.ndim == 1
    B2 = np.ascontiguousarray(B[:, None] if single else B, dtype=np.float64)
    if B2.shape[1] == 0:
        return np.zeros_like(B2), 0, 0.0
    X, its, res = _pcg_cols(L.indptr.astype(np.int64), L.indices.astype(np.int64), L.data,
                            1.0 / L.diagonal(), B2, rtol, maxiter)
    return (X[:, 0] if single else X), int(its.max()), float(res.max())


# ----------------------------------------------------------------------------
# JL sketch helpers


@njit(cache=True)
def _sketch_rhs(eu, ev, ew, ground_idx, ground_w, n, q, seed):
    """Y = C^T Q^T, rows of C: sqrt(w)(e_u - e_v) (u,v >= 0; -1 drops), sqrt(g) e_i."""
    np.random.seed(seed)
    Y = np.zeros((n, q))
    s = 1.0 / math.sqrt(q)
    for k in range(len(eu)):
        sw = math.sqrt(ew[k]) * s
        a = eu[k]; b = ev[k]
        for j in range(q):
            val = sw if np.random.random() < 0.5 else -sw
            if a >= 0:
                Y[a, j] += val
            if b >= 0:
                Y[b, j] -= val
    for k in range(len(ground_idx)):
        sw = math.sqrt(ground_w[k]) * s
        i = ground_idx[k]
        for j in range(q):
            Y[i, j] += sw if np.random.random() < 0.5 else -sw
    return Y


def approx_singleton(g, q=64, seed=1, rtol=1e-8):
    """Shared-reference singleton scores, Eq. (singleton-positive)."""
    t0 = time.perf_counter()
    r = int(np.argmax(g.d))
    I = np.r_[0:r, r + 1:g.n]
    pos = -np.ones(g.n, np.int64); pos[I] = np.arange(len(I))
    L = (sp.diags(g.d) - g.A).tocsr()[I][:, I].tocsr()
    eu, ev = pos[g.u], pos[g.v]
    Y = _sketch_rhs(eu, ev, g.w, np.zeros(0, np.int64), np.zeros(0), len(I), q, seed)
    Z, it, res = pcg(L, Y, rtol=rtol)
    Zf = np.zeros((g.n, q)); Zf[I] = Z
    pi = g.d / g.vol
    zpi = pi[I] @ Z
    Zu, Zv = Zf[g.u], Zf[g.v]
    A_ = Zu - Zv
    C_ = Zu + Zv - 2 * zpi
    Rr = np.einsum('ij,ij->i', A_, A_)
    Ss = np.einsum('ij,ij->i', C_, C_)
    Cc = np.einsum('ij,ij->i', A_, C_)
    V = g.vol; w = g.w
    Rr = np.minimum(Rr, 1.0 / w)  # effective resistance never exceeds 1/w
    score = V / (2 * w) + V * Ss / 4 + w * V * Cc ** 2 / (4 * (2 - w * Rr))
    e1 = int(np.argmin(score))
    return e1, dict(init_seconds=time.perf_counter() - t0, init_q=q, init_cg_iter=int(it),
                    init_cg_res=res, init_score=float(score[e1])), score


def exact_singleton_scores(g):
    """Exact singleton values from the Laplacian pseudo-inverse (n <= 10^4)."""
    L = (sp.diags(g.d) - g.A).toarray()
    Lp = np.linalg.pinv(L, hermitian=True)
    V = g.vol; pi = g.d / V
    x = Lp @ pi; ppp = pi @ x
    u, v = g.u, g.v
    luu, lvv, luv = Lp[u, u], Lp[v, v], Lp[u, v]
    R = luu + lvv - 2 * luv
    S = luu + lvv + 2 * luv - 4 * x[u] - 4 * x[v] + 4 * ppp
    C = luu - lvv - 2 * x[u] + 2 * x[v]
    w = g.w
    return V / (2 * w) + V * S / 4 + w * V * C ** 2 / (4 * (2 - w * R))


# ----------------------------------------------------------------------------
# objective evaluation


def egwc(g, sel_mask, rtol=1e-12):
    L = g.grounded(sel_mask)
    if g.n <= 5000:
        h = splu(L.tocsc()).solve(g.d)
    else:
        h, _, _ = pcg(L, g.d, rtol=rtol, maxiter=200000)
    return float(g.d @ h / g.vol)


def gains_from_entries(g, ids, h, a, b, c):
    w = g.w[ids]
    x, y = h[g.u[ids]], h[g.v[ids]]
    beta = b + 1.0 / w
    return (c * x * x - 2 * beta * x * y + a * y * y) / (g.vol * (a * c - beta * beta))

# ----------------------------------------------------------------------------
# exact greedy


def exact_entries(g, sel_mask):
    lu = splu(g.grounded(sel_mask).tocsc())
    h = lu.solve(g.d)
    diag = np.zeros(g.n); cross = np.zeros(g.m)
    order = np.argsort(g.u, kind='stable')
    off = np.r_[0, np.cumsum(np.bincount(g.u, minlength=g.n))]
    blk = 64
    for s in range(0, g.n, blk):
        vs = np.arange(s, min(s + blk, g.n))
        rhs = np.zeros((g.n, len(vs))); rhs[vs, np.arange(len(vs))] = 1
        cols = lu.solve(rhs)
        for j, v in enumerate(vs):
            diag[v] = cols[v, j]
            ids = order[off[v]:off[v + 1]]
            cross[ids] = cols[g.v[ids], j]
    return h, diag, cross


def exact_gains(g, sel_mask):
    h, diag, cross = exact_entries(g, sel_mask)
    ids = np.flatnonzero(~sel_mask)
    return ids, gains_from_entries(g, ids, h, diag[g.u[ids]], cross[ids], diag[g.v[ids]])


def exact_greedy(g, k, e1):
    t0 = time.perf_counter()
    sel = np.zeros(g.m, bool); sel[e1] = True
    chosen = [e1]
    while len(chosen) < k:
        ids, gain = exact_gains(g, sel)
        e = int(ids[np.argmax(gain)])
        sel[e] = True; chosen.append(e)
    return chosen, time.perf_counter() - t0

# ----------------------------------------------------------------------------
# Wilson forests on the augmented graph (sink = index n)


@njit(cache=True)
def _wilson_one(ptr, nbr, cum, is_root, n1, seed):
    np.random.seed(seed)
    inside = is_root.copy()
    parent = np.full(n1, -1, np.int64)
    rootof = np.full(n1, -1, np.int64)
    for v in range(n1):
        if is_root[v]:
            rootof[v] = v
    steps = 0
    for s in range(n1):
        if inside[s]:
            continue
        v = s
        while not inside[v]:
            lo = ptr[v]; hi = ptr[v + 1]
            t = np.random.random() * cum[hi - 1]
            L = lo; R = hi
            while L < R:
                mid = (L + R) // 2
                if cum[mid] <= t:
                    L = mid + 1
                else:
                    R = mid
            parent[v] = nbr[L]
            v = nbr[L]
            steps += 1
        r = rootof[v]
        v = s
        while not inside[v]:
            inside[v] = True; rootof[v] = r; v = parent[v]
    return rootof, steps


@njit(parallel=True, cache=True)
def _wilson_batch(ptr, nbr, cum, is_root, n1, seeds, Uidx, tid, tptr, tnbr, tw, b):
    """Return assigned terminal id (-1 for sink) for each U vertex and forest,
    plus ||Y_i||_F^2 where Y_i = L_TU X_i."""
    nb = len(seeds)
    out = np.empty((nb, len(Uidx)), np.int32)
    fro = np.zeros(nb)
    steps = np.zeros(nb, np.int64)
    for f in prange(nb):
        rootof, st = _wilson_one(ptr, nbr, cum, is_root, n1, seeds[f])
        steps[f] = st
        for k in range(len(Uidx)):
            out[f, k] = tid[rootof[Uidx[k]]]
        # Frobenius norm of Y_i: row t' sums -w(t',u) over U-neighbours u grouped by root
        acc = np.zeros(b)
        tot = 0.0
        for tp in range(b):
            for e in range(tptr[tp], tptr[tp + 1]):
                u = tnbr[e]
                j = tid[rootof[u]]
                if j >= 0:
                    acc[j] -= tw[e]
            for e in range(tptr[tp], tptr[tp + 1]):
                u = tnbr[e]
                j = tid[rootof[u]]
                if j >= 0 and acc[j] != 0.0:
                    tot += acc[j] * acc[j]
                    acc[j] = 0.0
        fro[f] = tot
    return out, fro, steps


@njit(parallel=True, cache=True)
def _accumulate(G, out):
    nb, nU = out.shape
    for k in prange(nU):
        for f in range(nb):
            j = out[f, k]
            if j >= 0:
                G[k, j] += 1.0


@njit(parallel=True, cache=True)
def _smooth_step(G, Gn, rptr, rcol, rw, U, uidx, tid, dinvU):
    nU, b = G.shape
    for i in prange(nU):
        row = np.zeros(b, np.float32)
        gi = U[i]
        for e in range(rptr[gi], rptr[gi + 1]):
            j = rcol[e]; w = rw[e] * dinvU[i]
            if uidx[j] >= 0:
                row += np.float32(w) * G[uidx[j]]
            else:
                row[tid[j]] += np.float32(w)
        Gn[i] = row


@njit(parallel=True, cache=True)
def _row_quads(G, P, s, rows_u, rows_v, out_cross, diag_out):
    # diag for rows s..s+len(P); cross for (u,v) pairs whose u is in this chunk
    c, b = P.shape
    for i in prange(c):
        acc = 0.0
        for t in range(b):
            acc += G[s + i, t] * P[i, t]
        diag_out[s + i] = acc
    for k in prange(len(rows_u)):
        if rows_u[k] >= s and rows_u[k] < s + c:
            acc = 0.0
            iu = rows_u[k] - s; iv = rows_v[k]
            for t in range(b):
                acc += P[iu, t] * G[iv, t]
            out_cross[k] = acc


def schur_round(g, sel, eps_s, delta_t, q, rng, batch=32, r_min=None, r_max=4096,
                b=None, rtol=1e-8, threads=16, smooth=4, refine=64):
    n = g.n
    stats = {}
    t0 = time.perf_counter()
    cand = np.flatnonzero(~sel)
    # terminals: b highest-degree endpoints of unselected edges
    if b is None:
        b = int(math.ceil(math.sqrt(n)))
    ends = np.unique(np.r_[g.u[cand], g.v[cand]])
    b = min(b, len(ends))
    T = ends[np.argsort(-g.d[ends], kind='stable')[:b]]
    inT = np.zeros(n, bool); inT[T] = True
    U = np.flatnonzero(~inT)
    tid_full = -np.ones(n + 1, np.int64); tid_full[T] = np.arange(b)
    uidx = -np.ones(n, np.int64); uidx[U] = np.arange(len(U))
    # augmented adjacency (nonselected edges + sink edges)
    qj = np.bincount(np.r_[g.u[sel], g.v[sel]], weights=np.r_[g.w[sel], g.w[sel]], minlength=n)
    nu, nv, nw = g.u[cand], g.v[cand], g.w[cand]
    sinkv = np.flatnonzero(qj > 0)
    rows = np.r_[nu, nv, sinkv, np.full(len(sinkv), n)]
    cols = np.r_[nv, nu, np.full(len(sinkv), n), sinkv]
    vals = np.r_[nw, nw, qj[sinkv], qj[sinkv]]
    Aug = sp.csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    Aug.sort_indices()
    ptr = Aug.indptr.astype(np.int64); nbr = Aug.indices.astype(np.int64)
    cum = np.empty(len(Aug.data))
    for i in range(n + 1):
        lo, hi = ptr[i], ptr[i + 1]
        cum[lo:hi] = np.cumsum(Aug.data[lo:hi])
    is_root = np.zeros(n + 1, np.bool_); is_root[T] = True; is_root[n] = True
    # L_TU adjacency rows (terminal -> nonterminal neighbours via nonselected edges)
    Wn = sp.csr_matrix((np.r_[nw, nw], (np.r_[nu, nv], np.r_[nv, nu])), shape=(n, n))
    WTU = Wn[T][:, U].tocsr()
    tptr = WTU.indptr.astype(np.int64)
    tnbr = U[WTU.indices].astype(np.int64)
    tw = WTU.data.copy()
    stats['prep_seconds'] = time.perf_counter() - t0
    # ---- forest sampling with empirical-variance stopping
    t1 = time.perf_counter()
    set_num_threads(threads)
    nU = len(U)
    G = np.zeros((nU, b), np.float32)
    fro_sum = 0.0
    r = 0
    total_steps = 0
    zfac = 1.0 + math.sqrt(math.log(1.0 / delta_t))
    stop_ratio = None
    if r_min is None:
        r_min = int(math.ceil(0.4 / eps_s ** 2))
    if nU > 0:
        while True:
            seeds = rng.integers(1, 2 ** 31 - 1, size=batch)
            out, fro, st = _wilson_batch(ptr, nbr, cum, is_root, n + 1, seeds, U, tid_full,
                                         tptr, tnbr, tw, b)
            total_steps += int(st.sum())
            _accumulate(G, out)
            fro_sum += float(fro.sum())
            r += batch
            Yb = np.asarray(WTU @ G, np.float64) / r
            ybar2 = float((Yb ** 2).sum())
            var = max(fro_sum / r - ybar2, 0.0) / max(r - 1, 1)
            stop_ratio = zfac * math.sqrt(var) / max(math.sqrt(ybar2), 1e-300)
            if (r >= r_min and stop_ratio <= eps_s) or r >= r_max:
                break
        G /= np.float32(r)
    stats.update(sampling_seconds=time.perf_counter() - t1, forests=r, walk_steps=total_steps,
                 stop_ratio=stop_ratio)
    # ---- Harmonic coupling updates: Gamma = P_UU Gamma + P_UT
    t2 = time.perf_counter()
    L = g.grounded(sel)
    if nU > 0 and smooth > 0:
        Wc = Wn.tocsr()
        rptr = Wc.indptr.astype(np.int64); rcol = Wc.indices.astype(np.int64); rw = Wc.data
        dinvU = 1.0 / g.d[U]
        Gn = np.empty_like(G)
        for _ in range(smooth):
            _smooth_step(G, Gn, rptr, rcol, rw, U, uidx, tid_full, dinvU)
            G, Gn = Gn, G
        del Gn
    stats['smooth'] = int(smooth)
    LTT = L[T][:, T].toarray()
    LTU = L[T][:, U]
    LUU = L[U][:, U].tocsr()
    S = LTT + np.asarray(LTU @ G, np.float64) if nU > 0 else LTT
    Minv = np.linalg.inv(S)
    # JL sketch of L_UU^{-1} plus the potential column, solved together
    if nU > 0:
        keepUU = (~inT[nu]) & (~inT[nv])
        eu = uidx[nu[keepUU]]; ev = uidx[nv[keepUU]]; ew = nw[keepUU]
        ground = np.asarray(LUU.sum(1)).ravel()
        gidx = np.flatnonzero(ground > 1e-14)
        Y = _sketch_rhs(eu, ev, ew, gidx.astype(np.int64), ground[gidx], nU, q,
                        int(rng.integers(1, 2 ** 31 - 1)))
        RHS = np.column_stack([g.d[U], Y])
        Z, it, res = pcg(LUU, RHS, rtol=rtol)
        yU = Z[:, 0]; Zs = Z[:, 1:]
        hT = Minv @ (g.d[T] + G.T.astype(np.float64) @ g.d[U])
        hU = yU + G.astype(np.float64) @ hT if nU * b <= 4e7 else yU + np.asarray(G @ hT.astype(np.float32), np.float64)
    else:
        it, res = 0, 0.0
        hT = Minv @ g.d[T]; hU = np.zeros(0); Zs = np.zeros((0, q))
    h = np.zeros(n); h[T] = hT; h[U] = hU
    stats.update(cg_iterations=int(it), cg_residual=float(res))
    # ---- local entries for every candidate
    cu, cv = g.u[cand], g.v[cand]
    a = np.zeros(len(cand)); bb = np.zeros(len(cand)); c = np.zeros(len(cand))
    # diagonal for U vertices: ||Z_i||^2 + Gamma_i M Gamma_i^T ; P = Gamma M chunked
    diagU = np.einsum('ij,ij->i', Zs, Zs) if nU > 0 else np.zeros(0)
    tu, tv = inT[cu], inT[cv]
    # TT
    m_tt = tu & tv
    a[m_tt] = Minv[tid_full[cu[m_tt]], tid_full[cu[m_tt]]]
    c[m_tt] = Minv[tid_full[cv[m_tt]], tid_full[cv[m_tt]]]
    bb[m_tt] = Minv[tid_full[cu[m_tt]], tid_full[cv[m_tt]]]
    if nU > 0:
        # orient mixed and UU edges so that x = U endpoint
        ux = np.where(tu, cv, cu)   # a U endpoint (for UU edges: cu)
        oth = np.where(tu, cu, cv)
        m_mix = tu ^ tv
        m_uu = (~tu) & (~tv)
        diagG = np.zeros(nU)
        crossG = np.zeros(len(cand))
        lu_idx = uidx[ux]
        ia_ = np.flatnonzero(m_uu)
        ru = uidx[cu[ia_]].astype(np.int64); rv = uidx[cv[ia_]].astype(np.int64)
        crs = np.zeros(len(ia_))
        im = np.flatnonzero(m_mix)
        Minv32 = Minv.astype(np.float32)
        mixP = np.zeros(len(cand))
        chunk = max(1, int(2e7 // max(b, 1)))
        for s_ in range(0, nU, chunk):
            e_ = min(s_ + chunk, nU)
            P = G[s_:e_] @ Minv32
            _row_quads(G, P, s_, ru, rv, crs, diagG)
            sel_m = im[(lu_idx[im] >= s_) & (lu_idx[im] < e_)]
            if len(sel_m):
                mixP[sel_m] = P[lu_idx[sel_m] - s_, tid_full[oth[sel_m]]]
        crossG[ia_] = crs
        dU = diagU + diagG
        # mixed: x endpoint in U
        ia = np.flatnonzero(m_mix)
        da = dU[uidx[ux[ia]]]
        dt = Minv[tid_full[oth[ia]], tid_full[oth[ia]]]
        # place according to original orientation (u,v)
        a[ia] = np.where(tu[ia], dt, da)
        c[ia] = np.where(tu[ia], da, dt)
        bb[ia] = mixP[ia]
        iu = np.flatnonzero(m_uu)
        a[iu] = dU[uidx[cu[iu]]]
        c[iu] = dU[uidx[cv[iu]]]
        bb[iu] = np.einsum('ij,ij->i', Zs[uidx[cu[iu]]], Zs[uidx[cv[iu]]]) + crossG[iu]
    gain = gains_from_entries(g, cand, h, a, bb, c)
    stats['recovery_seconds'] = time.perf_counter() - t2
    t3 = time.perf_counter()
    if refine > 0 and nU > 0:
        top = np.argsort(-gain)[:refine]
        tu_, tv_ = inT[cu[top]], inT[cv[top]]
        verts = np.unique(np.r_[cu[top][~tu_], cv[top][~tv_]])
        Eu = np.zeros((nU, len(verts))); Eu[uidx[verts], np.arange(len(verts))] = 1.0
        Zc, _, _ = pcg(LUU, Eu, rtol=rtol)
        col = {int(v_): j for j, v_ in enumerate(verts)}
        for k_, e_ in enumerate(top):
            u_, v_ = int(cu[e_]), int(cv[e_])
            if not inT[u_]:
                a[e_] += Zc[uidx[u_], col[u_]] - diagU[uidx[u_]]
            if not inT[v_]:
                c[e_] += Zc[uidx[v_], col[v_]] - diagU[uidx[v_]]
            if not inT[u_] and not inT[v_]:
                bb[e_] += Zc[uidx[v_], col[u_]] - float(Zs[uidx[u_]] @ Zs[uidx[v_]])
        gref = gains_from_entries(g, cand[top], h, a[top], bb[top], c[top])
        screen = gain.copy(); gain = screen.copy(); gain[top] = gref
        stats['pick'] = int(top[np.argmax(gref)]); stats['_screen'] = screen
        stats['refined'] = int(len(top)); stats['refine_rhs'] = int(len(verts))
    stats['refine_seconds'] = time.perf_counter() - t3
    stats['terminals'] = int(b)
    stats['nonterminals'] = int(nU)
    stats['sdd_rhs'] = int(q + 1) if nU > 0 else 0
    return cand, gain, stats


def schur_greedy(g, k, e1, eps_s=0.1, delta=0.05, q=64, seed=0, track_exact=False, **kw):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    sel = np.zeros(g.m, bool); sel[e1] = True
    chosen = [e1]
    rounds = []
    delta_t = delta / max(k - 1, 1)
    exact_time = 0.0
    while len(chosen) < k:
        cand, gain, st = schur_round(g, sel, eps_s, delta_t, q, rng, **kw)
        j = st.pop('pick', int(np.argmax(gain)))
        screen = st.pop('_screen', gain)
        e = int(cand[j])
        if track_exact:
            te = time.perf_counter()
            ids, eg = exact_gains(g, sel)
            assert np.array_equal(ids, cand)
            rel = np.abs(screen - eg) / eg
            st['max_rel_gain_err'] = float(rel.max())
            st['median_rel_gain_err'] = float(np.median(rel))
            st['alpha_realized'] = float(eg[j] / eg.max())
            exact_time += time.perf_counter() - te
        sel[e] = True; chosen.append(e)
        rounds.append(st)
    total = time.perf_counter() - t0 - exact_time
    return chosen, total, rounds
