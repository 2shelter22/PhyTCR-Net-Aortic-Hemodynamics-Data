"""Rigid vascular-tree resistance-inertance and RCR response model."""
import os
from pathlib import Path
import sys

_library_bin = Path(sys.prefix)/'Library/bin'
if os.name == 'nt' and _library_bin.is_dir():
    os.environ['PATH'] = str(_library_bin)+os.pathsep+sys.prefix+os.pathsep+os.environ.get('PATH', '')
_native_dll_directory = os.add_dll_directory(str(_library_bin)) if os.name == 'nt' and _library_bin.is_dir() else None

import numpy as np
from scipy.linalg import expm, solve
from scipy.integrate import solve_ivp

MMHG = 1333.22

def tapered_coefficients(length, r0, r1, mu, rho):

    length, r0, r1 = np.broadcast_arrays(length, r0, r1)
    if np.any(length < 0) or np.any(r0 <= 0) or np.any(r1 <= 0):
        raise ValueError('Nonpositive radius or negative length')
    resistance = 8 * mu * length / np.pi * (r0*r0 + r0*r1 + r1*r1) / (3*r0**3*r1**3)
    inertance = rho * length / (np.pi*r0*r1)
    return resistance, inertance

def affine_propagator(A, dt):

    d = len(A)
    augmented = np.zeros((3*d, 3*d))
    augmented[:d, :d] = A
    augmented[:d, d:2*d] = np.eye(d)
    augmented[d:2*d, 2*d:] = np.eye(d)
    matrix = expm(augmented * dt)
    return matrix[:d, :d], matrix[:d, d:2*d], matrix[:d, 2*d:]

class RigidTreeRCR:
    def __init__(self, inputs):
        self.inputs = inputs
        g, b, maps = inputs['graph'], inputs['boundary'], inputs['maps']
        cfg = inputs['source_configuration']
        self.cfg = cfg
        self.period = float(b['period_seconds'])
        self.times = np.asarray(b['times_seconds'], float)
        self.qin = np.asarray(b['inlet_flow_cm3_s'], float)
        self.T = maps['terminal_sum_map'].astype(float)
        self.edges = np.asarray(g['edge_index'], int)
        self.outlets = np.asarray(b['outlet_nodes'], int)
        self.m = len(self.outlets)
        self.k = self.m - 1
        self.n = len(g['radius_cm'])
        self.root = int(np.flatnonzero(g['inlet_mask'])[0])
        self.incidence = maps['incidence'].astype(float)
        self.interior = maps['interior_nodes']
        self.R, self.L = tapered_coefficients(g['edge_length_cm'],
            g['radius_cm'][self.edges[0]], g['radius_cm'][self.edges[1]],
            cfg['viscosity_g_cm_s'], cfg['density_g_cm3'])
        ports = cfg['ports']
        if [p['graph_node'] for p in ports] != self.outlets.tolist():
            raise ValueError('RCR port order changed')
        rcap = np.sqrt(np.asarray([p['area_cm2'] for p in ports]) / np.pi)
        lengths = np.linalg.norm([p['graph_to_cap_centroid_cm'] for p in ports], axis=1)
        self.Rext, self.Lext = tapered_coefficients(lengths,
            g['radius_cm'][self.outlets], rcap,
            cfg['viscosity_g_cm_s'], cfg['density_g_cm3'])
        self.Rp, self.C, self.Rd = np.asarray(b['rcr'], float).T
        if not np.all(np.asarray(b['rcr']) > 0):
            raise ValueError('Invalid RCR parameters')
        self.Lpath = self.T.T @ (self.L[:, None] * self.T) + np.diag(self.Lext)
        self.Rpath = self.T.T @ (self.R[:, None] * self.T) + np.diag(self.Rext + self.Rp)
        one = np.ones(self.m)
        inverse_one = solve(self.Lpath, one, assume_a='pos')
        self.a = inverse_one / inverse_one.sum()

        self.Z = np.vstack((np.eye(self.k), -np.ones(self.k)))
        self.mass = self.Z.T @ self.Lpath @ self.Z
        if self.k:
            self.H = solve(self.mass, self.Z.T, assume_a='pos')
            self.Av = -self.H @ self.Rpath @ self.Z
            self.Apc = -MMHG * self.H
            self.bv = -self.H @ self.Rpath @ self.a
        else:
            self.H = np.zeros((0, self.m))
            self.Av = np.zeros((0, 0))
            self.Apc = np.zeros((0, 1))
            self.bv = np.zeros(0)
        self.A = np.block([[self.Av, self.Apc],
            [self.Z / (MMHG*self.C[:, None]), -np.diag(1/(self.Rd*self.C))]])
        self.b = np.r_[self.bv, self.a/(MMHG*self.C)]
        self.Bpd = np.vstack((np.zeros((self.k, self.m)), np.diag(1/(self.Rd*self.C))))
        self.pd = np.asarray(cfg['Pd_curves_dyn_cm2'], float)

        if self.pd.shape[0] != self.m or not np.all(self.pd[:, :, 1] == self.pd[:, :1, 1]):
            raise ValueError('RI-RCR requires constant distal-pressure curves')
        self.pd_mmHg = self.pd[:, 0, 1] / MMHG
        self.fpd = self.Bpd @ self.pd_mmHg
        self.path = np.zeros((self.n, len(self.R)))
        children = [[] for _ in range(self.n)]
        for edge, (s, t) in enumerate(self.edges.T):
            children[s].append((edge, t))
        order = [self.root]
        for s in order:
            for edge, t in children[s]:
                self.path[t] = self.path[s]
                self.path[t, edge] = 1
                order.append(int(t))
        self.eigenvalues = np.linalg.eigvals(self.A)
        if np.max(self.eigenvalues.real) >= 0:
            raise ValueError('Passive linear network did not produce a stable state matrix')
        self.propagators = {}

    def inlet(self, time, side='right'):
        phase = np.remainder(np.asarray(time, float), self.period)
        
        phase = np.where(np.abs(phase-self.period) < 1e-10, 0., phase)
        phase = np.round(phase, 12)
        indices = np.searchsorted(self.times, phase, side=side) - 1
        indices = np.clip(indices, 0, len(self.times)-2)
        slopes = np.diff(self.qin)/np.diff(self.times)
        values = self.qin[indices] + slopes[indices]*(phase-self.times[indices])
        return values, slopes[indices]

    def propagator(self, dt):
        key = round(float(dt), 12)
        if key not in self.propagators:
            self.propagators[key] = affine_propagator(self.A, key)
        return self.propagators[key]

    def integrate(self, initial, stop, dt, record_times):
        count = round(stop/dt)
        if abs(count*dt-stop) > 1e-9:
            raise ValueError('Stop not on integration grid')
        
        if np.max(abs(self.times/dt-np.round(self.times/dt))) > 1e-7:
            raise ValueError('Inlet knot not on propagation grid')
        record_indices = np.rint(np.asarray(record_times)/dt).astype(int)
        if np.max(abs(record_indices*dt-record_times)) > 1e-9:
            raise ValueError('Observation times not on propagation grid')
        state = np.asarray(initial, float).copy()
        if state.ndim == 1:
            state = state[:, None]
        Phi, G0, G1 = self.propagator(dt)
        t = np.arange(count) * dt
        q, qdot = self.inlet(t)
        f = q[:, None]*self.b + self.fpd
        delta = f @ G0.T + qdot[:, None]*(G1@self.b)
        result = np.empty((len(record_indices), len(self.A), state.shape[1]))
        lookup = {index: j for j, index in enumerate(record_indices)}
        if 0 in lookup:
            result[lookup[0]] = state
        for index in range(count):
            state = Phi@state + delta[index, :, None]
            if index+1 in lookup:
                result[lookup[index+1]] = state
        return result

    def initial_states(self, dt):
        source = np.r_[np.zeros(self.k),
            np.full(self.m, self.cfg['initial_pressure_mmHg'])]
        zero = np.r_[np.zeros(self.k), self.pd_mmHg]
        forced = self.integrate(np.zeros(len(self.A)), self.period, dt, [self.period])[0, :, 0]
        periodic = solve(np.eye(len(self.A))-expm(self.A*self.period), forced)
        return np.column_stack((source, zero, periodic))

    def observe(self, states, times):
        states = np.asarray(states)
        if states.ndim != 2:
            raise ValueError('Expected time by state for a single initialization')
        qin, qdotin = self.inlet(times)
        derivative = states@self.A.T + qin[:, None]*self.b + self.fpd
        terminal_q = qin[:, None]*self.a + states[:, :self.k]@self.Z.T
        terminal_qdot = qdotin[:, None]*self.a + derivative[:, :self.k]@self.Z.T
        q = terminal_q@self.T.T
        qdot = terminal_qdot@self.T.T
        pc = states[:, self.k:]*MMHG
        root_candidates = terminal_q@self.Rpath.T + terminal_qdot@self.Lpath.T + pc
        root_p = root_candidates.mean(axis=1)
        drops = q*self.R + qdot*self.L
        p = root_p[:, None] - drops@self.path.T
        cap_p = pc + terminal_q*self.Rp
        port_from_graph = p[:, self.outlets] - terminal_q*self.Rext - terminal_qdot*self.Lext
        return dict(pressure_mmHg=p/MMHG, flow_cm3_s=q,
            qin_cm3_s=qin, qin_derivative_cm3_s2=qdotin,
            capacitor_pressure_mmHg=pc/MMHG, outlet_cap_pressure_mmHg=cap_p/MMHG,
            terminal_flow_cm3_s=terminal_q, terminal_flow_derivative_cm3_s2=terminal_qdot,
            state_derivative=derivative,
            mass_residual_cm3_s=q@self.incidence[self.interior].T,
            inlet_balance_cm3_s=terminal_q.sum(axis=1)-qin,
            terminal_path_pressure_spread_mmHg=np.ptp(root_candidates, axis=1)/MMHG,
            cap_map_residual_mmHg=(port_from_graph-cap_p)/MMHG,
            edge_RI_residual_mmHg=(p[:, self.edges[0]]-p[:, self.edges[1]]-drops)/MMHG,
            RCR_equation_residual_cm3_s=MMHG*self.C*derivative[:, self.k:]
                - terminal_q + (pc-MMHG*self.pd_mmHg)/self.Rd)

    def independent_radau(self, initial, times, stop):
        
        solution = solve_ivp(lambda t, y: self.A@y + self.b*float(self.inlet(t)[0]) + self.fpd,
            [0., stop], initial, method='Radau', jac=self.A, t_eval=times,
            rtol=2e-10, atol=2e-9)
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution.y.T, dict(nfev=solution.nfev, njev=solution.njev, nlu=solution.nlu)

    def direct_terminal_saddle(self, state, time):

        qin, qdotin = self.inlet(time)
        q = self.a*qin + self.Z@state[:self.k]
        pc = state[self.k:]*MMHG
        saddle = np.block([[self.Lpath, -np.ones((self.m, 1))],
            [np.ones((1, self.m)), np.zeros((1, 1))]])
        result = solve(saddle, np.r_[-self.Rpath@q-pc, qdotin])
        return result[:self.m], result[-1]/MMHG
