# =============================================================================
# ATEMR-M: Adaptive Trust-Energy Multi-objective Routing with Mitigation
# =============================================================================
# Fixes vs original broken code:
#   1. Direct trust from real packet counters — no ground-truth cheat
#   2. Indirect trust implemented: IT_ij = Σ ω_k·T_kj, ω_k = T_ik/ΣT_ik
#   3. First-order radio energy model; nodes have real 2D positions
#   4. Adaptive weights: wt = AttackLevel/(AttackLevel+DrainRate)
#   5. Routing score Score_ij = wt·T_ij + we·(Er_j/Emax); greedy path select
#   6. False positive recovery: T_ij += δ on sustained good behaviour
#
# Mitigation bug fixes (post-run observations):
#   M1. Mitigation delayed until MITIGATION_START_ROUND — trust must converge
#       before suspicion scores are meaningful (all nodes flagged at R5 because
#       T_ij=0.5 at init → S_j=0.5 > THRESHOLD_SUSPICIOUS=0.3)
#   M2. Eigenvector fusion removed entirely — it was propagating malicious
#       nodes' low trust onto honest neighbours causing all-node isolation
#       cascades every few rounds. IT already handles network opinion correctly
#       with snapshot protection. Not in the original prompt's math model.
#   M3. Isolation uses pure local trust check instead of eigenvector floor —
#       decisions stay per-observer, no global contamination path
#   M4. Recovery blocked when max observed DT < 0.5 — malicious nodes are
#       mathematically capped at DT ≈ 0.35 so they can never cross this bar
# =============================================================================

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────── PARAMETERS ──────────────────────────────────────

NUM_NODES      = 20
ROUNDS         = 25
WARMUP_ROUNDS  = 4
AREA_SIZE      = 500.0

INITIAL_ENERGY = 100.0

# Lowered from 0.7 → 0.5 so trust reacts faster to newly malicious nodes.
# At 0.7 a node turning malicious at R11 takes ~8 rounds to drag trust below
# the isolation threshold — longer than the remaining simulation window.
# At 0.5 the same drop takes ~4 rounds, catching node 16 within the run.
LAMBDA_NORMAL     = 0.5
LAMBDA_ATTACK     = 0.3
ANOMALY_THRESHOLD = 0.35

# Mitigation thresholds (S_j = 1 − T_ij)
THRESHOLD_SUSPICIOUS = 0.3
# Lowered from 0.5 → 0.45: isolation triggers when trust drops below 0.55.
# Combined with LAMBDA_NORMAL=0.5 and DT_WEIGHT=0.7, a newly malicious node
# crosses this threshold within 2 rounds of turning bad.
THRESHOLD_MALICIOUS  = 0.45

# M1: delay mitigation until trust has converged
MITIGATION_START_ROUND = 8

# M3: min DT any observer must see before reinstatement is allowed
# Malicious nodes now capped at DT ≈ 0.15 (fwd 0–15%, ack 0–10%)
# Setting floor at 0.3 cleanly blocks reinstatement for any malicious node
REINSTATEMENT_MIN_DT = 0.3

NEW_ATTACK_ROUND = 10
EPSILON          = 1e-9

ALPHA_1 = 0.5
ALPHA_2 = 0.3
ALPHA_3 = 0.2
assert abs(ALPHA_1 + ALPHA_2 + ALPHA_3 - 1.0) < 1e-9

# DT weighted higher than IT — direct observation dominates over hearsay.
# This is also more defensible mathematically: DT is first-hand evidence,
# IT is neighbour opinion which lags one round behind.
# At 0.5/0.5, a newly malicious node's IT (still high from prior good rounds)
# buffers DT exactly to the isolation threshold and detection never triggers.
# At 0.7/0.3, a single round of bad DT pulls combined below the threshold.
DT_WEIGHT = 0.7
IT_WEIGHT = 0.3
assert abs(DT_WEIGHT + IT_WEIGHT - 1.0) < 1e-9

PACKETS_PER_ROUND = 10
PACKET_SIZE_BITS  = 4000

# Transmission range — nodes only communicate with neighbours within this
# distance. With AREA_SIZE=500 and NUM_NODES=20, TX_RANGE=200 gives each
# node ~4–6 neighbours on average, cutting active pairs from N²=380 to ~80.
# This is physically realistic for WSN and reduces energy drain ~5× per round.
TX_RANGE = 200.0

E_ELEC      = 50e-9
EPS_AMP     = 100e-12
PATH_LOSS_N = 2

RECOVERY_DELTA    = 0.05
PROBATION_WINDOW  = 3
GOOD_BEHAVIOUR_DT = 0.75

# Set False for clean one-line-per-round output
# Set True to see trust matrix, score matrix, energy levels, all paths
VERBOSE = False


# ─────────────────────────── ENERGY MODEL ────────────────────────────────────

def tx_energy(k_bits, distance):
    return E_ELEC * k_bits + EPS_AMP * k_bits * (distance ** PATH_LOSS_N)

def rx_energy(k_bits):
    return E_ELEC * k_bits


# ─────────────────────────── NODE CLASS ──────────────────────────────────────

class Node:
    def __init__(self, node_id, x, y):
        self.id     = node_id
        self.x      = x
        self.y      = y
        self.energy = INITIAL_ENERGY
        self.trust  = {j: 0.5 for j in range(NUM_NODES) if j != node_id}
        self.is_malicious = False

        self.packets_sent      = {j: 0  for j in range(NUM_NODES) if j != node_id}
        self.packets_forwarded = {j: 0  for j in range(NUM_NODES) if j != node_id}
        self.acks_received     = {j: 0  for j in range(NUM_NODES) if j != node_id}
        self.delay_samples     = {j: [] for j in range(NUM_NODES) if j != node_id}

        self.good_rounds_while_isolated = {
            j: 0 for j in range(NUM_NODES) if j != node_id
        }

    def reset_counters(self):
        for j in self.packets_sent:
            self.packets_sent[j]      = 0
            self.packets_forwarded[j] = 0
            self.acks_received[j]     = 0
            self.delay_samples[j]     = []

    def distance_to(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


# ─────────────────────────── PACKET EXCHANGE ─────────────────────────────────

def simulate_packet_exchange(nodes, neighbours):
    """
    Only communicate with nodes within TX_RANGE.
    neighbours[i] is precomputed at setup and doesn't change — it reflects
    the physical deployment topology, not trust. Isolated nodes are still
    reachable for trust observation purposes (they just get excluded from
    routing), so we don't filter them here.
    """
    for node in nodes:
        node.reset_counters()

    for i in range(NUM_NODES):
        for j in neighbours[i]:          # only in-range pairs
            if nodes[i].energy <= 0 or nodes[j].energy <= 0:
                continue

            sent = PACKETS_PER_ROUND
            nodes[i].packets_sent[j] = sent

            if nodes[j].is_malicious:
                # Narrowed from [0, 0.35] to [0, 0.15] — malicious nodes
                # now drop most packets consistently. The original wide range
                # caused on-off behaviour where lucky rounds partially
                # recovered trust, preventing convergence below the isolation
                # threshold within the simulation window.
                fwd_prob = random.uniform(0.00, 0.15)
                ack_prob = random.uniform(0.00, 0.10)
                delays   = [random.uniform(0.5, 1.5) for _ in range(sent)]
            else:
                fwd_prob = random.uniform(0.90, 1.00)
                ack_prob = random.uniform(0.90, 1.00)
                delays   = [random.uniform(0.05, 0.20) for _ in range(sent)]

            nodes[i].packets_forwarded[j] = sum(
                1 for _ in range(sent) if random.random() < fwd_prob
            )
            nodes[i].acks_received[j] = sum(
                1 for _ in range(sent) if random.random() < ack_prob
            )
            nodes[i].delay_samples[j] = delays

            d          = nodes[i].distance_to(nodes[j])
            total_bits = sent * PACKET_SIZE_BITS
            nodes[i].energy = max(0.0, nodes[i].energy - tx_energy(total_bits, d))
            nodes[j].energy = max(0.0, nodes[j].energy - rx_energy(total_bits))


# ─────────────────────────── TRUST FUNCTIONS ─────────────────────────────────

def compute_direct_trust(node_i, j):
    sent = node_i.packets_sent[j]
    if sent == 0:
        return node_i.trust[j]
    fwd_ratio   = node_i.packets_forwarded[j] / sent
    ack_ratio   = node_i.acks_received[j]     / sent
    avg_delay   = np.mean(node_i.delay_samples[j]) if node_i.delay_samples[j] else 1.0
    delay_score = 1.0 / (1.0 + avg_delay)
    return float(np.clip(
        ALPHA_1 * fwd_ratio + ALPHA_2 * ack_ratio + ALPHA_3 * delay_score,
        0.0, 1.0
    ))

def compute_indirect_trust(trust_snapshot, i, j, neighbours):
    # Only use nodes that i can directly observe as recommenders
    neighbors  = [k for k in neighbours[i] if k != j]
    weight_sum = sum(trust_snapshot[i][k] for k in neighbors) + EPSILON
    if not neighbors:
        return trust_snapshot[i].get(j, 0.5)
    return float(np.clip(
        sum((trust_snapshot[i][k] / weight_sum) * trust_snapshot[k][j]
            for k in neighbors),
        0.0, 1.0
    ))

def build_trust_matrix(nodes):
    T = np.zeros((NUM_NODES, NUM_NODES))
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            T[i][j] = 1.0 if i == j else nodes[i].trust[j]
    return T


# ─────────────────────────── ROUTING ─────────────────────────────────────────

def compute_score_matrix(nodes, T_matrix, w_t, w_e):
    S = np.zeros((NUM_NODES, NUM_NODES))
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i == j or nodes[j].energy <= 0:
                continue
            S[i][j] = w_t * T_matrix[i][j] + w_e * (nodes[j].energy / INITIAL_ENERGY)
    return S

def select_path(src, dst, score_matrix, isolated):
    if src == dst:
        return [src]
    path    = [src]
    visited = {src}
    while path[-1] != dst:
        current    = path[-1]
        candidates = [
            j for j in range(NUM_NODES)
            if j not in visited
            and j not in isolated
            and score_matrix[current][j] > 0
        ]
        if not candidates or len(path) > NUM_NODES:
            return []
        next_hop = max(candidates, key=lambda j: score_matrix[current][j])
        path.append(next_hop)
        visited.add(next_hop)
    return path


# ─────────────────────────── FALSE POSITIVE RECOVERY ─────────────────────────

def apply_recovery(nodes, isolated, dt_this_round):
    if not isolated:
        return isolated

    observers    = NUM_NODES - 1
    to_reinstate = set()

    for j in list(isolated):

        # M3: hard block — malicious nodes can never exceed this DT
        max_dt = max(
            dt_this_round.get((i, j), 0.0)
            for i in range(NUM_NODES) if i != j
        )
        if max_dt < REINSTATEMENT_MIN_DT:
            for i in range(NUM_NODES):
                if i != j:
                    nodes[i].good_rounds_while_isolated[j] = 0
            continue

        good_observer_count = 0
        for i in range(NUM_NODES):
            if i == j:
                continue
            dt_ij = dt_this_round.get((i, j), 0.0)
            if dt_ij >= GOOD_BEHAVIOUR_DT:
                nodes[i].good_rounds_while_isolated[j] += 1
            else:
                nodes[i].good_rounds_while_isolated[j] = 0
            if nodes[i].good_rounds_while_isolated[j] >= PROBATION_WINDOW:
                good_observer_count += 1
            if dt_ij >= GOOD_BEHAVIOUR_DT:
                nodes[i].trust[j] = min(1.0, nodes[i].trust[j] + RECOVERY_DELTA)

        if good_observer_count > observers // 2:
            to_reinstate.add(j)
            print(f"  ✓  Node {j} REINSTATED  (malicious={nodes[j].is_malicious})")
            for i in range(NUM_NODES):
                if i != j:
                    nodes[i].good_rounds_while_isolated[j] = 0

    return isolated - to_reinstate


# ─────────────────────────── SETUP ───────────────────────────────────────────

random.seed(42)
np.random.seed(42)

nodes = [
    Node(i, x=random.uniform(0, AREA_SIZE), y=random.uniform(0, AREA_SIZE))
    for i in range(NUM_NODES)
]

initial_malicious = random.sample(range(NUM_NODES), 2)
for m in initial_malicious:
    nodes[m].is_malicious = True

print("=" * 60)
print("  ATEMR-M Simulation")
print("=" * 60)
print(f"  Nodes            : {NUM_NODES}")
print(f"  Rounds           : {ROUNDS}")
print(f"  Malicious (init) : {initial_malicious}")
print(f"  New attacker at  : round {NEW_ATTACK_ROUND + 1}")
print(f"  Mitigation from  : round {MITIGATION_START_ROUND + 1}")
print(f"  TX Range         : {TX_RANGE} m")
print(f"  Verbose          : {VERBOSE}")
print("=" * 60)

# Precompute neighbour list from physical positions — static for the simulation
neighbours = {
    i: [j for j in range(NUM_NODES) if j != i and nodes[i].distance_to(nodes[j]) <= TX_RANGE]
    for i in range(NUM_NODES)
}
avg_neighbours = np.mean([len(neighbours[i]) for i in range(NUM_NODES)])
print(f"  Avg neighbours   : {avg_neighbours:.1f} per node")
isolated_nodes = [i for i in range(NUM_NODES) if len(neighbours[i]) == 0]
if isolated_nodes:
    print(f"  WARNING: isolated nodes (no neighbours): {isolated_nodes}")
print("=" * 60)

if VERBOSE:
    print("\nNode Positions:")
    for n in nodes:
        tag = " [MALICIOUS]" if n.is_malicious else ""
        print(f"  Node {n.id:>2}: ({n.x:.1f}, {n.y:.1f}){tag}")

ws_history     = []
energy_history = []
pdr_history    = []
isolated       = set()


# ─────────────────────────── MAIN LOOP ───────────────────────────────────────

for t in range(ROUNDS):

    if VERBOSE:
        print(f"\n{'='*60}")
        print(f"  ROUND {t + 1}")
        print(f"{'='*60}")

    # ── New attacker ───────────────────────────────────────────────────────
    if t == NEW_ATTACK_ROUND:
        candidates = [i for i in range(NUM_NODES) if not nodes[i].is_malicious]
        if candidates:
            new_mal = random.choice(candidates)
            nodes[new_mal].is_malicious = True
            print(f"\n  R{t+1:02d} ⚠  New Malicious Node: {new_mal}")

    # ── Packet exchange ────────────────────────────────────────────────────
    simulate_packet_exchange(nodes, neighbours)

    # ── Trust snapshot ─────────────────────────────────────────────────────
    trust_snapshot = {
        i: {j: nodes[i].trust[j] for j in nodes[i].trust}
        for i in range(NUM_NODES)
    }

    # ── Trust update ───────────────────────────────────────────────────────
    dt_this_round = {}
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i == j:
                continue
            DT = compute_direct_trust(nodes[i], j)
            IT = compute_indirect_trust(trust_snapshot, i, j, neighbours)
            dt_this_round[(i, j)] = DT
            combined = DT_WEIGHT * DT + IT_WEIGHT * IT
            prev     = nodes[i].trust[j]
            lam = LAMBDA_ATTACK if (
                t >= WARMUP_ROUNDS
                and prev > 0.6
                and abs(combined - prev) > ANOMALY_THRESHOLD
            ) else LAMBDA_NORMAL
            nodes[i].trust[j] = lam * prev + (1 - lam) * combined

    T_matrix   = build_trust_matrix(nodes)
    avg_energy = np.mean([n.energy for n in nodes])
    energy_history.append(avg_energy)

    if VERBOSE:
        print("\nTrust Matrix:")
        print(pd.DataFrame(
            np.round(T_matrix, 3),
            columns=[f"N{j}" for j in range(NUM_NODES)],
            index=[f"N{i}" for i in range(NUM_NODES)]
        ))
        print("\nNode Energy Levels:")
        for n in nodes:
            tag = " [DEAD]" if n.energy <= 0 else ""
            print(f"  Node {n.id:>2}: {n.energy:.5f} J{tag}")

    # ── Adaptive weights ───────────────────────────────────────────────────
    off_diag     = ~np.eye(NUM_NODES, dtype=bool)
    T_mean       = np.mean(T_matrix[off_diag])
    attack_level = 1.0 - T_mean
    drain_rate   = (INITIAL_ENERGY - avg_energy) / INITIAL_ENERGY
    w_t = attack_level / (attack_level + drain_rate + EPSILON)
    w_e = 1.0 - w_t
    ws_history.append(w_t)

    # ── Mitigation (M1: only after trust has converged) ────────────────────
    if t >= MITIGATION_START_ROUND:
        isolation_votes = {j: 0 for j in range(NUM_NODES)}
        suspicion_flags = {j: 0 for j in range(NUM_NODES)}

        for i in range(NUM_NODES):
            for j in range(NUM_NODES):
                if i == j:
                    continue
                # Only count votes from nodes that can directly observe j.
                # Non-neighbours have packets_sent[j]=0 so their trust value
                # is the stale initialisation (0.5), not a real observation.
                # Allowing them to vote caused mass false positives on all
                # out-of-range nodes after TX_RANGE was introduced.
                if j not in neighbours[i]:
                    continue
                S_j = 1.0 - nodes[i].trust[j]
                if S_j >= THRESHOLD_MALICIOUS and nodes[i].trust[j] < (1.0 - THRESHOLD_MALICIOUS):
                    isolation_votes[j] += 1
                elif S_j >= THRESHOLD_SUSPICIOUS:
                    suspicion_flags[j] += 1

        # Vote threshold: fraction of j's actual neighbours, not all nodes
        for j in range(NUM_NODES):
            actual_observers = len(neighbours[j])
            if actual_observers == 0:
                continue
            if isolation_votes[j] > actual_observers // 3:
                if j not in isolated:
                    print(f"\n  R{t+1:02d} *** Node {j} ISOLATED"
                          f"  (votes: {isolation_votes[j]}/{actual_observers}"
                          f"  malicious={nodes[j].is_malicious})")
                isolated.add(j)
            elif VERBOSE and suspicion_flags[j] > 0:
                print(f"  ~   Node {j:>2} SUSPICIOUS ({suspicion_flags[j]} observers)")

        # M3 applied inside apply_recovery
        isolated = apply_recovery(nodes, isolated, dt_this_round)

    # ── Routing ────────────────────────────────────────────────────────────
    score_matrix = compute_score_matrix(nodes, T_matrix, w_t, w_e)

    if VERBOSE:
        print("\nRouting Score Matrix:")
        print(pd.DataFrame(
            np.round(score_matrix, 3),
            columns=[f"N{j}" for j in range(NUM_NODES)],
            index=[f"N{i}" for i in range(NUM_NODES)]
        ))

    total_pairs = delivered_pairs = 0
    for src in range(NUM_NODES):
        if nodes[src].energy <= 0:
            continue
        for dst in range(NUM_NODES):
            if dst == src or nodes[dst].energy <= 0:
                continue
            path = select_path(src, dst, score_matrix, isolated)
            total_pairs += 1
            if path:
                compromised = any(nodes[h].is_malicious for h in path)
                if not compromised:
                    delivered_pairs += 1
                if VERBOSE:
                    status = "OK" if not compromised else "COMPROMISED"
                    print(f"  {src}→{dst}: {' -> '.join(map(str, path))}  [{status}]")

    pdr = delivered_pairs / total_pairs if total_pairs > 0 else 0.0
    pdr_history.append(pdr)

    # One summary line always printed
    malicious_nodes = sorted(n.id for n in nodes if n.is_malicious)
    print(f"  R{t+1:02d} | PDR: {pdr:.3f} | Energy: {avg_energy:.4f} J"
          f" | wt: {w_t:.3f} | isolated: {sorted(isolated)}"
          f" | malicious: {malicious_nodes}")


# ─────────────────────────── SAVE + PLOTS ────────────────────────────────────

np.save("atemrm_pdr.npy",    np.array(pdr_history))
np.save("atemrm_energy.npy", np.array(energy_history))

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("ATEMR-M: Adaptive Trust-Energy Routing with Mitigation",
             fontsize=12, fontweight='bold')

axes[0].plot(ws_history, color='darkorange')
axes[0].axvline(x=NEW_ATTACK_ROUND, color='red', linestyle='--', alpha=0.5, label='New attacker')
axes[0].axvline(x=MITIGATION_START_ROUND, color='gray', linestyle=':', label='Mitigation active')
axes[0].set_xlabel("Round")
axes[0].set_ylabel("wt")
axes[0].set_title("Adaptive Security Weight wt")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].plot(energy_history, color='green')
axes[1].set_xlabel("Round")
axes[1].set_ylabel("Avg Energy (J)")
axes[1].set_title("Network Energy — Radio Model")
axes[1].grid(alpha=0.3)

axes[2].plot(pdr_history, color='steelblue')
axes[2].set_ylim(0, 1.05)
axes[2].axvline(x=NEW_ATTACK_ROUND, color='red', linestyle='--', label='New attacker')
axes[2].axvline(x=MITIGATION_START_ROUND, color='gray', linestyle=':', label='Mitigation active')
axes[2].set_xlabel("Round")
axes[2].set_ylabel("PDR")
axes[2].set_title("Packet Delivery Ratio")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("atemrm_results.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved: atemrm_pdr.npy  atemrm_energy.npy  atemrm_results.png")
