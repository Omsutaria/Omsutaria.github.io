#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║       AI-AUGMENTED HYBRID NETWORK INTRUSION DETECTION SYSTEM        ║
║       Threat Scoring & Profiling Engine  v2.0                       ║
║       Platform : Kali Linux  |  Python 3.8+                         ║
║       Accuracy : ~99 % (ensemble: RF + IsolationForest + rules)     ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE
-----
  sudo python3 hybrid_nids.py --demo            # safe demo (no root needed)
  sudo python3 hybrid_nids.py --iface eth0      # live capture
  sudo python3 hybrid_nids.py --iface wlan0 --report  # live + HTML report

INSTALL (Kali)
--------------
  pip3 install scapy scikit-learn numpy pandas rich joblib
"""

import os, sys, time, json, socket, logging, argparse, threading, collections
import ipaddress, random, hashlib, html, csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ── ML ────────────────────────────────────────────────────────────────────────
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
    ML_OK = True
except ImportError:
    ML_OK = False
    print("[!] scikit-learn / joblib not found.  pip3 install scikit-learn joblib")

# ── Network ───────────────────────────────────────────────────────────────────
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP, DNS, Raw, get_if_list
    from scapy.all import wrpcap, rdpcap
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False

# ── UI ────────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich import box
    RICH_OK = True
    console = Console()
except ImportError:
    RICH_OK = False
    class _Console:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 60)
    console = _Console()


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    "interface"        : "eth0",
    "log_file"         : "nids_alerts.json",
    "report_file"      : "nids_report.html",
    "model_file"       : "/tmp/nids_rf.pkl",
    "scaler_file"      : "/tmp/nids_scaler.pkl",
    "if_file"          : "/tmp/nids_if.pkl",
    "threshold_ml"     : 0.55,   # probability above = suspicious
    "threshold_high"   : 0.75,
    "threshold_crit"   : 0.90,
    "window_size"      : 50,     # packets per flow window
    "profile_ttl"      : 300,    # seconds before profile expires
    "demo_packets"     : 300,    # synthetic packets in demo
}

ATTACK_LABELS = {
    0: "NORMAL",
    1: "PORT_SCAN",
    2: "SYN_FLOOD",
    3: "UDP_FLOOD",
    4: "ICMP_FLOOD",
    5: "BRUTE_FORCE",
    6: "SQL_INJECTION",
    7: "XSS_ATTACK",
    8: "DNS_AMPLIFICATION",
    9: "ARP_SPOOF",
    10: "OS_FINGERPRINT",
}

SEVERITY = {
    "NORMAL"        : ("INFO",     0),
    "PORT_SCAN"     : ("MEDIUM",  45),
    "SYN_FLOOD"     : ("HIGH",    70),
    "UDP_FLOOD"     : ("HIGH",    65),
    "ICMP_FLOOD"    : ("MEDIUM",  55),
    "BRUTE_FORCE"   : ("HIGH",    75),
    "SQL_INJECTION" : ("CRITICAL",90),
    "XSS_ATTACK"    : ("HIGH",    72),
    "DNS_AMPLIFICATION":("HIGH",  68),
    "ARP_SPOOF"     : ("CRITICAL",88),
    "OS_FINGERPRINT": ("MEDIUM",  40),
}

SEV_COLOR = {
    "CRITICAL": "bold red",
    "HIGH"    : "red",
    "MEDIUM"  : "yellow",
    "LOW"     : "green",
    "INFO"    : "cyan",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC TRAINING DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
class DataGenerator:
    """Generates labelled feature vectors for training.

    Feature vector (15 features):
      0  proto          (0=ICMP, 1=TCP, 2=UDP, 3=ARP, 4=OTHER)
      1  pkt_len
      2  src_port
      3  dst_port
      4  tcp_flags      (0-63 bitmap)
      5  ttl
      6  pkt_rate       (pkts/sec from this src in last window)
      7  byte_rate      (bytes/sec)
      8  unique_dsts    (unique dest IPs in window)
      9  unique_ports   (unique dst ports in window)
     10  syn_ratio      (SYN pkts / total)
     11  ack_ratio
     12  rst_ratio
     13  payload_len
     14  entropy        (payload byte-entropy 0-8)
    """
    NFEATURES = 15

    @staticmethod
    def _entropy(length: int, seed: int) -> float:
        rng = random.Random(seed)
        if length == 0:
            return 0.0
        counts = collections.Counter(rng.randint(0, 255) for _ in range(min(length, 64)))
        total = sum(counts.values())
        ent = -sum((c/total) * (c/total).bit_length() for c in counts.values() if c)
        return min(abs(ent) * 1.5, 8.0)

    @classmethod
    def _normal(cls, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            proto = random.choices([1, 2, 0], weights=[60, 30, 10])[0]
            if proto == 1:      # TCP
                sp = random.choice([1024 + random.randint(0, 60000)])
                dp = random.choice([80, 443, 22, 25, 53, 8080, 3306])
                flags = random.choice([2, 18, 16, 24])  # SYN, SYN-ACK, ACK, PSH-ACK
                plen = random.randint(40, 1500)
            elif proto == 2:    # UDP
                sp = random.randint(1024, 65535)
                dp = random.choice([53, 123, 161, 514])
                flags = 0
                plen = random.randint(28, 512)
            else:               # ICMP
                sp, dp, flags = 0, 0, 0
                plen = random.randint(28, 64)
            rows.append([
                proto, plen, sp, dp, flags,
                random.randint(60, 128),
                random.uniform(0.1, 5.0),
                random.uniform(100, 5000),
                random.randint(1, 5),
                random.randint(1, 3),
                random.uniform(0.0, 0.3),
                random.uniform(0.4, 0.9),
                random.uniform(0.0, 0.05),
                random.randint(0, 800),
                random.uniform(4.5, 7.5),
            ])
        return np.array(rows, dtype=np.float32)

    @classmethod
    def _port_scan(cls, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            rows.append([
                1, random.randint(40, 80), random.randint(1024, 65535),
                random.randint(1, 65535), 2,  # SYN only
                random.randint(50, 128),
                random.uniform(10, 200),   # high rate
                random.uniform(400, 8000),
                random.randint(1, 3),
                random.randint(20, 65534), # many unique ports
                random.uniform(0.85, 1.0), # almost all SYN
                random.uniform(0.0, 0.05),
                random.uniform(0.1, 0.4),
                0,
                random.uniform(0.0, 1.0),
            ])
        return np.array(rows, dtype=np.float32)

    @classmethod
    def _syn_flood(cls, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            rows.append([
                1, random.randint(40, 60), random.randint(1024, 65535),
                random.choice([80, 443]),
                2,  # SYN
                random.randint(40, 64),
                random.uniform(500, 5000),  # very high rate
                random.uniform(20000, 300000),
                random.randint(1, 2),
                random.randint(1, 2),
                random.uniform(0.95, 1.0),
                0.0,
                0.0,
                0,
                0.0,
            ])
        return np.array(rows, dtype=np.float32)

    @classmethod
    def _udp_flood(cls, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            rows.append([
                2, random.randint(512, 1500),
                random.randint(1024, 65535),
                random.randint(1, 65535),
                0,
                random.randint(40, 64),
                random.uniform(500, 3000),
                random.uniform(250000, 1500000),
                random.randint(1, 3),
                random.randint(1, 5),
                0.0, 0.0, 0.0,
                random.randint(512, 1460),
                random.uniform(7.0, 8.0),  # high entropy (random data)
            ])
        return np.array(rows, dtype=np.float32)

    @classmethod
    def _icmp_flood(cls, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            rows.append([
                0, random.randint(64, 1500), 0, 0, 0,
                random.randint(40, 64),
                random.uniform(300, 2000),
                random.uniform(20000, 100000),
                random.randint(1, 3),
                0,
                0.0, 0.0, 0.0,
                random.randint(56, 1450),
                random.uniform(6.5, 8.0),
            ])
        return np.array(rows, dtype=np.float32)

    @classmethod
    def _brute_force(cls, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            rows.append([
                1, random.randint(60, 200),
                random.randint(1024, 65535),
                random.choice([22, 21, 3389, 23, 5900]),
                random.choice([2, 18, 24]),
                random.randint(60, 128),
                random.uniform(5, 50),
                random.uniform(300, 10000),
                random.randint(1, 2),
                random.randint(1, 2),
                random.uniform(0.2, 0.5),
                random.uniform(0.3, 0.6),
                random.uniform(0.0, 0.1),
                random.randint(50, 200),
                random.uniform(3.0, 5.5),
            ])
        return np.array(rows, dtype=np.float32)

    @classmethod
    def _sql_injection(cls, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            rows.append([
                1, random.randint(200, 1500),
                random.randint(1024, 65535),
                random.choice([80, 443, 8080, 3306]),
                24,  # PSH+ACK
                random.randint(60, 128),
                random.uniform(0.5, 10),
                random.uniform(200, 15000),
                random.randint(1, 2),
                random.randint(1, 2),
                random.uniform(0.0, 0.2),
                random.uniform(0.5, 0.9),
                0.0,
                random.randint(200, 1400),
                random.uniform(5.5, 7.8),
            ])
        return np.array(rows, dtype=np.float32)

    @classmethod
    def _xss(cls, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            rows.append([
                1, random.randint(150, 800),
                random.randint(1024, 65535),
                random.choice([80, 443, 8080]),
                24,
                random.randint(60, 128),
                random.uniform(0.5, 8),
                random.uniform(200, 8000),
                random.randint(1, 2),
                random.randint(1, 2),
                random.uniform(0.0, 0.15),
                random.uniform(0.6, 0.9),
                0.0,
                random.randint(150, 700),
                random.uniform(5.0, 7.2),
            ])
        return np.array(rows, dtype=np.float32)

    @classmethod
    def _dns_amp(cls, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            rows.append([
                2, random.randint(500, 1500), random.randint(1024, 65535),
                53, 0,
                random.randint(50, 128),
                random.uniform(50, 500),
                random.uniform(25000, 750000),
                random.randint(1, 4),
                random.randint(1, 3),
                0.0, 0.0, 0.0,
                random.randint(500, 1460),
                random.uniform(6.0, 8.0),
            ])
        return np.array(rows, dtype=np.float32)

    @classmethod
    def _arp_spoof(cls, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            rows.append([
                3, random.randint(28, 42), 0, 0, 0,  # proto=ARP
                0,
                random.uniform(5, 200),
                random.uniform(140, 8400),
                random.randint(2, 50),
                0,
                0.0, 0.0, 0.0,
                0,
                0.0,
            ])
        return np.array(rows, dtype=np.float32)

    @classmethod
    def _os_fp(cls, n: int) -> np.ndarray:
        rows = []
        for _ in range(n):
            rows.append([
                1, random.choice([40, 41, 42, 44, 60]),
                random.randint(1024, 65535),
                random.randint(1, 1024),
                random.choice([2, 1, 41]),  # unusual flags
                random.choice([255, 64, 128, 1]),  # probing TTL values
                random.uniform(0.1, 5),
                random.uniform(40, 500),
                random.randint(1, 5),
                random.randint(5, 30),
                random.uniform(0.7, 1.0),
                0.0,
                random.uniform(0.2, 0.5),
                0,
                0.0,
            ])
        return np.array(rows, dtype=np.float32)

    @classmethod
    def build_dataset(cls, samples_per_class: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """Build labelled training set."""
        generators = [
            (0,  cls._normal),
            (1,  cls._port_scan),
            (2,  cls._syn_flood),
            (3,  cls._udp_flood),
            (4,  cls._icmp_flood),
            (5,  cls._brute_force),
            (6,  cls._sql_injection),
            (7,  cls._xss),
            (8,  cls._dns_amp),
            (9,  cls._arp_spoof),
            (10, cls._os_fp),
        ]
        X_parts, y_parts = [], []
        for label, fn in generators:
            X_parts.append(fn(samples_per_class))
            y_parts.append(np.full(samples_per_class, label, dtype=np.int32))
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  ML MODEL MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class ModelManager:
    def __init__(self):
        self.rf: Optional[RandomForestClassifier] = None
        self.iso: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.accuracy: float = 0.0
        self.trained: bool = False

    def train(self) -> float:
        if not ML_OK:
            return 0.0
        console.print("[bold cyan]  Training ML models on synthetic dataset …[/bold cyan]" if RICH_OK
                      else "  Training ML models …")
        random.seed(42)
        np.random.seed(42)
        X, y = DataGenerator.build_dataset(samples_per_class=2000)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.scaler = StandardScaler()
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=3,
            n_jobs=-1, random_state=42, class_weight="balanced"
        )
        self.rf.fit(X_tr_s, y_tr)
        y_pred = self.rf.predict(X_te_s)
        self.accuracy = accuracy_score(y_te, y_pred)

        # Isolation Forest trained only on normal traffic
        normal_mask = y_tr == 0
        self.iso = IsolationForest(n_estimators=150, contamination=0.05, random_state=42, n_jobs=-1)
        self.iso.fit(X_tr_s[normal_mask])

        joblib.dump(self.rf,     CFG["model_file"])
        joblib.dump(self.scaler, CFG["scaler_file"])
        joblib.dump(self.iso,    CFG["if_file"])

        self.trained = True
        return self.accuracy

    def load(self) -> bool:
        if not ML_OK:
            return False
        try:
            self.rf     = joblib.load(CFG["model_file"])
            self.scaler = joblib.load(CFG["scaler_file"])
            self.iso    = joblib.load(CFG["if_file"])
            self.trained = True
            return True
        except Exception:
            return False

    def predict(self, features: np.ndarray) -> Tuple[int, float, bool]:
        """Returns (label_index, confidence, is_anomaly)."""
        if not self.trained:
            return 0, 0.0, False
        fv = self.scaler.transform(features.reshape(1, -1))
        proba = self.rf.predict_proba(fv)[0]
        label = int(np.argmax(proba))
        conf  = float(proba[label])
        iso_score = self.iso.decision_function(fv)[0]
        is_anomaly = bool(iso_score < 0)
        return label, conf, is_anomaly


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNATURE ENGINE  (rule-based)
# ══════════════════════════════════════════════════════════════════════════════
class SignatureEngine:
    """Fast rule-based first-pass detector."""

    SQLI_PATTERNS  = [b"union select", b"or 1=1", b"drop table", b"xp_cmdshell",
                      b"information_schema", b"' or '", b"/**/", b"benchmark("]
    XSS_PATTERNS   = [b"<script", b"javascript:", b"onerror=", b"alert(",
                      b"document.cookie", b"eval(", b"onload="]

    def check(self, pkt_info: dict) -> Tuple[Optional[str], int]:
        """
        Returns (attack_type_or_None, base_score).
        pkt_info keys: proto, src_ip, dst_ip, src_port, dst_port,
                       flags, payload, pkt_len, ttl
        """
        proto    = pkt_info.get("proto", "")
        dport    = pkt_info.get("dst_port", 0)
        flags    = pkt_info.get("flags", 0)
        payload  = pkt_info.get("payload", b"").lower()
        ttl      = pkt_info.get("ttl", 64)
        pkt_len  = pkt_info.get("pkt_len", 0)

        # ── SYN-only with no ACK ──────────────────────────────────────────
        if proto == "TCP" and (flags & 0x02) and not (flags & 0x10):
            if dport in (80, 443, 22, 8080, 25):
                return "SYN_FLOOD", 65

        # ── SQL Injection ─────────────────────────────────────────────────
        for pat in self.SQLI_PATTERNS:
            if pat in payload:
                return "SQL_INJECTION", 88

        # ── XSS ──────────────────────────────────────────────────────────
        for pat in self.XSS_PATTERNS:
            if pat in payload:
                return "XSS_ATTACK", 72

        # ── ICMP oversized (possible flood / smurf) ───────────────────────
        if proto == "ICMP" and pkt_len > 500:
            return "ICMP_FLOOD", 55

        # ── UDP small-port to DNS (possible amplification) ────────────────
        if proto == "UDP" and dport == 53 and pkt_len > 400:
            return "DNS_AMPLIFICATION", 65

        # ── Unusual TTL (OS fingerprinting probe) ─────────────────────────
        if ttl in (0, 1, 255) and proto == "TCP":
            return "OS_FINGERPRINT", 38

        # ── Brute-force ports ─────────────────────────────────────────────
        if proto == "TCP" and dport in (22, 21, 3389, 23, 5900) and pkt_len < 100:
            return "BRUTE_FORCE", 45

        return None, 0


# ══════════════════════════════════════════════════════════════════════════════
#  FLOW TRACKER  (per-source statistics)
# ══════════════════════════════════════════════════════════════════════════════
class FlowTracker:
    def __init__(self):
        self._lock = threading.Lock()
        # src_ip -> deque of (timestamp, pkt_len, dst_ip, dst_port, flags)
        self._flows: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=CFG["window_size"])
        )

    def update(self, src_ip: str, pkt_len: int, dst_ip: str, dst_port: int, flags: int) -> dict:
        now = time.time()
        with self._lock:
            q = self._flows[src_ip]
            q.append((now, pkt_len, dst_ip, dst_port, flags))
            if len(q) < 2:
                return self._empty_stats()
            oldest = q[0][0]
            window  = max(now - oldest, 0.001)
            total   = len(q)
            bytes_  = sum(e[1] for e in q)
            u_dsts  = len({e[2] for e in q})
            u_ports = len({e[3] for e in q})
            syns    = sum(1 for e in q if e[4] & 0x02)
            acks    = sum(1 for e in q if e[4] & 0x10)
            rsts    = sum(1 for e in q if e[4] & 0x04)
            return {
                "pkt_rate"    : total  / window,
                "byte_rate"   : bytes_ / window,
                "unique_dsts" : u_dsts,
                "unique_ports": u_ports,
                "syn_ratio"   : syns  / total,
                "ack_ratio"   : acks  / total,
                "rst_ratio"   : rsts  / total,
            }

    @staticmethod
    def _empty_stats() -> dict:
        return {"pkt_rate": 0, "byte_rate": 0, "unique_dsts": 1,
                "unique_ports": 1, "syn_ratio": 0, "ack_ratio": 0, "rst_ratio": 0}


# ══════════════════════════════════════════════════════════════════════════════
#  THREAT PROFILER
# ══════════════════════════════════════════════════════════════════════════════
class ThreatProfile:
    def __init__(self, ip: str):
        self.ip           = ip
        self.first_seen   = datetime.now()
        self.last_seen    = datetime.now()
        self.total_pkts   = 0
        self.alert_count  = 0
        self.attack_types : Dict[str, int] = collections.defaultdict(int)
        self.max_score    = 0
        self.avg_score    = 0.0
        self._score_sum   = 0.0
        self.risk_level   = "LOW"
        try:
            self.hostname = socket.gethostbyaddr(ip)[0]
        except Exception:
            self.hostname = "unknown"
        try:
            obj = ipaddress.ip_address(ip)
            self.is_private = obj.is_private
        except Exception:
            self.is_private = False

    def update(self, attack_type: str, score: int):
        self.last_seen   = datetime.now()
        self.total_pkts += 1
        if attack_type != "NORMAL":
            self.alert_count += 1
            self.attack_types[attack_type] += 1
        self.max_score    = max(self.max_score, score)
        self._score_sum  += score
        self.avg_score    = self._score_sum / self.total_pkts
        if self.max_score >= 85:
            self.risk_level = "CRITICAL"
        elif self.max_score >= 65:
            self.risk_level = "HIGH"
        elif self.max_score >= 40:
            self.risk_level = "MEDIUM"
        else:
            self.risk_level = "LOW"

    def top_attack(self) -> str:
        if not self.attack_types:
            return "NORMAL"
        return max(self.attack_types, key=self.attack_types.get)

    def to_dict(self) -> dict:
        return {
            "ip"          : self.ip,
            "hostname"    : self.hostname,
            "is_private"  : self.is_private,
            "first_seen"  : self.first_seen.isoformat(),
            "last_seen"   : self.last_seen.isoformat(),
            "total_pkts"  : self.total_pkts,
            "alert_count" : self.alert_count,
            "attack_types": dict(self.attack_types),
            "max_score"   : self.max_score,
            "avg_score"   : round(self.avg_score, 2),
            "risk_level"  : self.risk_level,
        }


class Profiler:
    def __init__(self):
        self._lock    = threading.Lock()
        self._profiles: Dict[str, ThreatProfile] = {}

    def update(self, ip: str, attack_type: str, score: int) -> ThreatProfile:
        with self._lock:
            if ip not in self._profiles:
                self._profiles[ip] = ThreatProfile(ip)
            p = self._profiles[ip]
            p.update(attack_type, score)
            return p

    def get_top(self, n: int = 10) -> List[ThreatProfile]:
        with self._lock:
            return sorted(self._profiles.values(), key=lambda p: p.max_score, reverse=True)[:n]

    def all(self) -> List[ThreatProfile]:
        with self._lock:
            return list(self._profiles.values())

    def count(self) -> int:
        return len(self._profiles)


# ══════════════════════════════════════════════════════════════════════════════
#  THREAT SCORER
# ══════════════════════════════════════════════════════════════════════════════
class ThreatScorer:
    """Fuses ML confidence, signature score, and flow stats into 0-100 score."""

    @staticmethod
    def score(attack_type: str, ml_conf: float, sig_score: int,
              is_anomaly: bool, flow_stats: dict) -> int:
        _, base = SEVERITY.get(attack_type, ("LOW", 20))

        # ML confidence bonus
        ml_bonus = ml_conf * 30 if attack_type != "NORMAL" else 0

        # Anomaly bonus
        anom_bonus = 10 if is_anomaly and attack_type != "NORMAL" else 0

        # Flow-rate penalty
        rate    = flow_stats.get("pkt_rate", 0)
        rate_b  = min(rate / 100, 1.0) * 15

        # Port diversity penalty
        u_ports = flow_stats.get("unique_ports", 1)
        port_b  = min(u_ports / 50, 1.0) * 10

        # SYN ratio penalty
        syn_b   = flow_stats.get("syn_ratio", 0) * 10

        raw = base + ml_bonus + anom_bonus + rate_b + port_b + syn_b
        return min(int(raw), 100)

    @staticmethod
    def severity_label(score: int) -> str:
        if score >= 85: return "CRITICAL"
        if score >= 65: return "HIGH"
        if score >= 40: return "MEDIUM"
        if score >= 20: return "LOW"
        return "INFO"


# ══════════════════════════════════════════════════════════════════════════════
#  ALERT MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class AlertManager:
    def __init__(self, log_file: str):
        self.log_file  = log_file
        self._lock     = threading.Lock()
        self.alerts    : List[dict] = []
        self.total     = 0
        self.by_type   : Dict[str, int] = collections.defaultdict(int)
        self.by_sev    : Dict[str, int] = collections.defaultdict(int)

        logging.basicConfig(
            filename="nids_system.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )

    def record(self, alert: dict):
        with self._lock:
            self.alerts.append(alert)
            self.total += 1
            self.by_type[alert["attack_type"]] += 1
            self.by_sev[alert["severity"]] += 1
            try:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(alert) + "\n")
            except Exception:
                pass
            logging.info(json.dumps(alert))

    def recent(self, n: int = 20) -> List[dict]:
        with self._lock:
            return list(reversed(self.alerts[-n:]))


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
def extract_features(pkt_info: dict, flow_stats: dict) -> np.ndarray:
    PROTO_MAP = {"ICMP": 0, "TCP": 1, "UDP": 2, "ARP": 3}
    proto_id  = PROTO_MAP.get(pkt_info.get("proto", ""), 4)
    payload   = pkt_info.get("payload", b"")
    plen      = len(payload) if payload else 0

    # Shannon entropy
    if plen > 0:
        counts  = collections.Counter(payload)
        total   = plen
        entropy = -sum((c/total) * np.log2(c/total) for c in counts.values() if c > 0)
    else:
        entropy = 0.0

    return np.array([
        proto_id,
        pkt_info.get("pkt_len", 0),
        pkt_info.get("src_port", 0),
        pkt_info.get("dst_port", 0),
        pkt_info.get("flags", 0),
        pkt_info.get("ttl", 64),
        flow_stats.get("pkt_rate", 0),
        flow_stats.get("byte_rate", 0),
        flow_stats.get("unique_dsts", 1),
        flow_stats.get("unique_ports", 1),
        flow_stats.get("syn_ratio", 0),
        flow_stats.get("ack_ratio", 0),
        flow_stats.get("rst_ratio", 0),
        plen,
        entropy,
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  PACKET PROCESSOR  (called per packet)
# ══════════════════════════════════════════════════════════════════════════════
class PacketProcessor:
    def __init__(self, model: ModelManager, sig: SignatureEngine,
                 flow: FlowTracker, scorer: ThreatScorer,
                 profiler: Profiler, alerts: AlertManager):
        self.model    = model
        self.sig      = sig
        self.flow     = flow
        self.scorer   = scorer
        self.profiler = profiler
        self.alerts   = alerts
        self.pkt_count = 0

    def process(self, pkt_info: dict):
        self.pkt_count += 1
        src_ip   = pkt_info.get("src_ip", "0.0.0.0")
        dst_ip   = pkt_info.get("dst_ip", "0.0.0.0")
        dst_port = pkt_info.get("dst_port", 0)
        pkt_len  = pkt_info.get("pkt_len", 0)
        flags    = pkt_info.get("flags", 0)

        # 1. Flow stats update
        flow_stats = self.flow.update(src_ip, pkt_len, dst_ip, dst_port, flags)

        # 2. Signature check
        sig_attack, sig_score = self.sig.check(pkt_info)

        # 3. ML prediction
        features   = extract_features(pkt_info, flow_stats)
        ml_label, ml_conf, is_anomaly = self.model.predict(features)
        ml_attack  = ATTACK_LABELS[ml_label]

        # 4. Fuse: prefer signature if high confidence, else use ML
        if sig_attack and sig_score >= 60:
            attack_type = sig_attack
            confidence  = sig_score / 100.0
        elif ml_attack != "NORMAL" and ml_conf >= CFG["threshold_ml"]:
            attack_type = ml_attack
            confidence  = ml_conf
        elif is_anomaly and ml_attack != "NORMAL":
            attack_type = ml_attack
            confidence  = ml_conf
        else:
            attack_type = "NORMAL"
            confidence  = ml_conf

        # 5. Score
        score    = self.scorer.score(attack_type, confidence, sig_score, is_anomaly, flow_stats)
        severity = self.scorer.severity_label(score)

        # 6. Profile update
        profile  = self.profiler.update(src_ip, attack_type, score)

        # 7. Alert if not normal
        if attack_type != "NORMAL":
            alert = {
                "timestamp"   : datetime.now().isoformat(),
                "src_ip"      : src_ip,
                "dst_ip"      : dst_ip,
                "dst_port"    : dst_port,
                "proto"       : pkt_info.get("proto", "?"),
                "attack_type" : attack_type,
                "severity"    : severity,
                "score"       : score,
                "ml_conf"     : round(float(confidence) * 100, 1),
                "sig_match"   : sig_attack is not None,
                "is_anomaly"  : is_anomaly,
                "risk_level"  : profile.risk_level,
            }
            self.alerts.record(alert)
            return alert
        return None

    def process_scapy_pkt(self, pkt):
        """Adapter: scapy packet → pkt_info dict."""
        info = {"payload": b"", "flags": 0, "ttl": 64,
                "src_port": 0, "dst_port": 0}

        if pkt.haslayer("ARP"):
            info.update({"proto": "ARP", "src_ip": pkt["ARP"].psrc,
                         "dst_ip": pkt["ARP"].pdst, "pkt_len": len(pkt)})
        elif pkt.haslayer("IP"):
            ip = pkt["IP"]
            info.update({"src_ip": ip.src, "dst_ip": ip.dst,
                         "ttl": ip.ttl, "pkt_len": len(pkt)})
            if pkt.haslayer("TCP"):
                tcp = pkt["TCP"]
                info.update({"proto": "TCP", "src_port": tcp.sport,
                             "dst_port": tcp.dport, "flags": int(tcp.flags)})
                if pkt.haslayer("Raw"):
                    info["payload"] = bytes(pkt["Raw"])[:512]
            elif pkt.haslayer("UDP"):
                udp = pkt["UDP"]
                info.update({"proto": "UDP", "src_port": udp.sport,
                             "dst_port": udp.dport})
                if pkt.haslayer("Raw"):
                    info["payload"] = bytes(pkt["Raw"])[:512]
            elif pkt.haslayer("ICMP"):
                info["proto"] = "ICMP"
        else:
            return  # skip non-IP/ARP frames

        self.process(info)


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO PACKET SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
class DemoSimulator:
    """Generates synthetic pkt_info dicts without touching the network."""

    SCENARIOS = [
        {"name": "Normal HTTP traffic",      "label": 0,  "count": 50},
        {"name": "Port Scan (nmap)",          "label": 1,  "count": 30},
        {"name": "SYN Flood attack",          "label": 2,  "count": 25},
        {"name": "UDP Flood",                 "label": 3,  "count": 20},
        {"name": "ICMP Flood (ping flood)",   "label": 4,  "count": 20},
        {"name": "SSH Brute Force",           "label": 5,  "count": 20},
        {"name": "SQL Injection attempt",     "label": 6,  "count": 15},
        {"name": "XSS attack",                "label": 7,  "count": 15},
        {"name": "DNS Amplification",         "label": 8,  "count": 15},
        {"name": "ARP Spoofing",              "label": 9,  "count": 15},
        {"name": "OS Fingerprinting (nmap)",  "label": 10, "count": 15},
    ]

    @staticmethod
    def _rand_ip(private: bool = True) -> str:
        if private:
            return f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"
        return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

    @classmethod
    def generate(cls, label: int) -> dict:
        src = cls._rand_ip(private=(label in (5, 9)))
        dst = cls._rand_ip(private=True)
        base = {"src_ip": src, "dst_ip": dst, "payload": b"", "flags": 0, "ttl": 64}

        if label == 0:  # Normal
            base.update({"proto": "TCP", "src_port": random.randint(1024, 65535),
                         "dst_port": random.choice([80, 443]), "pkt_len": random.randint(200, 1200),
                         "flags": 24, "payload": b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"})
        elif label == 1:  # Port Scan
            base.update({"proto": "TCP", "src_port": random.randint(1024, 65535),
                         "dst_port": random.randint(1, 65535), "pkt_len": 40, "flags": 2})
        elif label == 2:  # SYN Flood
            base.update({"proto": "TCP", "src_port": random.randint(1024, 65535),
                         "dst_port": 80, "pkt_len": 40, "flags": 2, "ttl": 48})
        elif label == 3:  # UDP Flood
            base.update({"proto": "UDP", "src_port": random.randint(1024, 65535),
                         "dst_port": random.randint(1, 65535),
                         "pkt_len": random.randint(800, 1500),
                         "payload": bytes(random.randint(0, 255) for _ in range(900))})
        elif label == 4:  # ICMP Flood
            base.update({"proto": "ICMP", "src_port": 0, "dst_port": 0,
                         "pkt_len": random.randint(600, 1200),
                         "payload": bytes(1100)})
        elif label == 5:  # Brute Force
            base.update({"proto": "TCP", "src_port": random.randint(1024, 65535),
                         "dst_port": 22, "pkt_len": 80, "flags": 24,
                         "payload": b"SSH-2.0-OpenSSH"})
        elif label == 6:  # SQLi
            base.update({"proto": "TCP", "src_port": random.randint(1024, 65535),
                         "dst_port": 80, "pkt_len": 450, "flags": 24,
                         "payload": b"GET /?id=1 union select username,password from users--"})
        elif label == 7:  # XSS
            base.update({"proto": "TCP", "src_port": random.randint(1024, 65535),
                         "dst_port": 80, "pkt_len": 300, "flags": 24,
                         "payload": b"GET /?q=<script>alert(document.cookie)</script>"})
        elif label == 8:  # DNS Amp
            base.update({"proto": "UDP", "src_port": random.randint(1024, 65535),
                         "dst_port": 53, "pkt_len": random.randint(600, 1400),
                         "payload": bytes(random.randint(0, 255) for _ in range(1000))})
        elif label == 9:  # ARP Spoof
            base.update({"proto": "ARP", "src_port": 0, "dst_port": 0,
                         "pkt_len": 42, "flags": 0})
        elif label == 10:  # OS FP
            base.update({"proto": "TCP", "src_port": random.randint(1024, 65535),
                         "dst_port": random.randint(1, 1024), "pkt_len": 40,
                         "flags": 2, "ttl": 255})
        return base


# ══════════════════════════════════════════════════════════════════════════════
#  RICH DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
class Dashboard:
    def __init__(self, alerts: AlertManager, profiler: Profiler,
                 model: ModelManager, processor: PacketProcessor):
        self.alerts    = alerts
        self.profiler  = profiler
        self.model     = model
        self.proc      = processor
        self._start    = time.time()

    def _header(self) -> Panel:
        elapsed = int(time.time() - self._start)
        h, m, s = elapsed//3600, (elapsed%3600)//60, elapsed%60
        acc_str = f"{self.model.accuracy*100:.2f}%" if self.model.accuracy else "N/A"
        txt = (
            f"[bold cyan]AI HYBRID NIDS[/bold cyan]  |  "
            f"Packets: [yellow]{self.proc.pkt_count}[/yellow]  |  "
            f"Alerts: [red]{self.alerts.total}[/red]  |  "
            f"Profiles: [green]{self.profiler.count()}[/green]  |  "
            f"Accuracy: [green]{acc_str}[/green]  |  "
            f"Uptime: [dim]{h:02d}:{m:02d}:{s:02d}[/dim]"
        )
        return Panel(Align.center(txt), style="bold blue", box=box.DOUBLE_EDGE)

    def _alert_table(self) -> Table:
        t = Table(title="[bold red]Recent Alerts[/bold red]",
                  box=box.SIMPLE_HEAVY, expand=True, show_lines=False)
        t.add_column("Time",        style="dim",          width=10)
        t.add_column("Src IP",      style="cyan",         width=16)
        t.add_column("Dst IP",      style="white",        width=16)
        t.add_column("Port",        style="yellow",       width=6)
        t.add_column("Attack",      style="bold",         width=18)
        t.add_column("Score",       justify="right",      width=7)
        t.add_column("Severity",    style="bold",         width=10)
        t.add_column("Conf %",      justify="right",      width=8)

        for a in self.alerts.recent(15):
            sev   = a["severity"]
            color = SEV_COLOR.get(sev, "white")
            t.add_row(
                a["timestamp"][11:19],
                a["src_ip"],
                a["dst_ip"],
                str(a["dst_port"]),
                f"[{color}]{a['attack_type']}[/{color}]",
                f"[{color}]{a['score']}[/{color}]",
                f"[{color}]{sev}[/{color}]",
                f"{a['ml_conf']}%",
            )
        return t

    def _profile_table(self) -> Table:
        t = Table(title="[bold yellow]Top Threat Profiles[/bold yellow]",
                  box=box.SIMPLE_HEAVY, expand=True, show_lines=False)
        t.add_column("IP",         style="cyan",  width=16)
        t.add_column("Hostname",   style="dim",   width=18)
        t.add_column("Pkts",       justify="right", width=6)
        t.add_column("Alerts",     justify="right", width=7)
        t.add_column("Max Score",  justify="right", width=10)
        t.add_column("Top Attack", style="bold",  width=18)
        t.add_column("Risk",       style="bold",  width=10)

        for p in self.profiler.get_top(10):
            color = SEV_COLOR.get(p.risk_level, "white")
            t.add_row(
                p.ip, p.hostname[:17],
                str(p.total_pkts), str(p.alert_count),
                str(p.max_score),
                p.top_attack(),
                f"[{color}]{p.risk_level}[/{color}]",
            )
        return t

    def _stats_panel(self) -> Panel:
        lines = []
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
            cnt   = self.alerts.by_sev.get(sev, 0)
            color = SEV_COLOR.get(sev, "white")
            bar   = "█" * min(cnt, 30)
            lines.append(f"[{color}]{sev:9s}[/{color}] {bar} {cnt}")
        return Panel("\n".join(lines), title="[bold]Severity Distribution[/bold]",
                     box=box.ROUNDED, expand=True)

    def _type_panel(self) -> Panel:
        lines = []
        for atype, cnt in sorted(self.alerts.by_type.items(),
                                 key=lambda x: x[1], reverse=True)[:8]:
            sev, _  = SEVERITY.get(atype, ("INFO", 0))
            color   = SEV_COLOR.get(sev, "white")
            bar     = "█" * min(cnt, 20)
            lines.append(f"[{color}]{atype:18s}[/{color}] {bar} {cnt}")
        if not lines:
            lines = ["[dim]No attacks detected yet[/dim]"]
        return Panel("\n".join(lines), title="[bold]Attack Types[/bold]",
                     box=box.ROUNDED, expand=True)

    def render(self) -> str:
        if not RICH_OK:
            return ""
        from rich.console import Console as C
        from io import StringIO
        buf = StringIO()
        c2  = C(file=buf, width=130)
        c2.print(self._header())
        c2.print(Columns([self._stats_panel(), self._type_panel()]))
        c2.print(self._alert_table())
        c2.print(self._profile_table())
        return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  HTML REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_html_report(alerts: AlertManager, profiler: Profiler,
                         model: ModelManager, proc: PacketProcessor) -> str:
    acc_str = f"{model.accuracy*100:.2f}%" if model.accuracy else "N/A"
    ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_alerts = ""
    for a in reversed(alerts.alerts[-200:]):
        sev = a["severity"]
        color_map = {"CRITICAL": "#ff4444", "HIGH": "#ff8800",
                     "MEDIUM": "#ffcc00", "LOW": "#44cc44", "INFO": "#4488ff"}
        c = color_map.get(sev, "#999")
        rows_alerts += (
            f"<tr style='border-left:4px solid {c}'>"
            f"<td>{html.escape(a['timestamp'][11:19])}</td>"
            f"<td>{html.escape(a['src_ip'])}</td>"
            f"<td>{html.escape(a['dst_ip'])}</td>"
            f"<td>{a['dst_port']}</td>"
            f"<td style='color:{c};font-weight:bold'>{html.escape(a['attack_type'])}</td>"
            f"<td style='color:{c};font-weight:bold'>{a['score']}</td>"
            f"<td style='color:{c}'>{sev}</td>"
            f"<td>{a['ml_conf']}%</td>"
            f"</tr>\n"
        )

    rows_profiles = ""
    for p in profiler.get_top(20):
        color_map2 = {"CRITICAL": "#ff4444", "HIGH": "#ff8800",
                      "MEDIUM": "#ffcc00", "LOW": "#44cc44"}
        c2 = color_map2.get(p.risk_level, "#999")
        rows_profiles += (
            f"<tr><td>{html.escape(p.ip)}</td>"
            f"<td>{html.escape(p.hostname)}</td>"
            f"<td>{p.total_pkts}</td><td>{p.alert_count}</td>"
            f"<td>{p.max_score}</td>"
            f"<td>{html.escape(p.top_attack())}</td>"
            f"<td style='color:{c2};font-weight:bold'>{p.risk_level}</td></tr>\n"
        )

    type_stats = "".join(
        f"<li><b>{html.escape(k)}</b>: {v}</li>"
        for k, v in sorted(alerts.by_type.items(), key=lambda x: x[1], reverse=True)
    )

    template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Hybrid NIDS Report</title>
<style>
  body{{background:#0a0e1a;color:#ccd6f6;font-family:'Courier New',monospace;margin:0;padding:20px}}
  h1{{color:#64ffda;text-align:center;border-bottom:2px solid #233554;padding-bottom:10px}}
  h2{{color:#ccd6f6;margin-top:30px;border-left:4px solid #64ffda;padding-left:10px}}
  .badge{{display:inline-block;padding:4px 10px;border-radius:4px;margin:4px;font-size:0.85em}}
  .crit{{background:#ff4444;color:#fff}} .high{{background:#ff8800;color:#fff}}
  .med{{background:#ffcc00;color:#000}} .info{{background:#4488ff;color:#fff}}
  table{{width:100%;border-collapse:collapse;margin-top:10px;font-size:0.85em}}
  th{{background:#112240;color:#64ffda;padding:8px;text-align:left}}
  td{{padding:6px 8px;border-bottom:1px solid #1e3a5f}}
  tr:hover{{background:#112240}}
  .stat-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:20px 0}}
  .stat-box{{background:#112240;border:1px solid #233554;border-radius:8px;padding:15px;text-align:center}}
  .stat-num{{font-size:2em;color:#64ffda;font-weight:bold}}
  .stat-lbl{{color:#8892b0;font-size:0.85em}}
  ul{{list-style:none;padding:0}} ul li{{padding:3px 0;border-bottom:1px solid #1e3a5f}}
  footer{{text-align:center;margin-top:40px;color:#8892b0;font-size:0.8em}}
</style>
</head>
<body>
<h1>🛡️ AI-Augmented Hybrid NIDS — Threat Report</h1>
<p style="text-align:center;color:#8892b0">Generated: {ts} | ML Accuracy: <b style="color:#64ffda">{acc_str}</b></p>

<div class="stat-grid">
  <div class="stat-box"><div class="stat-num">{proc.pkt_count}</div><div class="stat-lbl">Packets Analysed</div></div>
  <div class="stat-box"><div class="stat-num" style="color:#ff4444">{alerts.total}</div><div class="stat-lbl">Total Alerts</div></div>
  <div class="stat-box"><div class="stat-num" style="color:#ffcc00">{profiler.count()}</div><div class="stat-lbl">IP Profiles</div></div>
  <div class="stat-box"><div class="stat-num">{len(alerts.by_type)}</div><div class="stat-lbl">Attack Types Seen</div></div>
</div>

<h2>Severity Breakdown</h2>
{"".join(f'<span class="badge {'crit' if s=='CRITICAL' else 'high' if s=='HIGH' else 'med' if s=='MEDIUM' else 'info'}">{s}: {alerts.by_sev.get(s,0)}</span>' for s in ("CRITICAL","HIGH","MEDIUM","LOW","INFO"))}

<h2>Attack Type Distribution</h2><ul>{type_stats}</ul>

<h2>Recent Alerts (last 200)</h2>
<table>
<thead><tr><th>Time</th><th>Src IP</th><th>Dst IP</th><th>Port</th>
<th>Attack</th><th>Score</th><th>Severity</th><th>Confidence</th></tr></thead>
<tbody>{rows_alerts}</tbody>
</table>

<h2>Top Threat Profiles</h2>
<table>
<thead><tr><th>IP</th><th>Hostname</th><th>Packets</th><th>Alerts</th>
<th>Max Score</th><th>Top Attack</th><th>Risk Level</th></tr></thead>
<tbody>{rows_profiles}</tbody>
</table>

<footer>AI Hybrid NIDS v2.0 | Ensemble: RandomForest + IsolationForest + Signature Rules</footer>
</body></html>"""

    path = CFG["report_file"]
    with open(path, "w") as f:
        f.write(template)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  NIDS ENGINE  (orchestrator)
# ══════════════════════════════════════════════════════════════════════════════
class NIDSEngine:
    def __init__(self, interface: str = "eth0", demo: bool = False,
                 report: bool = False, no_live: bool = False):
        self.interface = interface
        self.demo      = demo
        self.report    = report
        self.no_live   = no_live
        self._stop     = threading.Event()

        self.model     = ModelManager()
        self.sig       = SignatureEngine()
        self.flow      = FlowTracker()
        self.profiler  = Profiler()
        self.alerts    = AlertManager(CFG["log_file"])
        self.scorer    = ThreatScorer()
        self.proc      = PacketProcessor(self.model, self.sig, self.flow,
                                         self.scorer, self.profiler, self.alerts)
        self.dashboard = Dashboard(self.alerts, self.profiler, self.model, self.proc)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    def _init_model(self):
        loaded = self.model.load()
        if not loaded:
            acc = self.model.train()
            if RICH_OK:
                console.print(f"  [green]✓ Model trained — accuracy: {acc*100:.2f}%[/green]")
            else:
                print(f"  Model trained — accuracy: {acc*100:.2f}%")
        else:
            if RICH_OK:
                console.print("  [green]✓ Model loaded from cache[/green]")

    # ── Demo mode ──────────────────────────────────────────────────────────────
    def _run_demo(self):
        if RICH_OK:
            console.rule("[bold cyan]DEMO MODE — Simulating attacks[/bold cyan]")
        else:
            print("=" * 60)
            print("DEMO MODE")

        total_pkts = sum(s["count"] for s in DemoSimulator.SCENARIOS)

        if RICH_OK and not self.no_live:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                          console=console) as prog:
                task = prog.add_task("[cyan]Processing packets...", total=total_pkts)
                for scenario in DemoSimulator.SCENARIOS:
                    prog.update(task, description=f"[cyan]{scenario['name']}")
                    for _ in range(scenario["count"]):
                        pkt = DemoSimulator.generate(scenario["label"])
                        # Simulate flow by repeating src_ip in quick succession
                        for burst in range(random.randint(1, 5)):
                            self.proc.process(pkt)
                        prog.advance(task, 1)
                        time.sleep(0.01)
        else:
            for scenario in DemoSimulator.SCENARIOS:
                print(f"  → {scenario['name']}")
                for _ in range(scenario["count"]):
                    pkt = DemoSimulator.generate(scenario["label"])
                    for _ in range(random.randint(1, 5)):
                        self.proc.process(pkt)
                    time.sleep(0.005)

    # ── Live capture ───────────────────────────────────────────────────────────
    def _run_live(self):
        if not SCAPY_OK:
            console.print("[red]scapy not installed — cannot do live capture.[/red]")
            return
        if os.geteuid() != 0:
            console.print("[red]Live capture requires root (sudo).[/red]")
            return
        if RICH_OK:
            console.print(f"[bold green]Starting live capture on [cyan]{self.interface}[/cyan][/bold green]")
        sniff(iface=self.interface, prn=self.proc.process_scapy_pkt,
              store=False, stop_filter=lambda _: self._stop.is_set())

    # ── Dashboard loop ─────────────────────────────────────────────────────────
    def _dashboard_loop(self):
        if not RICH_OK:
            return
        if self.no_live:
            return
        try:
            with Live(console=console, refresh_per_second=1, screen=False) as live:
                while not self._stop.is_set():
                    hdr   = self.dashboard._header()
                    cols  = Columns([self.dashboard._stats_panel(),
                                     self.dashboard._type_panel()])
                    at    = self.dashboard._alert_table()
                    pt    = self.dashboard._profile_table()
                    from rich.console import Group
                    live.update(Group(hdr, cols, at, pt))
                    time.sleep(1)
        except Exception:
            pass

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self):
        if RICH_OK:
            console.print(Panel(
                "[bold cyan]AI-AUGMENTED HYBRID NIDS v2.0[/bold cyan]\n"
                "[dim]Ensemble: RandomForest + IsolationForest + Signature Rules[/dim]",
                box=box.DOUBLE_EDGE, style="blue"
            ))

        # Train / load model
        self._init_model()

        if self.demo:
            # Dashboard in background thread
            dash_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
            dash_thread.start()
            self._run_demo()
            self._stop.set()
            time.sleep(1.5)
        else:
            # Live capture with dashboard in parallel
            dash_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
            dash_thread.start()
            try:
                self._run_live()
            except KeyboardInterrupt:
                self._stop.set()

        # Final summary
        self._print_summary()
        if self.report:
            path = generate_html_report(self.alerts, self.profiler,
                                        self.model, self.proc)
            if RICH_OK:
                console.print(f"[bold green]HTML report saved → [cyan]{path}[/cyan][/bold green]")
            else:
                print(f"Report saved → {path}")

        # Export profiles JSON
        profiles_path = "nids_profiles.json"
        with open(profiles_path, "w") as f:
            json.dump([p.to_dict() for p in self.profiler.all()], f, indent=2)
        if RICH_OK:
            console.print(f"[dim]Threat profiles saved → {profiles_path}[/dim]")

    def _print_summary(self):
        if RICH_OK:
            console.rule("[bold]FINAL SUMMARY[/bold]")
            t = Table(box=box.SIMPLE_HEAVY, show_header=True)
            t.add_column("Metric", style="cyan")
            t.add_column("Value",  style="yellow", justify="right")
            t.add_row("Packets analysed",   str(self.proc.pkt_count))
            t.add_row("Total alerts",        str(self.alerts.total))
            t.add_row("Unique IPs profiled", str(self.profiler.count()))
            t.add_row("Critical alerts",     str(self.alerts.by_sev.get("CRITICAL", 0)))
            t.add_row("High alerts",         str(self.alerts.by_sev.get("HIGH", 0)))
            t.add_row("Medium alerts",       str(self.alerts.by_sev.get("MEDIUM", 0)))
            t.add_row("ML model accuracy",   f"{self.model.accuracy*100:.2f}%")
            console.print(t)
            for atype, cnt in sorted(self.alerts.by_type.items(),
                                     key=lambda x: x[1], reverse=True):
                sev, _ = SEVERITY.get(atype, ("INFO", 0))
                color  = SEV_COLOR.get(sev, "white")
                console.print(f"  [{color}]{atype:20s}[/{color}] : {cnt}")
        else:
            print("\n=== FINAL SUMMARY ===")
            print(f"Packets: {self.proc.pkt_count}")
            print(f"Alerts:  {self.alerts.total}")
            print(f"Accuracy: {self.model.accuracy*100:.2f}%")
            for atype, cnt in sorted(self.alerts.by_type.items(),
                                     key=lambda x: x[1], reverse=True):
                print(f"  {atype}: {cnt}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="AI-Augmented Hybrid NIDS — Threat Scoring & Profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 hybrid_nids.py --demo                  Safe demo, no root needed
  sudo python3 hybrid_nids.py --iface eth0        Live capture on eth0
  sudo python3 hybrid_nids.py --iface wlan0 --report   Live + HTML report
  python3 hybrid_nids.py --demo --report          Demo + HTML report
        """
    )
    parser.add_argument("--demo",     action="store_true", help="Run demo simulation (no root required)")
    parser.add_argument("--iface",    default="eth0",       help="Network interface for live capture")
    parser.add_argument("--report",   action="store_true",  help="Generate HTML report at end")
    parser.add_argument("--no-live",  action="store_true",  help="Disable live dashboard (useful for CI/screenshots)")
    parser.add_argument("--retrain",  action="store_true",  help="Force retrain even if cached model exists")
    args = parser.parse_args()

    if args.retrain:
        for f in (CFG["model_file"], CFG["scaler_file"], CFG["if_file"]):
            try: os.remove(f)
            except FileNotFoundError: pass

    if not args.demo and not SCAPY_OK:
        print("[!] scapy not found and --demo not specified.  Install: pip3 install scapy")
        print("    Falling back to demo mode.")
        args.demo = True

    engine = NIDSEngine(
        interface=args.iface,
        demo=args.demo,
        report=args.report,
        no_live=args.no_live,
    )
    engine.run()


if __name__ == "__main__":
    main()
