"""
Memory status that respects container limits.

psutil.virtual_memory() (and `free`) report the HOST's physical RAM. Inside a container with a
cgroup memory limit -- docker --memory, Kubernetes limits, Slurm --mem -- the kernel OOM-kills the
job at the cgroup limit long before the host runs out. Admission control and the emergency brake
must therefore use whichever is tighter: the host, or the cgroup.

    memory_status() -> {"total_gb", "available_gb", "source"}   source: "host" | "cgroupv1" | "cgroupv2"
"""
import os

import psutil

GB = 1024 ** 3
CGROUP_V2_ROOT = "/sys/fs/cgroup"
CGROUP_V1_ROOT = "/sys/fs/cgroup/memory"


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _stat(path):
    out = {}
    for line in (_read(path) or "").splitlines():
        key, _, value = line.partition(" ")
        if value.strip().isdigit():
            out[key] = int(value)
    return out


def cgroup_memory(root_v2=CGROUP_V2_ROOT, root_v1=CGROUP_V1_ROOT):
    """(limit_bytes, non_reclaimable_usage_bytes, "v1"|"v2"), or None when there is no limit.

    Usage excludes inactive page cache, which the kernel reclaims before OOM-killing; everything
    else (anon memory, active cache, kernel) is counted as used, to stay on the safe side.
    """
    mx = _read(os.path.join(root_v2, "memory.max"))
    if mx is not None:
        limits = [int(v) for v in (mx, _read(os.path.join(root_v2, "memory.high")))
                  if v is not None and v != "max"]
        if not limits:
            return None
        cur = int(_read(os.path.join(root_v2, "memory.current")) or 0)
        reclaimable = _stat(os.path.join(root_v2, "memory.stat")).get("inactive_file", 0)
        return min(limits), max(0, cur - reclaimable), "v2"

    lim = _read(os.path.join(root_v1, "memory.limit_in_bytes"))
    if lim is not None:
        limit = int(lim)
        if limit >= 1 << 60:          # cgroup v1 spells "unlimited" as a huge number
            return None
        cur = int(_read(os.path.join(root_v1, "memory.usage_in_bytes")) or 0)
        st = _stat(os.path.join(root_v1, "memory.stat"))
        reclaimable = st.get("total_inactive_file", st.get("inactive_file", 0))
        return limit, max(0, cur - reclaimable), "v1"
    return None


def memory_status(root_v2=CGROUP_V2_ROOT, root_v1=CGROUP_V1_ROOT, vm=None):
    vm = vm or psutil.virtual_memory()
    total, available, source = vm.total, vm.available, "host"
    cg = cgroup_memory(root_v2, root_v1)
    if cg:
        limit, used, version = cg
        cg_available = max(0, limit - used)
        if limit < total:
            total = limit
            source = "cgroup" + version
        if cg_available < available:
            available = cg_available
            source = "cgroup" + version
    return {"total_gb": total / GB, "available_gb": available / GB, "source": source}


if __name__ == "__main__":
    s = memory_status()
    print(f"total {s['total_gb']:.2f}GB  available {s['available_gb']:.2f}GB  (source: {s['source']})")
