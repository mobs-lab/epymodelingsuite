"""Resource tracking for telemetry - memory and CPU usage monitoring."""

import os
import warnings
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None
    warnings.warn(
        "psutil not available - memory tracking will be disabled. "
        "Install psutil for memory metrics: pip install psutil",
        ImportWarning,
        stacklevel=2,
    )


class ResourceTracker:
    """Track memory and CPU resource usage for a process.

    This class encapsulates resource tracking functionality, monitoring
    memory usage (baseline and peak) and CPU time (user and system) for
    the current process. It provides a clean interface for resource
    monitoring without cluttering the main telemetry class.

    Attributes
    ----------
    _process : psutil.Process | None
        Process handle for resource tracking (None if psutil unavailable)
    _baseline_memory : float
        Baseline memory usage in MB at initialization
    _peak_memory_mb : float
        Peak memory usage in MB since initialization
    """

    def __init__(self) -> None:
        """Initialize resource tracker with current process."""
        self._process = psutil.Process(os.getpid()) if psutil else None
        self._baseline_memory = self.get_current_memory_mb()
        self._peak_memory_mb = self._baseline_memory

    def get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB.

        Returns
        -------
        float
            Current memory usage in megabytes (0.0 if psutil unavailable)
        """
        if self._process:
            return self._process.memory_info().rss / (1024 * 1024)
        return 0.0

    def update_peak_memory(self) -> None:
        """Update peak memory if current usage is higher.

        Compares current memory usage to the stored peak and updates
        the peak if current usage exceeds it.
        """
        current = self.get_current_memory_mb()
        self._peak_memory_mb = max(self._peak_memory_mb, current)

    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB.

        Returns
        -------
        float
            Peak memory usage in megabytes since initialization
        """
        return self._peak_memory_mb

    def get_cpu_times(self) -> tuple[float, float]:
        """Get CPU time (user, system) in seconds.

        Returns
        -------
        tuple[float, float]
            Tuple of (user_time, system_time) in seconds.
            Returns (0.0, 0.0) if psutil unavailable.
        """
        if self._process:
            cpu_times = self._process.cpu_times()
            return cpu_times.user, cpu_times.system
        return 0.0, 0.0

    def finalize(self) -> dict[str, Any]:
        """Get final resource metrics.

        Collects and returns all resource usage metrics including
        peak memory and CPU time for inclusion in telemetry reports.

        Returns
        -------
        dict
            Dictionary containing:
            - peak_memory_mb: Peak memory usage in MB
            - cpu_time_user_seconds: User CPU time in seconds
            - cpu_time_system_seconds: System CPU time in seconds
        """
        user_time, system_time = self.get_cpu_times()
        return {
            "peak_memory_mb": self._peak_memory_mb,
            "cpu_time_user_seconds": user_time,
            "cpu_time_system_seconds": system_time,
        }
