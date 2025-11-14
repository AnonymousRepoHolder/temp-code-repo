// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
#ifndef PEAK_MEMORY_TRACKER_H
#define PEAK_MEMORY_TRACKER_H

#include <atomic>
#include <thread>
#include <chrono>
#include "memory_monitor.h"

/**
 * Peak Memory Tracker Class
 * Provides background thread-based continuous memory monitoring to capture peak RSS usage.
 */
class PeakMemoryTracker {
public:
    /**
     * Constructor
     * @param sample_interval_ms Sampling interval in milliseconds (default: 5ms)
     */
    explicit PeakMemoryTracker(int sample_interval_ms = 5);

    /**
     * Destructor - automatically stops monitoring if still active
     */
    ~PeakMemoryTracker();

    /**
     * Start background monitoring thread
     */
    void start();

    /**
     * Stop background monitoring thread
     */
    void stop();

    /**
     * Get baseline memory in MB (recorded at start())
     */
    double get_baseline_mb() const;

    /**
     * Get peak memory in MB (maximum RSS observed during monitoring)
     */
    double get_peak_mb() const;

    /**
     * Get memory overhead in MB (peak - baseline)
     */
    double get_overhead_mb() const;

private:
    int sample_interval_ms_;
    std::atomic<bool> monitoring_;
    double baseline_mb_;
    std::atomic<double> peak_mb_;
    std::thread monitor_thread_;
    MemoryMonitor mem_monitor_;
};

#endif // PEAK_MEMORY_TRACKER_H
