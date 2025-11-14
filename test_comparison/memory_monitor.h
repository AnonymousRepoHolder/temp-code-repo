// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
#ifndef MEMORY_MONITOR_H
#define MEMORY_MONITOR_H

#include <optional>
#include <string>
#include <iostream>

#ifdef __linux__
#include <fstream>
#include <sstream>
#elif defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

/**
 * Memory Monitor Class
 * Provides RSS (Resident Set Size) memory monitoring for both Windows and Linux
 */
class MemoryMonitor {
public:
    MemoryMonitor() = default;
    
    /**
     * Get current RSS memory usage in MB
     * @return RSS memory in MB, or nullopt if unable to measure
     */
    std::optional<double> rss_mb() const;
    
    /**
     * Print current memory usage with label
     * @param label Description label for the measurement
     */
    void print_mem(const std::string& label) const;
    
private:
    // Platform-specific memory measurement implementations
    #ifdef __linux__
    std::optional<double> get_linux_rss_mb() const;
    #elif defined(_WIN32)
    std::optional<double> get_windows_rss_mb() const;
    #endif
};

#endif // MEMORY_MONITOR_H
