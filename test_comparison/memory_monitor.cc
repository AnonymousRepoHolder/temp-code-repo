// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 The ModelVirtualizer Authors
#include "memory_monitor.h"
#include <cstdio>

std::optional<double> MemoryMonitor::rss_mb() const {
    #ifdef __linux__
    return get_linux_rss_mb();
    #elif defined(_WIN32)
    return get_windows_rss_mb();
    #else
    return std::nullopt;
    #endif
}

void MemoryMonitor::print_mem(const std::string& label) const {
    auto rss = rss_mb();
    if (rss.has_value()) {
        printf("[Memory] %s: RSS=%.2f MiB\n", label.c_str(), *rss);
    } else {
        printf("[Memory] %s: RSS=<unavailable>\n", label.c_str());
    }
}

#ifdef __linux__
std::optional<double> MemoryMonitor::get_linux_rss_mb() const {
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) {
        return std::nullopt;
    }
    
    std::string line;
    while (std::getline(status, line)) {
        if (line.find("VmRSS:") == 0) {
            std::istringstream iss(line);
            std::string label, value, unit;
            iss >> label >> value >> unit;
            try {
                return std::stod(value) / 1024.0; // Convert KB to MB
            } catch (const std::exception&) {
                return std::nullopt;
            }
        }
    }
    return std::nullopt;
}
#endif

#ifdef _WIN32
std::optional<double> MemoryMonitor::get_windows_rss_mb() const {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return std::nullopt;
}
#endif
