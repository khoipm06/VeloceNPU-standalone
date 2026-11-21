/// \file npu_platform.cpp
/// \brief which NPU generation this host has (aie2 vs aie2p vs dev04)
#include "utils/npu_platform.hpp"

#include <cstdlib>
#include <fstream>
#include <string>

#if defined(FLM_ENABLE_CORE)
#include "core/corelib_runtime.hpp"
#include "utils/utils.hpp"
#endif

namespace utils {

namespace {

npu_platform detect_device_impl() {
    if (const char* env_plat = std::getenv("VELOCE_PLATFORM")) {
        auto parsed = parse_platform(env_plat);
        if (parsed.has_value()) return *parsed;
    }
    if (const char* env_plat = std::getenv("FLM_FORCE_PLATFORM")) {
        auto parsed = parse_platform(env_plat);
        if (parsed.has_value()) return *parsed;
    }

#if defined(FLM_ENABLE_CORE)
    std::string_view arch = flm::corelib::CorelibRuntime::NpuArch(get_executable_directory());
    if (!arch.empty()) {
        return platform_from_corelib_arch(arch);
    }
#endif

#if defined(__linux__)
    for (int i = 0; i < 4; ++i) {
        std::string pci_path = "/sys/class/accel/accel" + std::to_string(i) + "/device/device";
        std::ifstream f(pci_path);
        if (f.is_open()) {
            std::string dev_id;
            f >> dev_id;
            if (dev_id == "0x1502") {
                return npu_platform::aie2; // Phoenix / Hawk Point (XDNA 1)
            }
            if (dev_id == "0x17f0") {
                return npu_platform::aie2p; // Strix / Strix Halo (XDNA 2)
            }
        }
    }
#endif

    return default_npu_platform();
}

}  // namespace

npu_platform get_device() {
    static const npu_platform platform = detect_device_impl();
    return platform;
}

std::optional<npu_platform> parse_platform(std::string_view text) {
    if (text == platform_id(npu_platform::aie2)) return npu_platform::aie2;
    if (text == platform_id(npu_platform::aie2p)) return npu_platform::aie2p;
    if (text == platform_id(npu_platform::dev04)) return npu_platform::dev04;
    return std::nullopt;
}

}  // namespace utils
