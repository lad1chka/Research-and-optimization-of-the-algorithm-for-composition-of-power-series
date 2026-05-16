// Global operator new/delete override that tracks peak heap usage.
#ifndef PSCOMP_WORKSPACE_HEAP_TRACKER_HPP
#define PSCOMP_WORKSPACE_HEAP_TRACKER_HPP

#include <cstdlib>
#include <cstring>
#include <new>

namespace pscomp {

struct HeapTracker {
    static thread_local std::size_t live;
    static thread_local std::size_t peak;
    static thread_local bool        active;

    static void reset() { live = 0; peak = 0; }
    static void alloc(std::size_t n) {
        if (!active) return;
        live += n;
        if (live > peak) peak = live;
    }
    static void dealloc(std::size_t n) {
        if (!active) return;
        live -= n;
    }
};

thread_local std::size_t HeapTracker::live   = 0;
thread_local std::size_t HeapTracker::peak   = 0;
thread_local bool        HeapTracker::active = false;

namespace detail {
inline constexpr std::size_t kHeapHeaderAlign =
    alignof(std::max_align_t) >= sizeof(std::size_t)
        ? alignof(std::max_align_t)
        : sizeof(std::size_t);
}

}  // namespace pscomp

void* operator new(std::size_t n) {
    void* raw = std::malloc(pscomp::detail::kHeapHeaderAlign + n);
    if (!raw) throw std::bad_alloc();
    std::memcpy(raw, &n, sizeof(n));
    pscomp::HeapTracker::alloc(n);
    return static_cast<char*>(raw) + pscomp::detail::kHeapHeaderAlign;
}

void operator delete(void* p) noexcept {
    if (!p) return;
    auto* raw = static_cast<char*>(p) - pscomp::detail::kHeapHeaderAlign;
    std::size_t n;
    std::memcpy(&n, raw, sizeof(n));
    pscomp::HeapTracker::dealloc(n);
    std::free(raw);
}

void operator delete(void* p, std::size_t) noexcept { operator delete(p); }
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete[](void* p) noexcept { operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { operator delete(p); }

#endif
