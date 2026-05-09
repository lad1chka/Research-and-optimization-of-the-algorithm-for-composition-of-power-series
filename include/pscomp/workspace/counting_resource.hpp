// std::pmr::memory_resource that totals allocated/live/peak bytes.

#ifndef PSCOMP_WORKSPACE_COUNTING_RESOURCE_HPP
#define PSCOMP_WORKSPACE_COUNTING_RESOURCE_HPP

#include <atomic>
#include <cstddef>
#include <memory_resource>

namespace pscomp {

class CountingResource final : public std::pmr::memory_resource {
public:
    explicit CountingResource(std::pmr::memory_resource* upstream =
                                  std::pmr::get_default_resource()) noexcept
        : upstream_(upstream) {}

    std::size_t bytes_allocated() const noexcept { return total_.load(); }
    std::size_t high_watermark()  const noexcept { return peak_.load();  }
    std::size_t live_bytes()      const noexcept { return live_.load();  }

    void reset() noexcept { total_ = 0; peak_ = 0; live_ = 0; }

private:
    void* do_allocate(std::size_t bytes, std::size_t align) override {
        void* p = upstream_->allocate(bytes, align);
        total_.fetch_add(bytes, std::memory_order_relaxed);
        std::size_t l = live_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        std::size_t prev = peak_.load(std::memory_order_relaxed);
        while (l > prev && !peak_.compare_exchange_weak(prev, l)) {}
        return p;
    }
    void do_deallocate(void* p, std::size_t bytes, std::size_t align) override {
        live_.fetch_sub(bytes, std::memory_order_relaxed);
        upstream_->deallocate(p, bytes, align);
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }

    std::pmr::memory_resource* upstream_;
    std::atomic<std::size_t>   total_{0};
    std::atomic<std::size_t>   peak_{0};
    std::atomic<std::size_t>   live_{0};
};

}  // namespace pscomp

#endif
