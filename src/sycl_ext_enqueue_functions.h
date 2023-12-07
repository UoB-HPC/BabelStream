#pragma once

#include "CL/sycl.hpp"

// Shim for https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/proposed/sycl_ext_oneapi_enqueue_functions.asciidoc
// Tested against 2023.2.0
namespace sycl::ext::oneapi::experimental {


    template<typename Range, typename Properties = detail::empty_properties_t>
    class launch_config {
        Range r;
        Properties p;
    public:
        launch_config(Range r, Properties p = {}) : r(r), p(p) {}
    };

    template<typename CommandGroupFunc>
    void submit(sycl::queue q, CommandGroupFunc &&cgf) {
        q.submit(std::forward<CommandGroupFunc>(cgf));
    }

    template<typename CommandGroupFunc>
    sycl::event submit_with_event(sycl::queue q, CommandGroupFunc &&cgf) {
        return q.submit(std::forward<CommandGroupFunc>(cgf));
    }


    template<typename KernelName = ::sycl::detail::auto_name, typename KernelType>
    void single_task(sycl::queue &q, const KernelType &k) {
        q.single_task<KernelName>(k);
    }

    template<typename KernelName = ::sycl::detail::auto_name, typename KernelType>
    void single_task(sycl::handler &h, const KernelType &k) {
        h.single_task<KernelName>(k);
    }

    template<typename KernelName = ::sycl::detail::auto_name, int Dimensions,
            typename KernelType, typename... Reductions>
    void parallel_for(sycl::queue &q, sycl::range<Dimensions> r,
                      const KernelType &k, Reductions &&... reductions) {
        q.parallel_for<KernelName>(r, std::forward<Reductions>(reductions)..., k);
    }

    template<typename KernelName = ::sycl::detail::auto_name, int Dimensions,
            typename KernelType, typename... Reductions>
    void parallel_for(sycl::handler &h, sycl::range<Dimensions> r,
                      const KernelType &k, Reductions &&... reductions) {
        h.parallel_for<KernelName>(r, std::forward<Reductions>(reductions)..., k);
    }

    template<typename KernelName = ::sycl::detail::auto_name, int Dimensions, typename... Args>
    void parallel_for(sycl::queue &q, sycl::range<Dimensions> r,
                      const sycl::kernel &k, Args &&... args) {
        q.parallel_for<KernelName>(r, std::forward<Args>(args)..., k);
    }

    template<typename KernelName = ::sycl::detail::auto_name, int Dimensions, typename... Args>
    void parallel_for(sycl::handler &h, sycl::range<Dimensions> r,
                      const sycl::kernel &k, Args &&... args) {
        h.parallel_for<KernelName>(r, std::forward<Args>(args)..., k);
    }


}