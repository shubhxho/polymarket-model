#include "cmf/engine.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace nb = nanobind;

using F32 = nb::ndarray<float, nb::numpy, nb::c_contig, nb::device::cpu>;

namespace {

nb::ndarray<nb::numpy, float> own_2d(std::vector<float>&& buf, size_t rows, size_t cols) {
    auto* heap = new std::vector<float>(std::move(buf));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
    return nb::ndarray<nb::numpy, float>(heap->data(), {rows, cols}, owner);
}

nb::ndarray<nb::numpy, float> own_1d(std::vector<float>&& buf, size_t n) {
    auto* heap = new std::vector<float>(std::move(buf));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
    return nb::ndarray<nb::numpy, float>(heap->data(), {n}, owner);
}

void require_2d(const F32& a, const char* name, int cols) {
    if (a.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be 2-D");
    }
    if (static_cast<int>(a.shape(1)) != cols) {
        throw std::invalid_argument(std::string(name) + " has wrong column count");
    }
}

}  // namespace

NB_MODULE(_cmf_native, m) {
    m.doc() = "C++20 cross-market microstructure + Hayashi-Yoshida lead-lag engine";

    m.attr("HISTORY") = cmf::kHistory;
    m.attr("FAST_DIM") = cmf::kFastDim;
    m.attr("SLOW_DIM") = cmf::kSlowDim;
    m.attr("POS_DIM") = cmf::kPosDim;
    m.attr("LAG_DIM") = cmf::kLagDim;
    m.attr("RAW_FAST_DIM") = cmf::kRawFastDim;
    m.attr("RAW_SLOW_DIM") = cmf::kRawSlowDim;

    nb::class_<cmf::Position>(m, "Position")
        .def(nb::init<>())
        .def_rw("has", &cmf::Position::has)
        .def_rw("side", &cmf::Position::side)
        .def_rw("pnl", &cmf::Position::pnl)
        .def_rw("entry", &cmf::Position::entry)
        .def_rw("time_in", &cmf::Position::time_in)
        .def_rw("time_remaining", &cmf::Position::time_remaining)
        .def_rw("shares", &cmf::Position::shares)
        .def_rw("inventory_risk", &cmf::Position::inventory_risk);

    nb::class_<cmf::FusionEngine>(m, "FusionEngine")
        .def(nb::init<>())
        .def("reset", &cmf::FusionEngine::reset)
        .def("set_position", &cmf::FusionEngine::set_position)
        .def(
            "snapshot",
            [](const cmf::FusionEngine& eng) {
                const auto frame = eng.snapshot();
                std::vector<float> fast(frame.fast.begin(), frame.fast.end());
                std::vector<float> slow(frame.slow.begin(), frame.slow.end());
                std::vector<float> pos(frame.pos.begin(), frame.pos.end());
                std::vector<float> lag(frame.lag.begin(), frame.lag.end());
                return nb::make_tuple(
                    own_1d(std::move(fast), cmf::kFastDim),
                    own_1d(std::move(slow), cmf::kSlowDim),
                    own_1d(std::move(pos), cmf::kPosDim),
                    own_1d(std::move(lag), cmf::kLagDim)
                );
            }
        )
        .def(
            "ingest_fast_raw",
            [](cmf::FusionEngine& eng, F32 row) {
                if (row.ndim() != 1 || static_cast<int>(row.shape(0)) != cmf::kRawFastDim) {
                    throw std::invalid_argument("fast raw tick must have shape (16,)");
                }
                eng.ingest_fast_raw(std::span<const float, cmf::kRawFastDim>(row.data(), cmf::kRawFastDim));
            }
        )
        .def(
            "ingest_slow_raw",
            [](cmf::FusionEngine& eng, F32 row) {
                if (row.ndim() != 1 || static_cast<int>(row.shape(0)) != cmf::kRawSlowDim) {
                    throw std::invalid_argument("slow raw tick must have shape (12,)");
                }
                eng.ingest_slow_raw(std::span<const float, cmf::kRawSlowDim>(row.data(), cmf::kRawSlowDim));
            }
        );

    m.def(
        "featurize_episode",
        [](F32 fast_raw, F32 slow_raw) {
            require_2d(fast_raw, "fast_raw", cmf::kRawFastDim);
            require_2d(slow_raw, "slow_raw", cmf::kRawSlowDim);
            const int ticks = static_cast<int>(fast_raw.shape(0));
            if (static_cast<int>(slow_raw.shape(0)) != ticks) {
                throw std::invalid_argument("fast_raw and slow_raw tick counts differ");
            }
            std::vector<float> fast(static_cast<size_t>(ticks) * cmf::kFastDim, 0.0f);
            std::vector<float> slow(static_cast<size_t>(ticks) * cmf::kSlowDim, 0.0f);
            std::vector<float> lag(static_cast<size_t>(ticks) * cmf::kLagDim, 0.0f);
            cmf::featurize_episode(
                std::span<const float>(fast_raw.data(), static_cast<size_t>(ticks) * cmf::kRawFastDim),
                std::span<const float>(slow_raw.data(), static_cast<size_t>(ticks) * cmf::kRawSlowDim),
                ticks,
                std::span<float>(fast.data(), fast.size()),
                std::span<float>(slow.data(), slow.size()),
                std::span<float>(lag.data(), lag.size())
            );
            return nb::make_tuple(
                own_2d(std::move(fast), static_cast<size_t>(ticks), cmf::kFastDim),
                own_2d(std::move(slow), static_cast<size_t>(ticks), cmf::kSlowDim),
                own_2d(std::move(lag), static_cast<size_t>(ticks), cmf::kLagDim)
            );
        },
        nb::arg("fast_raw"),
        nb::arg("slow_raw"),
        "Convert raw dual-venue ticks into fast/slow/lag feature matrices."
    );
}
