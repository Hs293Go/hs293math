// Generate the rotation golden record: seeded inputs + this library's outputs,
// one safetensors file. Downstream ports (e.g. isaacflight's so3.py) verify
// against the committed tests/golden/rotation_goldens.safetensors instead of
// building this library -- the file IS the parity contract, so regenerate and
// recommit it deliberately, only when rotation semantics change.
//
// Copyright © 2026 H S Helson Go
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "hs293go/rotation.hpp"

namespace {

constexpr int kNumSamples = 512;
constexpr unsigned kSeed = 7;

struct NamedTensor {
  std::string name;
  std::vector<std::int64_t> shape;
  std::vector<float> data;
};

// Minimal safetensors writer (little-endian F32 only): 8-byte header length,
// hand-built JSON header, raw buffers. Kept dependency-free on purpose.
void SaveSafeTensors(const std::string& path,
                     const std::vector<NamedTensor>& tensors) {
  std::string header = "{";
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    const std::uint64_t bytes = t.data.size() * sizeof(float);
    header += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) {
      header += (i ? "," : "") + std::to_string(t.shape[i]);
    }
    header += "],\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + bytes) + "]},";
    offset += bytes;
  }
  header.back() = '}';  // replace the trailing comma

  std::ofstream file(path, std::ios::binary);
  const std::uint64_t header_len = header.size();
  file.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  file.write(header.data(), static_cast<std::streamsize>(header_len));
  for (const auto& t : tensors) {
    file.write(reinterpret_cast<const char*>(t.data.data()),
               static_cast<std::streamsize>(t.data.size() * sizeof(float)));
  }
}

void Push(NamedTensor& t, const float* values, int count) {
  t.data.insert(t.data.end(), values, values + count);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string out =
      argc > 1 ? argv[1] : "tests/golden/rotation_goldens.safetensors";
  constexpr float kPi = std::numbers::pi_v<float>;
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<float> sym(-1.0F, 1.0F);
  std::normal_distribution<float> gauss(0.0F, 1.0F);

  NamedTensor angle_axis{"angle_axis", {kNumSamples, 3}, {}};
  NamedTensor quat{"quat_xyzw", {kNumSamples, 4}, {}};
  NamedTensor euler{"euler_rpy", {kNumSamples, 3}, {}};
  NamedTensor q_from_aa{"quat_from_angle_axis_xyzw", {kNumSamples, 4}, {}};
  NamedTensor aa_from_q{"angle_axis_from_quat", {kNumSamples, 3}, {}};
  NamedTensor m_from_aa{
      "matrix_from_angle_axis_rowmajor", {kNumSamples, 9}, {}};
  NamedTensor q_from_eu{"quat_from_euler_xyzw", {kNumSamples, 4}, {}};
  NamedTensor eu_from_q{"euler_from_quat_rpy", {kNumSamples, 3}, {}};

  for (int i = 0; i < kNumSamples; ++i) {
    // Angle-axis over the principal range; the first tenth stays tiny to
    // exercise the small-angle Taylor branches.
    Eigen::Vector3f axis(gauss(rng), gauss(rng), gauss(rng));
    axis.normalize();
    float angle = sym(rng) * 0.99F * kPi;
    if (i < kNumSamples / 10) {
      angle *= 1e-4F;
    }
    const Eigen::Vector3f aa = axis * angle;

    // Canonical (w >= 0) unit quaternion so inverse maps are sign-unambiguous.
    Eigen::Vector4f qv(gauss(rng), gauss(rng), gauss(rng), gauss(rng));
    qv.normalize();
    if (qv.w() < 0.0F) {
      qv = -qv;
    }
    const Eigen::Quaternionf q(qv.w(), qv.x(), qv.y(), qv.z());

    // Aerospace ZYX euler; pitch held off the gimbal lock.
    const auto eu = hs293go::EulerAngles<float>{.roll = sym(rng) * 0.99F * kPi,
                                                .pitch = sym(rng) * 1.45F,
                                                .yaw = sym(rng) * 0.99F * kPi};

    Push(angle_axis, aa.data(), 3);
    const float q_in[] = {q.x(), q.y(), q.z(), q.w()};
    Push(quat, q_in, 4);
    const float eu_in[] = {eu.roll, eu.pitch, eu.yaw};
    Push(euler, eu_in, 3);

    const Eigen::Quaternionf qa = hs293go::AngleAxisToQuaternion(aa);
    const float qa_out[] = {qa.x(), qa.y(), qa.z(), qa.w()};
    Push(q_from_aa, qa_out, 4);

    const Eigen::Vector3f aq = hs293go::QuaternionToAngleAxis(q);
    Push(aa_from_q, aq.data(), 3);

    const Eigen::Matrix3f m = hs293go::AngleAxisToRotationMatrix(aa);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        m_from_aa.data.push_back(m(r, c));
      }
    }

    const Eigen::Quaternionf qe = hs293go::EulerAnglesToQuaternion(eu);
    const float qe_out[] = {qe.x(), qe.y(), qe.z(), qe.w()};
    Push(q_from_eu, qe_out, 4);

    const auto ef = hs293go::QuaternionToEulerAngles(q);
    const float ef_out[] = {ef.roll, ef.pitch, ef.yaw};
    Push(eu_from_q, ef_out, 3);
  }

  SaveSafeTensors(out, {angle_axis, quat, euler, q_from_aa, aa_from_q,
                        m_from_aa, q_from_eu, eu_from_q});
  std::printf("wrote %s (%d samples x 5 ops)\n", out.c_str(), kNumSamples);
  return 0;
}
