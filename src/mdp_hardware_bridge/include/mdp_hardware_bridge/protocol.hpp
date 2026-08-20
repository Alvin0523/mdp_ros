/**
 * @file protocol.hpp
 * @brief Mirror of mdp_stm32/include/protocol.h - USART3 binary protocol.
 *
 * IMPORTANT: hand-mirrored, not shared at build time. If the STM32 firmware
 * changes its packet layout, update this file to match.
 */

#ifndef MDP_HARDWARE_BRIDGE__PROTOCOL_HPP_
#define MDP_HARDWARE_BRIDGE__PROTOCOL_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mdp_hardware_bridge
{

constexpr uint8_t kSync0 = 0xAA;
constexpr uint8_t kSync1 = 0x55;
constexpr uint8_t kTypeTelemetry = 0x01;
constexpr uint8_t kTypeCommand = 0x02;

#pragma pack(push, 1)

struct TelemetryPacket
{
  uint8_t sync0;
  uint8_t sync1;
  uint8_t type;
  int32_t enc_left;
  int32_t enc_right;
  float steer_deg;
  float accel_x;
  float accel_y;
  float accel_z;
  float gyro_x;
  float gyro_y;
  float gyro_z;
  float yaw_deg;
  uint8_t imu_ready;
  uint8_t estop;
  uint32_t uptime_ms;
  uint8_t checksum;
};

struct CommandPacket
{
  uint8_t sync0;
  uint8_t sync1;
  uint8_t type;
  float left_wheel_rad_s;
  float right_wheel_rad_s;
  float steer_rad;
  uint8_t checksum;
};

#pragma pack(pop)

inline uint8_t xor_checksum(const uint8_t * bytes, size_t len)
{
  uint8_t checksum = 0;
  for (size_t i = 0; i < len; i++) {
    checksum ^= bytes[i];
  }
  return checksum;
}

/* Ticks per full wheel revolution: EncoderMultiples(4) * Hall_13(13 pulses/motor-rev) *
 * HALL_30F(30:1 gear ratio) = 1560. Sourced from WHEELTEC's vendor reference firmware
 * (references/WHEELTEC/.../robot_select_init.h, Akm_Car config) for this exact motor
 * (MG513P3012V, Hall encoder) - matches docs/hardware.md's 1:30 ratio and 0.065m wheel
 * diameter. */
constexpr double kTicksPerWheelRev = 1560.0;
constexpr double kRadPerTick = 2.0 * M_PI / kTicksPerWheelRev;

/* MG513P3012V rated max output speed: 330 RPM (post-gearbox) = ~34.56 rad/s.
 * Must match MOTOR_MAX_WHEEL_RAD_S in mdp_stm32/src/motor.c. */
constexpr double kMaxWheelRadS = 34.56;

}  // namespace mdp_hardware_bridge

#endif  // MDP_HARDWARE_BRIDGE__PROTOCOL_HPP_
