/**
 * @file serial_bridge_node.cpp
 * @brief Bridges mdp_stm32's USART3 binary protocol to ROS2 topics.
 *
 * Subscribes /joint_commands (from topic_based_ros2_control) -> sends a
 * CommandPacket to the STM32 over serial.
 * Reads TelemetryPacket frames from the STM32 -> publishes /joint_states,
 * /imu/data, and /estop.
 *
 * NOTE: left_joint/right_joint (Ackermann steering) are commanded
 * independently by ackermann_steering_controller, but this robot has only
 * ONE physical steering servo - the two commanded angles are averaged into
 * a single steer_rad sent to the MCU. This is a small-angle approximation
 * of true Ackermann geometry, not exact per-wheel steering.
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/battery_state.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"

#include "mdp_hardware_bridge/protocol.hpp"

namespace mdp_hardware_bridge
{

class SerialBridgeNode : public rclcpp::Node
{
public:
  SerialBridgeNode()
  : Node("serial_bridge_node")
  {
    declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");
    declare_parameter<int>("baud_rate", 115200);

    const std::string port = get_parameter("serial_port").as_string();
    fd_ = open_serial(port);
    if (fd_ < 0) {
      RCLCPP_FATAL(get_logger(), "Failed to open serial port %s", port.c_str());
      throw std::runtime_error("serial open failed");
    }
    RCLCPP_INFO(get_logger(), "Opened %s for mdp_stm32 bridge", port.c_str());

    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("/imu/data", 10);
    estop_pub_ = create_publisher<std_msgs::msg::Bool>("/estop", 10);
    link_ok_pub_ = create_publisher<std_msgs::msg::Bool>("/hardware_bridge/link_ok", 10);
    battery_pub_ = create_publisher<sensor_msgs::msg::BatteryState>("/battery_state", 10);

    joint_command_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_commands", 10,
      std::bind(&SerialBridgeNode::onJointCommand, this, std::placeholders::_1));

    /* Mirrors the STM32's own PROTOCOL_COMMAND_TIMEOUT_MS fail-safe window
     * (mdp_stm32/src/main.c) - if we haven't seen a valid telemetry frame
     * in that long, treat the link as down. Checked twice as often as the
     * timeout so the flag flips promptly. */
    last_telemetry_ns_.store(now().nanoseconds());
    link_watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(250),
      std::bind(&SerialBridgeNode::checkLinkHealth, this));

    /* Periodic serial-link diagnostic - distinguishes "nothing arriving on
     * the port" (bytes_rx stays 0: wrong device/baud/cable, or firmware not
     * running) from "bytes arriving but framing/checksum always fails"
     * (protocol.h/protocol.hpp struct-size mismatch between the flashed
     * STM32 firmware and this bridge build being the most common cause). */
    diag_timer_ = create_wall_timer(
      std::chrono::seconds(3),
      std::bind(&SerialBridgeNode::logSerialDiagnostic, this));

    running_ = true;
    read_thread_ = std::thread(&SerialBridgeNode::readLoop, this);
  }

  ~SerialBridgeNode() override
  {
    running_ = false;
    if (read_thread_.joinable()) {
      read_thread_.join();
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

private:
  static int open_serial(const std::string & port)
  {
    int fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd < 0) {
      return -1;
    }

    termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
      close(fd);
      return -1;
    }

    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);

    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;

    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tty.c_lflag &= ~ECHOE;
    tty.c_lflag &= ~ISIG;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
    tty.c_oflag &= ~OPOST;
    tty.c_oflag &= ~ONLCR;

    tty.c_cc[VTIME] = 1; /* 100ms read timeout */
    tty.c_cc[VMIN] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
      close(fd);
      return -1;
    }

    /* Clear O_NDELAY so subsequent reads block up to VTIME instead of
     * returning immediately. */
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags & ~O_NDELAY);

    return fd;
  }

  void onJointCommand(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    double left_steer_rad = 0.0, right_steer_rad = 0.0;
    double lb_vel = 0.0, rb_vel = 0.0;
    bool have_left_steer = false, have_right_steer = false;

    /* topic_based_ros2_control's TopicBasedSystem publishes position[] and
     * velocity[] as separate arrays, each containing only the joints that
     * actually use that command interface (in name[] order), NOT one slot
     * per entry in name[]. So position/velocity need their own running
     * indices, not the shared name[] loop index - reusing a single index
     * silently left lb_vel/rb_vel at 0 forever once they stopped lining up
     * with name[]'s indices, and the rear wheels never actually drove. */
    size_t position_idx = 0, velocity_idx = 0;

    for (size_t i = 0; i < msg->name.size(); i++) {
      if (msg->name[i] == "left_joint" && position_idx < msg->position.size()) {
        left_steer_rad = msg->position[position_idx++];
        have_left_steer = true;
      } else if (msg->name[i] == "right_joint" && position_idx < msg->position.size()) {
        right_steer_rad = msg->position[position_idx++];
        have_right_steer = true;
      } else if (msg->name[i] == "lb_joint" && velocity_idx < msg->velocity.size()) {
        lb_vel = msg->velocity[velocity_idx++];
      } else if (msg->name[i] == "rb_joint" && velocity_idx < msg->velocity.size()) {
        rb_vel = msg->velocity[velocity_idx++];
      }
    }

    double steer_rad = 0.0;
    if (have_left_steer && have_right_steer) {
      steer_rad = (left_steer_rad + right_steer_rad) / 2.0;
    } else if (have_left_steer) {
      steer_rad = left_steer_rad;
    } else if (have_right_steer) {
      steer_rad = right_steer_rad;
    }

    CommandPacket pkt{};
    pkt.sync0 = kSync0;
    pkt.sync1 = kSync1;
    pkt.type = kTypeCommand;
    pkt.left_wheel_rad_s = static_cast<float>(lb_vel);
    pkt.right_wheel_rad_s = static_cast<float>(rb_vel);
    /* Sign convention mismatch: ROS's left_joint/right_joint (axis 0 0 1)
     * follow REP-103 - positive = left (CCW from above). The STM32's
     * servo_set_angle() (mdp_stm32/src/servo.c) is documented the other
     * way - positive = right. Negate here, at the one seam that already
     * hand-translates between the two systems, rather than changing either
     * side's own internally-consistent convention. */
    pkt.steer_rad = static_cast<float>(-steer_rad);
    pkt.checksum = xor_checksum(
      reinterpret_cast<const uint8_t *>(&pkt.type),
      sizeof(CommandPacket) - offsetof(CommandPacket, type) - 1);

    ssize_t written = write(fd_, &pkt, sizeof(pkt));
    if (written != static_cast<ssize_t>(sizeof(pkt))) {
      RCLCPP_WARN(get_logger(), "Short/failed write to serial port");
    }
  }

  void readLoop()
  {
    enum class State { WaitSync0, WaitSync1, WaitType, Payload };
    State state = State::WaitSync0;

    constexpr size_t kPayloadLen = sizeof(TelemetryPacket) - 2; /* minus 2 sync bytes */
    uint8_t buf[kPayloadLen];
    size_t index = 0;

    while (running_) {
      uint8_t byte;
      ssize_t n = read(fd_, &byte, 1);
      if (n <= 0) {
        continue; /* timeout (VTIME) or nothing available yet */
      }
      bytes_rx_.fetch_add(1);

      switch (state) {
        case State::WaitSync0:
          if (byte == kSync0) {
            state = State::WaitSync1;
          }
          break;

        case State::WaitSync1:
          state = (byte == kSync1) ? State::WaitType : State::WaitSync0;
          break;

        case State::WaitType:
          if (byte == kTypeTelemetry) {
            buf[0] = byte;
            index = 1;
            state = State::Payload;
          } else {
            state = State::WaitSync0;
          }
          break;

        case State::Payload:
          buf[index++] = byte;
          if (index >= kPayloadLen) {
            uint8_t expected = buf[index - 1];
            uint8_t computed = xor_checksum(buf, index - 1);
            if (computed == expected) {
              frames_ok_.fetch_add(1);
              TelemetryPacket pkt{};
              pkt.sync0 = kSync0;
              pkt.sync1 = kSync1;
              std::memcpy(&pkt.type, buf, index);
              onTelemetry(pkt);
            } else {
              frames_bad_.fetch_add(1);
              RCLCPP_DEBUG(get_logger(), "Telemetry checksum mismatch, dropping frame");
            }
            state = State::WaitSync0;
          }
          break;
      }
    }
  }

  void logSerialDiagnostic()
  {
    const uint64_t bytes = bytes_rx_.exchange(0);
    const uint64_t ok = frames_ok_.exchange(0);
    const uint64_t bad = frames_bad_.exchange(0);
    if (bytes == 0) {
      RCLCPP_WARN(
        get_logger(),
        "Serial: 0 bytes received in the last 3s - check serial_port, baud rate, cable, "
        "and that the STM32 firmware is actually running (try 'pixi run monitor' from "
        "mdp_stm32 with this bridge node stopped, to rule out the ROS side entirely).");
    } else if (ok == 0 && bad > 0) {
      RCLCPP_WARN(
        get_logger(),
        "Serial: %lu bytes received, %lu frames failed checksum, 0 valid telemetry frames "
        "in the last 3s - likely a TelemetryPacket size/layout mismatch between the flashed "
        "STM32 firmware and this bridge build (protocol.h vs protocol.hpp out of sync, or "
        "STM32 not reflashed after a protocol change).",
        bytes, bad);
    } else {
      RCLCPP_INFO(get_logger(), "Serial: %lu bytes, %lu OK frames, %lu bad frames in last 3s", bytes, ok, bad);
    }
  }

  void checkLinkHealth()
  {
    constexpr int64_t kLinkTimeoutNs = 500'000'000; /* 500ms, matches the MCU's fail-safe window */
    const bool link_ok = now().nanoseconds() - last_telemetry_ns_.load() < kLinkTimeoutNs;
    std_msgs::msg::Bool msg;
    msg.data = link_ok;
    link_ok_pub_->publish(msg);
  }

  void onTelemetry(const TelemetryPacket & pkt)
  {
    const auto stamp = now();
    last_telemetry_ns_.store(stamp.nanoseconds());

    /* Differentiate cumulative ticks against the last packet to get
     * wheel angular velocity (rad/s). Uses the MCU's own uptime_ms so
     * jitter in host-side scheduling/serial latency doesn't skew dt. */
    double lb_vel = 0.0, rb_vel = 0.0;
    if (have_prev_telemetry_) {
      double dt_s = static_cast<double>(pkt.uptime_ms - prev_uptime_ms_) / 1000.0;
      if (dt_s > 0.0) {
        lb_vel = (pkt.enc_left - prev_enc_left_) * kRadPerTick / dt_s;
        rb_vel = (pkt.enc_right - prev_enc_right_) * kRadPerTick / dt_s;
      }
    }
    prev_enc_left_ = pkt.enc_left;
    prev_enc_right_ = pkt.enc_right;
    prev_uptime_ms_ = pkt.uptime_ms;
    have_prev_telemetry_ = true;

    sensor_msgs::msg::JointState js;
    js.header.stamp = stamp;
    js.name = {"left_joint", "right_joint", "lb_joint", "rb_joint"};
    /* Negated for the same reason as in onJointCommand() above - pkt.steer_deg
     * is in the STM32's positive=right convention, ROS's left_joint/right_joint
     * expect positive=left. */
    const double steer_rad = -(pkt.steer_deg * M_PI / 180.0);
    js.position = {
      steer_rad, steer_rad,
      pkt.enc_left * kRadPerTick, pkt.enc_right * kRadPerTick
    };
    js.velocity = {0.0, 0.0, lb_vel, rb_vel};
    joint_state_pub_->publish(js);

    sensor_msgs::msg::Imu imu;
    imu.header.stamp = stamp;
    imu.header.frame_id = "imu_link";
    imu.angular_velocity.x = pkt.gyro_x * M_PI / 180.0;
    imu.angular_velocity.y = pkt.gyro_y * M_PI / 180.0;
    imu.angular_velocity.z = pkt.gyro_z * M_PI / 180.0;
    imu.linear_acceleration.x = pkt.accel_x;
    imu.linear_acceleration.y = pkt.accel_y;
    imu.linear_acceleration.z = pkt.accel_z;
    /* Only yaw is estimated, and it's raw gyro-Z integration with no bias
     * calibration and no accel fusion (mdp_stm32/src/imu.c:imu_update()) -
     * expect drift over time. Roll/pitch are not estimated at all, so their
     * covariance is set high to tell consumers (e.g. robot_localization)
     * not to trust them. */
    const double yaw_rad = pkt.yaw_deg * M_PI / 180.0;
    imu.orientation.z = std::sin(yaw_rad / 2.0);
    imu.orientation.w = std::cos(yaw_rad / 2.0);
    imu.orientation_covariance = {
      1e6, 0, 0,
      0, 1e6, 0,
      0, 0, 0.05
    };
    if (!pkt.imu_ready) {
      imu.angular_velocity_covariance[0] = -1; /* signal "data invalid" */
    }
    imu_pub_->publish(imu);

    std_msgs::msg::Bool estop_msg;
    estop_msg.data = pkt.estop != 0;
    estop_pub_->publish(estop_msg);

    sensor_msgs::msg::BatteryState battery;
    battery.header.stamp = stamp;
    battery.voltage = pkt.battery_v;
    battery.present = true;
    /* Pack is 3S NCR18650B Li-ion (docs/hardware.md) - nominal 11.1V,
     * full-charge 12.6V. No current/charge sensing exists yet, so those
     * fields are left at BatteryState's default NaN ("unknown") rather
     * than guessed. */
    battery.design_capacity = 3.4f; /* Ah, 3400mAh pack */
    battery.power_supply_technology = sensor_msgs::msg::BatteryState::POWER_SUPPLY_TECHNOLOGY_LION;
    battery.power_supply_status = sensor_msgs::msg::BatteryState::POWER_SUPPLY_STATUS_UNKNOWN;
    battery.power_supply_health = sensor_msgs::msg::BatteryState::POWER_SUPPLY_HEALTH_UNKNOWN;
    battery_pub_->publish(battery);
  }

  int fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread read_thread_;

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr estop_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr link_ok_pub_;
  rclcpp::Publisher<sensor_msgs::msg::BatteryState>::SharedPtr battery_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_command_sub_;
  rclcpp::TimerBase::SharedPtr link_watchdog_timer_;
  rclcpp::TimerBase::SharedPtr diag_timer_;
  std::atomic<int64_t> last_telemetry_ns_{0};
  std::atomic<uint64_t> bytes_rx_{0};
  std::atomic<uint64_t> frames_ok_{0};
  std::atomic<uint64_t> frames_bad_{0};

  bool have_prev_telemetry_ = false;
  int32_t prev_enc_left_ = 0;
  int32_t prev_enc_right_ = 0;
  uint32_t prev_uptime_ms_ = 0;
};

}  // namespace mdp_hardware_bridge

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<mdp_hardware_bridge::SerialBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
