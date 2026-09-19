/**
 * @file bluetooth_bridge_node.cpp
 * @brief Bridges the Android tablet's Bluetooth RFCOMM link to ROS2 topics.
 *
 * Unlike serial_bridge_node's link to the STM32 (a fixed binary protocol
 * over a USB-serial cable that's expected to be present for the node's
 * whole lifetime), the RFCOMM device node here comes and goes with the
 * tablet's Bluetooth connection - the app can disconnect/reconnect at any
 * time (out of range, backgrounded, re-paired, rfcomm daemon restarted).
 * So this node never throws on a failed/lost open; it just keeps retrying
 * in the background at reconnect_interval_ms and reports link_ok = false
 * meanwhile.
 *
 * The app-facing message format is not yet finalized, so the interface is
 * intentionally a raw passthrough: '\n'-terminated text lines read from the
 * port are republished on /bluetooth_bridge/app_rx, and any String
 * published to /bluetooth_bridge/app_tx is written out to the port
 * ('\n'-terminated). Once the actual command/status line format is
 * settled, parsing/formatting should move into (or alongside) this node
 * rather than the raw String topics being consumed directly.
 */

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"

namespace mdp_bridge
{

class BluetoothBridgeNode : public rclcpp::Node
{
public:
  BluetoothBridgeNode()
  : Node("bluetooth_bridge_node")
  {
    declare_parameter<std::string>("serial_port", "/dev/rfcomm0");
    declare_parameter<int>("reconnect_interval_ms", 2000);
    /* Longer than serial_bridge_node's 500ms STM32 fail-safe window - the
     * app has no obligation to send anything periodically (it's an idle
     * text link, not a fixed-rate telemetry stream), so link_ok should only
     * flip false once the port itself has actually gone away for a while,
     * not merely because the app has been quiet. */
    declare_parameter<int>("link_timeout_ms", 5000);

    port_ = get_parameter("serial_port").as_string();
    reconnect_interval_ms_ = std::chrono::milliseconds(
      get_parameter("reconnect_interval_ms").as_int());
    link_timeout_ns_ = std::chrono::milliseconds(
      get_parameter("link_timeout_ms").as_int()).count() * 1'000'000LL;

    app_rx_pub_ = create_publisher<std_msgs::msg::String>("/bluetooth_bridge/app_rx", 10);
    link_ok_pub_ = create_publisher<std_msgs::msg::Bool>("/bluetooth_bridge/link_ok", 10);
    app_tx_sub_ = create_subscription<std_msgs::msg::String>(
      "/bluetooth_bridge/app_tx", 10,
      std::bind(&BluetoothBridgeNode::onAppTx, this, std::placeholders::_1));

    link_watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(250),
      std::bind(&BluetoothBridgeNode::checkLinkHealth, this));

    RCLCPP_INFO(
      get_logger(), "bluetooth_bridge_node starting, target port %s", port_.c_str());

    running_ = true;
    read_thread_ = std::thread(&BluetoothBridgeNode::readLoop, this);
  }

  ~BluetoothBridgeNode() override
  {
    running_ = false;
    if (read_thread_.joinable()) {
      read_thread_.join();
    }
    std::lock_guard<std::mutex> lock(fd_mutex_);
    closeLocked();
  }

private:
  /* Caller must hold fd_mutex_. Leaves fd_ untouched (-1) on failure so the
   * read loop just retries after reconnect_interval_ms_. */
  void openLocked()
  {
    int fd = open(port_.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd < 0) {
      return;
    }

    termios tty{};
    if (tcgetattr(fd, &tty) == 0) {
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

      tty.c_cc[VTIME] = 1; /* 100ms read timeout, so the loop keeps polling running_ */
      tty.c_cc[VMIN] = 0;

      tcsetattr(fd, TCSANOW, &tty);
      /* Not fatal if this device doesn't support one of these settings
       * (rfcomm devices often ignore termios line-discipline knobs
       * entirely) - the fd is still usable either way. */
    }

    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags & ~O_NDELAY);

    fd_ = fd;
    line_.clear();
    last_activity_ns_.store(now().nanoseconds());
    RCLCPP_INFO(get_logger(), "Bluetooth link connected on %s", port_.c_str());
  }

  /* Caller must hold fd_mutex_. */
  void closeLocked()
  {
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
      RCLCPP_WARN(get_logger(), "Bluetooth link on %s closed", port_.c_str());
    }
  }

  void onAppTx(const std_msgs::msg::String::SharedPtr msg)
  {
    std::string out = msg->data;
    if (out.empty() || out.back() != '\n') {
      out.push_back('\n');
    }

    std::lock_guard<std::mutex> lock(fd_mutex_);
    if (fd_ < 0) {
      RCLCPP_WARN(get_logger(), "app_tx dropped, Bluetooth link not connected");
      return;
    }

    size_t written = 0;
    while (written < out.size()) {
      ssize_t n = write(fd_, out.data() + written, out.size() - written);
      if (n < 0) {
        if (errno == EAGAIN || errno == EINTR) {
          continue;
        }
        RCLCPP_WARN(get_logger(), "Write to Bluetooth port failed, closing for reconnect");
        closeLocked();
        return;
      }
      written += static_cast<size_t>(n);
    }
  }

  void readLoop()
  {
    while (running_) {
      {
        std::lock_guard<std::mutex> lock(fd_mutex_);
        if (fd_ < 0) {
          openLocked();
        }
      }

      if (fd_ < 0) {
        std::this_thread::sleep_for(reconnect_interval_ms_);
        continue;
      }

      uint8_t byte;
      ssize_t n;
      {
        std::lock_guard<std::mutex> lock(fd_mutex_);
        if (fd_ < 0) {
          continue;
        }
        n = read(fd_, &byte, 1);
      }

      if (n < 0) {
        if (errno == EAGAIN || errno == EINTR) {
          continue;
        }
        RCLCPP_WARN(get_logger(), "Read from Bluetooth port failed, closing for reconnect");
        std::lock_guard<std::mutex> lock(fd_mutex_);
        closeLocked();
        continue;
      }
      if (n == 0) {
        continue; /* VTIME read timeout, nothing available yet */
      }

      last_activity_ns_.store(now().nanoseconds());

      if (byte == '\n') {
        if (!line_.empty() && line_.back() == '\r') {
          line_.pop_back();
        }
        if (!line_.empty()) {
          std_msgs::msg::String msg;
          msg.data = line_;
          app_rx_pub_->publish(msg);
        }
        line_.clear();
      } else {
        /* Guard against an unterminated garbage stream growing forever if
         * the app connects but never sends a newline. */
        constexpr size_t kMaxLineLen = 4096;
        if (line_.size() < kMaxLineLen) {
          line_.push_back(static_cast<char>(byte));
        } else {
          RCLCPP_WARN(get_logger(), "app_rx line exceeded %zu bytes without a newline, dropping", kMaxLineLen);
          line_.clear();
        }
      }
    }
  }

  void checkLinkHealth()
  {
    bool connected;
    {
      std::lock_guard<std::mutex> lock(fd_mutex_);
      connected = fd_ >= 0;
    }
    const bool link_ok = connected &&
      (now().nanoseconds() - last_activity_ns_.load() < link_timeout_ns_);

    std_msgs::msg::Bool msg;
    msg.data = link_ok;
    link_ok_pub_->publish(msg);
  }

  std::string port_;
  std::chrono::milliseconds reconnect_interval_ms_{2000};
  int64_t link_timeout_ns_ = 0;

  std::mutex fd_mutex_;
  int fd_ = -1;
  std::string line_;

  std::atomic<bool> running_{false};
  std::thread read_thread_;
  std::atomic<int64_t> last_activity_ns_{0};

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr app_rx_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr link_ok_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr app_tx_sub_;
  rclcpp::TimerBase::SharedPtr link_watchdog_timer_;
};

}  // namespace mdp_bridge

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<mdp_bridge::BluetoothBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
