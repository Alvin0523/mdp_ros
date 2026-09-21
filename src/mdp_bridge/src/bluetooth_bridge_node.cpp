/**
 * @file bluetooth_bridge_node.cpp
 * @brief Bridges Android Bluetooth RFCOMM messages to ROS 2.
 *
 * Android -> Raspberry Pi:
 *   newline-terminated text messages over /dev/rfcomm0
 *
 * Raspberry Pi -> Android:
 *   newline-terminated text messages from /android/status
 *
 * Example:
 *   Android sends:  MOVE,1,0\n
 *   ROS publishes: /android/command = "MOVE,1,0"
 *
 *   ROS publishes: /android/status = "ROBOT,MOVING"
 *   Android receives: ROBOT,MOVING\n
 */

#include <atomic>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <thread>

#include <termios.h>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace mdp_bridge
{

class BluetoothBridgeNode : public rclcpp::Node
{
public:
  BluetoothBridgeNode()
  : Node("bluetooth_bridge_node")
  {
    declare_parameter<std::string>("bluetooth_port", "/dev/rfcomm0");

    const std::string port =
      get_parameter("bluetooth_port").as_string();

    RCLCPP_INFO(
      get_logger(),
      "Opening Bluetooth RFCOMM device: %s",
      port.c_str());

    fd_ = open(
      port.c_str(),
      O_RDWR | O_NOCTTY);

    if (fd_ < 0) {
      RCLCPP_FATAL(
        get_logger(),
        "Failed to open Bluetooth device %s",
        port.c_str());

      throw std::runtime_error(
        "Bluetooth RFCOMM device open failed");
    }

    /*
     * Put the line discipline into raw mode so control characters
     * (e.g. 0x04 EOF, 0x7F erase) in the app's payload are passed
     * through as data instead of being interpreted by canonical-mode
     * line editing, and set VTIME so read() below can't block forever
     * (which would otherwise hang the destructor's thread join if the
     * link goes quiet during shutdown).
     */
    termios tty{};
    if (tcgetattr(fd_, &tty) == 0) {
      cfmakeraw(&tty);
      tty.c_cc[VTIME] = 1; /* 100ms read timeout */
      tty.c_cc[VMIN] = 0;
      tcsetattr(fd_, TCSANOW, &tty);
    }

    RCLCPP_INFO(
      get_logger(),
      "Bluetooth RFCOMM device opened successfully");

    command_pub_ =
      create_publisher<std_msgs::msg::String>(
        "/android/command",
        10);

    status_sub_ =
      create_subscription<std_msgs::msg::String>(
        "/android/status",
        10,
        std::bind(
          &BluetoothBridgeNode::onStatus,
          this,
          std::placeholders::_1));

    running_ = true;

    read_thread_ =
      std::thread(
        &BluetoothBridgeNode::readLoop,
        this);
  }

  ~BluetoothBridgeNode() override
  {
    running_ = false;

    if (read_thread_.joinable()) {
      read_thread_.join();
    }

    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }

private:

  void readLoop()
  {
    std::string buffer;

    while (running_) {

      char byte;

      const ssize_t n =
        read(fd_, &byte, 1);

      if (n <= 0) {
        std::this_thread::sleep_for(
          std::chrono::milliseconds(10));

        continue;
      }

      /*
       * Android BluetoothService.kt sends
       * newline-terminated messages.
       */
      if (byte == '\n') {

        if (!buffer.empty() && buffer.back() == '\r') {
          buffer.pop_back();
        }

        if (!buffer.empty()) {

          RCLCPP_INFO(
            get_logger(),
            "Android -> ROS: %s",
            buffer.c_str());

          std_msgs::msg::String msg;
          msg.data = buffer;

          command_pub_->publish(msg);
        }

        buffer.clear();

      } else {

        buffer += byte;

        /*
         * Prevent an accidentally malformed message
         * from growing indefinitely.
         */
        if (buffer.size() > 4096) {

          RCLCPP_WARN(
            get_logger(),
            "Bluetooth message exceeded 4096 bytes; clearing buffer");

          buffer.clear();
        }
      }
    }
  }

  void onStatus(
    const std_msgs::msg::String::SharedPtr msg)
  {
    if (fd_ < 0) {
      return;
    }

    std::string payload = msg->data;

    /*
     * Android BluetoothService.kt uses readLine(),
     * so every outgoing message must end with '\n'.
     */
    if (payload.empty() || payload.back() != '\n') {
      payload += '\n';
    }

    const ssize_t written =
      write(
        fd_,
        payload.data(),
        payload.size());

    if (written !=
      static_cast<ssize_t>(payload.size()))
    {
      RCLCPP_WARN(
        get_logger(),
        "Failed to send complete message to Android");
    } else {

      RCLCPP_INFO(
        get_logger(),
        "ROS -> Android: %s",
        msg->data.c_str());
    }
  }

  int fd_ = -1;

  std::atomic<bool> running_{false};

  std::thread read_thread_;

  rclcpp::Publisher<
    std_msgs::msg::String
  >::SharedPtr command_pub_;

  rclcpp::Subscription<
    std_msgs::msg::String
  >::SharedPtr status_sub_;
};

}  // namespace mdp_bridge


int main(
  int argc,
  char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<
      mdp_bridge::BluetoothBridgeNode
    >());

  rclcpp::shutdown();

  return 0;
}
