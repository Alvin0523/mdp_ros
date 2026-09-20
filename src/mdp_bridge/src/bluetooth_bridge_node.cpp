/**
 * @file bluetooth_bridge_node.cpp
 * @brief Runs a BlueZ RFCOMM server socket that accepts the Android
 *        tablet's Bluetooth connection and bridges it to ROS 2.
 *
 * Unlike the earlier /dev/rfcommN approach (which required something
 * outside this node - `rfcomm bind`/`rfcomm listen` run by hand - to
 * already have accepted the phone's connection before the node could even
 * open the device file), this node owns the connection itself: it opens a
 * kernel Bluetooth socket, bind()s it to rfcomm_channel, and listen()s for
 * the tablet to connect. No pairing/bind step is required beyond the
 * devices being paired at the OS level.
 *
 * NOTE ON DISCOVERY: this binds a fixed RFCOMM channel but does not
 * register an SDP service record, so it only works if the Android side
 * connects with a fixed-channel BluetoothSocket (e.g. via
 * `BluetoothDevice.createRfcommSocket(channel)` through reflection,
 * channel matching rfcomm_channel below - the same fixed-channel
 * convention the old `rfcomm bind rfcomm0 <MAC> 1` setup used). If the
 * app instead uses `createRfcommSocketToServiceRecord(uuid)` (SDP
 * lookup), this socket won't be discovered - registering an SDP record
 * (via BlueZ's D-Bus profile API or `sdptool`) would additionally be
 * needed, and isn't included here as the app's exact connection method
 * wasn't known at the time this was written.
 *
 * Android -> Raspberry Pi:
 *   newline-terminated text messages over the RFCOMM connection
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
#include <mutex>
#include <string>
#include <thread>

#include <bluetooth/bluetooth.h>
#include <bluetooth/rfcomm.h>
#include <sys/socket.h>
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
    declare_parameter<int>("rfcomm_channel", 1);
    const int channel = get_parameter("rfcomm_channel").as_int();

    server_fd_ = socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM);
    if (server_fd_ < 0) {
      RCLCPP_FATAL(
        get_logger(),
        "Failed to create RFCOMM socket - is a Bluetooth adapter present and BlueZ running?");
      throw std::runtime_error("RFCOMM socket creation failed");
    }

    sockaddr_rc local_addr{};
    local_addr.rc_family = AF_BLUETOOTH;
    local_addr.rc_bdaddr = {}; /* all-zero = BDADDR_ANY, bind to the local adapter */
    local_addr.rc_channel = static_cast<uint8_t>(channel);

    if (bind(server_fd_, reinterpret_cast<sockaddr *>(&local_addr), sizeof(local_addr)) < 0) {
      RCLCPP_FATAL(get_logger(), "Failed to bind RFCOMM socket on channel %d", channel);
      close(server_fd_);
      throw std::runtime_error("RFCOMM bind failed");
    }

    if (listen(server_fd_, 1) < 0) {
      RCLCPP_FATAL(get_logger(), "Failed to listen on RFCOMM socket");
      close(server_fd_);
      throw std::runtime_error("RFCOMM listen failed");
    }

    RCLCPP_INFO(
      get_logger(),
      "Listening for Android Bluetooth connection on RFCOMM channel %d",
      channel);

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

    accept_thread_ =
      std::thread(
        &BluetoothBridgeNode::acceptLoop,
        this);
  }

  ~BluetoothBridgeNode() override
  {
    running_ = false;

    /* Unblock a thread parked in accept()/read() so the join below can't
     * hang, since neither call otherwise notices running_ going false. */
    if (server_fd_ >= 0) {
      shutdown(server_fd_, SHUT_RDWR);
      close(server_fd_);
    }
    {
      std::lock_guard<std::mutex> lock(client_mutex_);
      if (client_fd_ >= 0) {
        shutdown(client_fd_, SHUT_RDWR);
        close(client_fd_);
      }
    }

    if (accept_thread_.joinable()) {
      accept_thread_.join();
    }
  }

private:

  void acceptLoop()
  {
    while (running_) {

      sockaddr_rc remote_addr{};
      socklen_t addr_len = sizeof(remote_addr);

      const int fd = accept(
        server_fd_,
        reinterpret_cast<sockaddr *>(&remote_addr),
        &addr_len);

      if (fd < 0) {
        if (running_) {
          RCLCPP_WARN(get_logger(), "accept() on RFCOMM socket failed, retrying");
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        continue;
      }

      char addr_str[19] = {0};
      ba2str(&remote_addr.rc_bdaddr, addr_str);
      RCLCPP_INFO(get_logger(), "Android connected from %s", addr_str);

      {
        std::lock_guard<std::mutex> lock(client_mutex_);
        client_fd_ = fd;
      }

      readClient(fd);

      {
        std::lock_guard<std::mutex> lock(client_mutex_);
        if (client_fd_ == fd) {
          client_fd_ = -1;
        }
      }
      close(fd);

      if (running_) {
        RCLCPP_WARN(get_logger(), "Android disconnected, waiting for next connection");
      }
    }
  }

  void readClient(int fd)
  {
    std::string buffer;

    while (running_) {

      char byte;

      const ssize_t n =
        read(fd, &byte, 1);

      if (n <= 0) {
        break; /* disconnected, or a fatal read error - back to accept() */
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
    std::lock_guard<std::mutex> lock(client_mutex_);

    if (client_fd_ < 0) {
      RCLCPP_WARN(get_logger(), "No Android device connected, dropping status message");
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
        client_fd_,
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

  int server_fd_ = -1;
  int client_fd_ = -1;
  std::mutex client_mutex_;

  std::atomic<bool> running_{false};

  std::thread accept_thread_;

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
