/**
 * @file bluetooth_bridge_node.cpp
 * @brief Android tablet <-> ROS2 bridge over the Bluetooth RFCOMM link.
 *
 * Link handling: the RFCOMM device node comes and goes with the tablet's
 * Bluetooth connection, so this node never throws on a failed/lost open; it
 * keeps retrying every reconnect_interval_ms and reports link_ok = false
 * meanwhile. A dropped link NEVER stops the robot (nothing here reacts to it
 * except by replaying state on reconnect): the run is autonomous on the RPi
 * and only an explicit STOP stops it.
 *
 * Protocol (see bluetooth_protocol.hpp; one '\n'-terminated line each):
 *   tablet -> RPi   OBSTACLE,n,x,y,F  collected; an OBSTACLE arriving after a
 *                                     DONE starts a new set (replaces the old)
 *                   DONE              publish the set on /obstacle_cells (cells, e.g. 1:0,1,N|2:9,8,W)
 *                   BEGIN / STOP      call /start_run / /stop_run
 *                   f b fl fr bl br   0.3s /cmd_vel burst then zeros (ignored
 *                                     while the runner drives or is stopped)
 *                   CLEAR             tablet cleared its map: resend ROBOT only
 *   runner -> tablet  the runner publishes ROBOT/PLAN/RESET/ESTOP/STATUS/TARGET lines
 *                   on /bluetooth_tx; this node converts ROBOT to grid cells,
 *                   caches the latest of each, forwards changes, and replays
 *                   the whole cache every time the link comes up.
 * /bluetooth_bridge/app_rx (raw lines in) and app_tx (raw lines out) remain
 * for debugging.
 */

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "mdp_bridge/bluetooth_protocol.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"

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
    /* Tablet manual-drive buttons: one tap = one short burst at these speeds. */
    declare_parameter<double>("drive_speed", 0.15);      /* m/s */
    declare_parameter<double>("drive_turn_rate", 0.8);   /* rad/s */
    declare_parameter<double>("drive_burst_s", 0.3);

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

    setup_pub_ = create_publisher<std_msgs::msg::String>("/obstacle_cells", 10);
    cmd_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel", 10);
    runner_tx_sub_ = create_subscription<std_msgs::msg::String>(
      "/bluetooth_tx", 50,
      std::bind(&BluetoothBridgeNode::onRunnerTx, this, std::placeholders::_1));
    start_cli_ = create_client<std_srvs::srv::Trigger>("/start_run");
    stop_cli_ = create_client<std_srvs::srv::Trigger>("/stop_run");
    drive_timer_ = create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&BluetoothBridgeNode::driveTick, this));

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

  /* Caller must hold fd_mutex_. Writes the whole string; on a hard write
   * error closes the port so the read loop reconnects. */
  bool writeAllLocked(const std::string & out)
  {
    if (fd_ < 0) {
      return false;
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
        return false;
      }
      written += static_cast<size_t>(n);
    }
    return true;
  }

  /* Send one line to the tablet. Must NOT be called with state_mutex_ held
   * (lock order is fd_mutex_ -> state_mutex_). Dropped if not connected; the
   * cache replay on reconnect covers anything missed. */
  void sendLine(const std::string & line)
  {
    std::lock_guard<std::mutex> lock(fd_mutex_);
    if (!writeAllLocked(line + "\n")) {
      RCLCPP_DEBUG(get_logger(), "not sent (link down): %s", line.c_str());
    }
  }

  void onAppTx(const std_msgs::msg::String::SharedPtr msg)
  {
    std::string out = msg->data;
    if (!out.empty() && out.back() == '\n') {
      out.pop_back();
    }
    sendLine(out);
  }

  /* Caller must hold fd_mutex_: on every (re)connect, resend the latest of each
   * cached line so the tablet catches up on whatever it missed. */
  void replayLocked()
  {
    std::vector<std::string> lines;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      for (const auto * l : {&plan_, &reset_, &estop_, &status_, &robot_}) {
        if (!l->empty()) {lines.push_back(*l);}
      }
      for (const auto & kv : targets_) {lines.push_back(kv.second);}
    }
    for (const auto & l : lines) {
      if (!writeAllLocked(l + "\n")) {
        return;
      }
    }
    RCLCPP_INFO(get_logger(), "Link up: replayed %zu cached line(s) to the tablet", lines.size());
  }

  // ---- tablet -> RPi ------------------------------------------------------
  void handleLine(const std::string & raw)
  {
    const std::string line = bt::trim(raw);
    const std::string key = bt::upper(bt::split(line, ',')[0]);
    int lin = 0, ang = 0;

    if (key == "OBSTACLE") {
      bt::ObstacleLine o;
      if (!bt::parseObstacle(line, o)) {
        RCLCPP_WARN(get_logger(), "Bad OBSTACLE line: '%s'", line.c_str());
        return;
      }
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (set_closed_) {   /* first obstacle after a DONE = a new set */
        obstacles_.clear();
        set_closed_ = false;
      }
      if (o.removed) {
        obstacles_.erase(o.n);
      } else {
        obstacles_[o.n] = o.obstacle;
      }
    } else if (key == "DONE") {
      handleDone();
    } else if (key == "BEGIN") {
      handleBegin();
    } else if (key == "STOP") {
      handleStop();
    } else if (key == "CLEAR") {
      std::string robot;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        robot = robot_;
      }
      if (!robot.empty()) {sendLine(robot);}   /* tablet wiped its robot icon */
    } else if (key == "ROBOT") {
      /* the tablet's own start-pose echo: not used, the start pose is a launch setting */
    } else if (bt::driveButton(line, lin, ang)) {
      handleDrive(lin, ang);
    } else {
      RCLCPP_WARN(get_logger(), "Unknown line from tablet: '%s'", line.c_str());
    }
  }

  void handleDone()
  {
    std::string setup;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (obstacles_.empty()) {
        RCLCPP_WARN(get_logger(), "DONE with no obstacles - ignored");
        return;
      }
      setup = bt::formatSetup(obstacles_);
      set_closed_ = true;
      targets_.clear();   /* new layout: old TARGET results no longer apply */
    }
    std_msgs::msg::String msg;
    msg.data = setup;
    setup_pub_->publish(msg);
    RCLCPP_INFO(get_logger(), "DONE: published /obstacle_cells %s", setup.c_str());
  }

  void callTrigger(
    const rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr & cli, const char * name)
  {
    if (!cli->service_is_ready()) {
      RCLCPP_WARN(get_logger(), "%s not available - is the runner up?", name);
      tellRejected("Runner not available");
      return;
    }
    cli->async_send_request(
      std::make_shared<std_srvs::srv::Trigger::Request>(),
      [this, name](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture f) {
        const auto res = f.get();
        if (!res->success) {
          RCLCPP_WARN(get_logger(), "%s rejected: %s", name, res->message.c_str());
          tellRejected(res->message);
        }
      });
  }

  /* One-off STATUS text on the tablet (e.g. why BEGIN was refused). The cached
   * status is flagged so the runner's next status line is sent even if it
   * equals the cached one - the tablet then returns to the true status. */
  void tellRejected(const std::string & why)
  {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      status_dirty_ = true;
    }
    sendLine("STATUS:" + why);
  }

  void handleBegin()
  {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      targets_.clear();   /* a new run starts with a clean TARGET list */
    }
    callTrigger(start_cli_, "/start_run");
  }

  void handleStop()
  {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      drive_until_ns_ = 0;
      zero_ticks_ = 0;
    }
    publishCmd(0.0, 0.0);   /* immediate zero, the runner's service does the rest */
    callTrigger(stop_cli_, "/stop_run");
  }

  void handleDrive(int lin, int ang)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (bt::runnerBusy(status_)) {
      RCLCPP_WARN(get_logger(), "Manual drive ignored: runner is driving or stopped");
      return;
    }
    drive_lin_ = lin * get_parameter("drive_speed").as_double();
    drive_ang_ = ang * get_parameter("drive_turn_rate").as_double();
    drive_until_ns_ = now().nanoseconds() +
      static_cast<int64_t>(get_parameter("drive_burst_s").as_double() * 1e9);
    zero_ticks_ = 3;   /* explicit zeros after the burst: the controller holds the last command */
  }

  void publishCmd(double linear, double angular)
  {
    geometry_msgs::msg::TwistStamped cmd;
    cmd.header.stamp = now();
    cmd.header.frame_id = "base_link";
    cmd.twist.linear.x = linear;
    cmd.twist.angular.z = angular;
    cmd_pub_->publish(cmd);
  }

  void driveTick()
  {
    double lin = 0.0, ang = 0.0;
    bool publish = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (now().nanoseconds() < drive_until_ns_) {
        lin = drive_lin_;
        ang = drive_ang_;
        publish = true;
      } else if (zero_ticks_ > 0) {
        zero_ticks_--;
        publish = true;
      }
    }
    if (publish) {publishCmd(lin, ang);}
  }

  // ---- runner -> tablet ---------------------------------------------------
  void onRunnerTx(const std_msgs::msg::String::SharedPtr msg)
  {
    const std::string text = bt::trim(msg->data);
    std::string out;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (bt::startsWith(text, "ROBOT,")) {
        const std::string conv = bt::convertRobot(text);
        if (conv.empty()) {
          RCLCPP_WARN(get_logger(), "Bad ROBOT line from runner: '%s'", text.c_str());
        } else if (conv != robot_) {
          robot_ = out = conv;
        }
      } else if (bt::startsWith(text, "PLAN:")) {
        if (text != plan_) {plan_ = out = text;}
      } else if (bt::startsWith(text, "RESET:")) {
        if (text != reset_) {reset_ = out = text;}
      } else if (bt::startsWith(text, "ESTOP:")) {
        if (text != estop_) {estop_ = out = text;}
      } else if (bt::startsWith(text, "STATUS:")) {
        if (text != status_ || status_dirty_) {out = text;}
        status_ = text;
        status_dirty_ = false;
      } else if (bt::startsWith(text, "TARGET,")) {
        const auto p = bt::split(text, ',');
        double n = 0;
        if (p.size() >= 3 && bt::parseDouble(p[1], n)) {
          std::string & cached = targets_[static_cast<int>(n)];
          if (cached != text) {
            cached = out = text;
          }
        } else {
          RCLCPP_WARN(get_logger(), "Bad TARGET line from runner: '%s'", text.c_str());
        }
      } else if (!text.empty()) {
        out = text;   /* anything else: forward untouched */
      }
    }
    if (!out.empty()) {sendLine(out);}
  }

  void readLoop()
  {
    while (running_) {
      {
        std::lock_guard<std::mutex> lock(fd_mutex_);
        if (fd_ < 0) {
          openLocked();
          if (fd_ >= 0) {
            replayLocked();
          }
        }
      }

      int fd;
      {
        std::lock_guard<std::mutex> lock(fd_mutex_);
        fd = fd_;
      }
      if (fd < 0) {
        std::this_thread::sleep_for(reconnect_interval_ms_);
        continue;
      }

      /* Wait for data WITHOUT holding fd_mutex_. Blocking in read() under the
       * lock (as this used to) starves every writer - std::mutex is not fair,
       * and this thread re-took it immediately - which stalled the whole
       * single-threaded executor: drive timers and status forwarding ran late. */
      pollfd pfd{fd, POLLIN, 0};
      const int pr = poll(&pfd, 1, 100);
      if (pr == 0 || (pr < 0 && errno == EINTR)) {
        continue; /* timeout: loop to re-check running_ */
      }
      if (pr < 0 || (pfd.revents & (POLLERR | POLLNVAL)) ||
        ((pfd.revents & POLLHUP) && !(pfd.revents & POLLIN)))
      {
        RCLCPP_WARN(get_logger(), "Bluetooth port error/hangup, closing for reconnect");
        std::lock_guard<std::mutex> lock(fd_mutex_);
        if (fd_ == fd) {
          closeLocked();
        }
        continue;
      }

      uint8_t byte;
      ssize_t n;
      {
        std::lock_guard<std::mutex> lock(fd_mutex_);
        if (fd_ != fd) {
          continue; /* closed (and maybe reopened) since poll() */
        }
        n = read(fd_, &byte, 1);
      }

      if (n < 0) {
        if (errno == EAGAIN || errno == EINTR) {
          continue;
        }
        RCLCPP_WARN(get_logger(), "Read from Bluetooth port failed, closing for reconnect");
        std::lock_guard<std::mutex> lock(fd_mutex_);
        if (fd_ == fd) {
          closeLocked();
        }
        continue;
      }
      if (n == 0) {
        if (pfd.revents & POLLHUP) {   /* peer gone: don't spin on an EOF */
          std::lock_guard<std::mutex> lock(fd_mutex_);
          if (fd_ == fd) {
            closeLocked();
          }
        }
        continue;
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
          handleLine(line_);
        }
        line_.clear();
      } else {
        /* Guard against an unterminated garbage stream growing forever if
         * the app connects but never sends a newline. */
        constexpr size_t kMaxLineLen = 4096;
        if (line_.size() < kMaxLineLen) {
          line_.push_back(static_cast<char>(byte));
        } else {
          RCLCPP_WARN(
            get_logger(), "app_rx line exceeded %zu bytes without a newline, dropping",
            kMaxLineLen);
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

  /* Protocol state, guarded by state_mutex_. Lock order: fd_mutex_ -> state_mutex_. */
  std::mutex state_mutex_;
  std::map<int, bt::Obstacle> obstacles_;
  bool set_closed_ = false;                 /* a DONE was received for obstacles_ */
  std::string plan_, reset_, estop_, status_, robot_;   /* latest cached lines for the tablet */
  bool status_dirty_ = false;
  std::map<int, std::string> targets_;      /* TARGET line per obstacle number */
  double drive_lin_ = 0.0, drive_ang_ = 0.0;
  int64_t drive_until_ns_ = 0;
  int zero_ticks_ = 0;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr setup_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr runner_tx_sub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr start_cli_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stop_cli_;
  rclcpp::TimerBase::SharedPtr drive_timer_;

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
