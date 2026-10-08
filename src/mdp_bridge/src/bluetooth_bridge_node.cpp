/**
 * @file bluetooth_bridge_node.cpp
 * @brief Android tablet <-> ROS bridge over a Bluetooth RFCOMM serial device.
 *
 * The link is a plain serial device (default /dev/rfcomm0). Every message is one
 * line ending in '\n'. The bridge only translates - all decisions live in
 * task1_runner.
 *
 * Tablet -> ROS
 *   OBSTACLE,<n>,<col>,<row>,<N|E|S|W>  collected; col,row are the tablet's CELL,
 *                                   0..19 (10 cm cells, (0,0) bottom-left). An OBSTACLE
 *                                   arriving after a DONE starts a new set. (Until
 *                                   2026-10-08 the tablet sent the cell's corner in cm.)
 *   DONE                            publish the set on /obstacle_setup (metres,
 *                                   at the CELL CENTRE = (col + 0.5) * 10 cm)
 *   BEGIN                           call /start_run
 *   STOP                            call /stop_run
 *   RESET                           call /reset_pose (car back at the start pose,
 *                                   like `pixi run reset`)
 *   CLEAR                           resend the current ROBOT line
 *   f b fl fr bl br                 publish on /manual_drive
 *
 * ROS -> tablet: every String on /bluetooth_tx is written as a line. The latest
 * ROBOT / PLAN / RESET / STATUS line is remembered and resent when the link
 * comes up (and ROBOT again on CLEAR).
 *
 * Creating the device (not done here): e.g. `sudo rfcomm listen /dev/rfcomm0 1`
 * after pairing. The node keeps retrying to open it, so start order does not
 * matter, and it reopens if the link drops.
 */

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace mdp_bridge
{

namespace
{

std::string trim(const std::string & s)
{
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {++b;}
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {--e;}
  return s.substr(b, e - b);
}

std::string upper(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
    [](unsigned char c) {return static_cast<char>(std::toupper(c));});
  return s;
}

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  return s;
}

std::vector<std::string> splitTrimmed(const std::string & s, char sep)
{
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    const size_t pos = s.find(sep, start);
    out.push_back(trim(s.substr(start, pos == std::string::npos ? pos : pos - start)));
    if (pos == std::string::npos) {break;}
    start = pos + 1;
  }
  return out;
}

std::optional<int> parseInt(const std::string & s)
{
  if (s.empty()) {return std::nullopt;}
  char * end = nullptr;
  errno = 0;
  const long v = std::strtol(s.c_str(), &end, 10);
  if (errno != 0 || *end != '\0') {return std::nullopt;}
  return static_cast<int>(v);
}

/// N/E/S/W, or the full words the spec says the bridge also accepts.
std::optional<char> parseFacing(const std::string & s)
{
  const std::string u = upper(s);
  if (u == "N" || u == "NORTH") {return 'N';}
  if (u == "E" || u == "EAST") {return 'E';}
  if (u == "S" || u == "SOUTH") {return 'S';}
  if (u == "W" || u == "WEST") {return 'W';}
  return std::nullopt;
}

struct Obstacle
{
  int n;
  int col;
  int row;
  char facing;
};

/// The tablet's grid: 20 x 20 cells of 10 cm. OBSTACLE col,row name a cell; the
/// obstacle block fills it, so its centre is half a cell in from its corner.
constexpr int kCells = 20;
constexpr int kCellCm = 10;

}  // namespace

class BluetoothBridgeNode : public rclcpp::Node
{
public:
  BluetoothBridgeNode()
  : Node("bluetooth_bridge_node")
  {
    device_ = declare_parameter<std::string>("device", "/dev/rfcomm0");
    reopen_period_s_ = declare_parameter<double>("reopen_period_s", 1.0);

    setup_pub_ = create_publisher<std_msgs::msg::String>("/obstacle_setup", 10);
    manual_pub_ = create_publisher<std_msgs::msg::String>("/manual_drive", 10);
    // Monitoring only - nothing consumes these. /bluetooth_rx is every line the
    // tablet sent (raw, trimmed), the mirror of /bluetooth_tx; link_ok mirrors
    // /hardware_bridge/link_ok for the tablet link.
    rx_pub_ = create_publisher<std_msgs::msg::String>("/bluetooth_rx", 50);
    link_pub_ = create_publisher<std_msgs::msg::Bool>("/bluetooth_bridge/link_ok", 10);
    link_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {publishLink();});
    tx_sub_ = create_subscription<std_msgs::msg::String>(
      "/bluetooth_tx", 50,
      [this](const std_msgs::msg::String::SharedPtr msg) {onTx(msg->data);});
    start_client_ = create_client<std_srvs::srv::Trigger>("/start_run");
    stop_client_ = create_client<std_srvs::srv::Trigger>("/stop_run");
    reset_client_ = create_client<std_srvs::srv::Trigger>("/reset_pose");

    timer_ = create_wall_timer(std::chrono::milliseconds(10), [this]() {poll();});
    RCLCPP_INFO(get_logger(), "BT        waiting for the tablet on %s", device_.c_str());
  }

  ~BluetoothBridgeNode() override {closeLink("shutdown");}

private:
  // ---------------------------------------------------------------- link ----

  void openLink()
  {
    fd_ = ::open(device_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
      // Keep retrying every reopen_period_s. No such device = no tablet yet,
      // which the "waiting" line already said; any other reason is a real
      // problem, said once (and again only if it changes). The live state is
      // on /diagnostics (mdp/Tablet link) and /bluetooth_bridge/link_ok.
      const std::string why = std::strerror(errno);
      if (errno != ENOENT && why != last_open_error_) {
        last_open_error_ = why;
        RCLCPP_WARN(get_logger(), "BT        cannot open %s: %s - retrying", device_.c_str(), why.c_str());
      }
      return;
    }
    last_open_error_.clear();
    termios tio{};
    if (tcgetattr(fd_, &tio) == 0) {
      cfmakeraw(&tio);
      // VMIN=1 with O_NONBLOCK: read() gives EAGAIN when idle, so a return of 0
      // unambiguously means the peer hung up.
      tio.c_cc[VMIN] = 1;
      tio.c_cc[VTIME] = 0;
      tcsetattr(fd_, TCSANOW, &tio);
    }
    rx_buf_.clear();
    tx_buf_.clear();
    got_rx_since_open_ = false;
    RCLCPP_INFO(get_logger(), "BT        link up (%s)", device_.c_str());
    publishLink();
    resendState();
  }

  /// True while the RFCOMM device is open with no read/write error since.
  /// (The device can open a moment before the tablet's first line - see
  /// handleLine - so this means "link up", not "tablet has spoken".)
  void publishLink()
  {
    std_msgs::msg::Bool msg;
    msg.data = (fd_ >= 0);
    link_pub_->publish(msg);
  }

  void closeLink(const char * why)
  {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
      RCLCPP_WARN(get_logger(), "BT        link down (%s)", why);
      publishLink();
    }
  }

  void poll()
  {
    if (fd_ < 0) {
      const auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration<double>(now - last_open_try_).count() < reopen_period_s_) {
        return;
      }
      last_open_try_ = now;
      openLink();
      if (fd_ < 0) {return;}
    }

    char buf[256];
    while (fd_ >= 0) {
      const ssize_t n = ::read(fd_, buf, sizeof(buf));
      if (n > 0) {
        rx_buf_.append(buf, static_cast<size_t>(n));
        drainLines();
      } else if (n == 0) {
        closeLink("peer closed");
      } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      } else if (errno == EINTR) {
        continue;
      } else {
        closeLink(std::strerror(errno));
      }
    }
    flushTx();
  }

  void drainLines()
  {
    size_t nl;
    while ((nl = rx_buf_.find('\n')) != std::string::npos) {
      const std::string line = rx_buf_.substr(0, nl);
      rx_buf_.erase(0, nl + 1);
      handleLine(line);
    }
    // No terminator ever arriving would otherwise grow this without bound.
    if (rx_buf_.size() > 4096) {
      RCLCPP_WARN(get_logger(), "BT        discarding %zu bytes with no newline", rx_buf_.size());
      rx_buf_.clear();
    }
  }

  void queueLine(const std::string & line)
  {
    if (fd_ < 0) {return;}
    tx_buf_ += line;
    tx_buf_ += '\n';
  }

  void flushTx()
  {
    while (fd_ >= 0 && !tx_buf_.empty()) {
      const ssize_t n = ::write(fd_, tx_buf_.data(), tx_buf_.size());
      if (n > 0) {
        tx_buf_.erase(0, static_cast<size_t>(n));
      } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        return;
      } else if (n < 0 && errno == EINTR) {
        continue;
      } else {
        closeLink(std::strerror(errno));
      }
    }
  }

  /// The latest ROBOT / PLAN / RESET / STATUS line, so a tablet that connects
  /// (or reconnects) mid-session catches up without waiting for a change.
  void resendState()
  {
    for (const std::string * line : {&robot_line_, &plan_line_, &reset_line_, &status_line_}) {
      if (!line->empty()) {queueLine(*line);}
    }
  }

  // ------------------------------------------------------- ROS -> tablet ----

  void onTx(const std::string & data)
  {
    const std::string line = trim(data);
    if (line.empty()) {return;}
    // State lines are remembered, and a repeat of the current value is not
    // forwarded: the runner repeats them periodically so this node has them
    // even if it started later, but the tablet should only see changes.
    std::string * slot = nullptr;
    if (line.rfind("ROBOT,", 0) == 0) {
      slot = &robot_line_;
    } else if (line.rfind("PLAN:", 0) == 0) {
      slot = &plan_line_;
    } else if (line.rfind("RESET:", 0) == 0) {
      slot = &reset_line_;
    } else if (line.rfind("STATUS:", 0) == 0) {
      slot = &status_line_;
    }
    if (slot != nullptr) {
      if (*slot == line) {return;}
      *slot = line;
    }
    queueLine(line);
    flushTx();
  }

  // ------------------------------------------------------- tablet -> ROS ----

  void handleLine(const std::string & raw)
  {
    const std::string line = trim(raw);
    if (line.empty()) {return;}

    {
      std_msgs::msg::String rx;
      rx.data = line;
      rx_pub_->publish(rx);
    }

    if (!got_rx_since_open_) {
      // The device can open before the tablet has actually connected, in which
      // case the link-up resend went nowhere.
      got_rx_since_open_ = true;
      resendState();
    }

    const std::vector<std::string> f = splitTrimmed(line, ',');
    const std::string cmd = upper(f[0]);
    const std::string manual = lower(f[0]);

    if (cmd == "OBSTACLE") {
      handleObstacle(f);
    } else if (cmd == "DONE" && f.size() == 1) {
      publishSetup();
    } else if (cmd == "BEGIN" && f.size() == 1) {
      callTrigger(start_client_, "/start_run");
    } else if (cmd == "STOP" && f.size() == 1) {
      callTrigger(stop_client_, "/stop_run");
    } else if (cmd == "RESET" && f.size() == 1) {
      callTrigger(reset_client_, "/reset_pose");
    } else if (cmd == "CLEAR" && f.size() == 1) {
      if (!robot_line_.empty()) {
        queueLine(robot_line_);
        flushTx();
      }
    } else if (f.size() == 1 &&
      (manual == "f" || manual == "b" || manual == "fl" || manual == "fr" ||
      manual == "bl" || manual == "br"))
    {
      std_msgs::msg::String out;
      out.data = manual;
      manual_pub_->publish(out);
    } else {
      RCLCPP_WARN(get_logger(), "BT        unrecognised line from the tablet: '%s'", line.c_str());
    }
  }

  void handleObstacle(const std::vector<std::string> & f)
  {
    if (f.size() != 5) {
      RCLCPP_WARN(get_logger(), "BT        OBSTACLE needs 4 fields (n,col,row,facing), got %zu", f.size() - 1);
      return;
    }
    const auto n = parseInt(f[1]);
    const auto x = parseInt(f[2]);
    const auto y = parseInt(f[3]);
    const auto facing = parseFacing(f[4]);
    if (!n || !x || !y || !facing) {
      RCLCPP_WARN(get_logger(), "BT        bad OBSTACLE line (n=%s col=%s row=%s facing=%s)",
        f[1].c_str(), f[2].c_str(), f[3].c_str(), f[4].c_str());
      return;
    }
    if (*x < 0 || *x >= kCells || *y < 0 || *y >= kCells) {
      RCLCPP_WARN(get_logger(), "BT        OBSTACLE %d at cell (%d,%d) is outside the %dx%d grid - ignored",
        *n, *x, *y, kCells, kCells);
      return;
    }

    if (set_closed_) {
      obstacles_.clear();   // first obstacle after a DONE begins a new set
      set_closed_ = false;
    }
    const Obstacle ob{*n, *x, *y, *facing};
    auto it = std::find_if(obstacles_.begin(), obstacles_.end(),
        [&](const Obstacle & o) {return o.n == ob.n;});
    if (it != obstacles_.end()) {
      *it = ob;
    } else {
      obstacles_.push_back(ob);
    }
  }

  void publishSetup()
  {
    if (obstacles_.empty()) {
      RCLCPP_WARN(get_logger(), "BT        DONE with no obstacles - ignored");
      return;
    }
    // task1_runner's format: "id:x_m,y_m,facing|..." in metres, at the centre
    // of the tablet cell (the runner draws and plans the block centred there).
    std::string out;
    for (const auto & o : obstacles_) {
      char item[64];
      std::snprintf(item, sizeof(item), "%d:%.2f,%.2f,%c",
        o.n, (o.col + 0.5) * kCellCm / 100.0, (o.row + 0.5) * kCellCm / 100.0, o.facing);
      if (!out.empty()) {out += '|';}
      out += item;
    }
    std_msgs::msg::String msg;
    msg.data = out;
    setup_pub_->publish(msg);
    set_closed_ = true;   // task1_runner logs the set (OBSTACLES ...)
  }

  void callTrigger(
    const rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr & client, const char * name)
  {
    if (!client->service_is_ready()) {
      RCLCPP_WARN(get_logger(), "BT        %s not available - is a task runner up?", name);
      return;
    }
    // The runner logs the outcome (GO / `go` rejected / STOP / RESET done).
    client->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
  }

  std::string device_;
  double reopen_period_s_{1.0};
  int fd_{-1};
  std::chrono::steady_clock::time_point last_open_try_{};
  std::string rx_buf_;
  std::string tx_buf_;
  bool got_rx_since_open_{false};
  std::string last_open_error_;   // last open() failure already logged

  std::vector<Obstacle> obstacles_;
  bool set_closed_{false};

  std::string robot_line_, plan_line_, reset_line_, status_line_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr setup_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr manual_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr rx_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr link_pub_;
  rclcpp::TimerBase::SharedPtr link_timer_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr tx_sub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr start_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stop_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_client_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace mdp_bridge

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<mdp_bridge::BluetoothBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
