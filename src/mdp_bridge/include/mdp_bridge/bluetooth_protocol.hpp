/**
 * @file bluetooth_protocol.hpp
 * @brief Pure (ROS-free) parsing/formatting for the Android <-> RPi line protocol
 *        spoken by bluetooth_bridge_node. Header-only so it can be unit tested.
 *
 * Wire format: one '\n'-terminated ASCII line per message.
 *
 *   Android -> RPi
 *     OBSTACLE,<n>,<x>,<y>,<N|E|S|W>   x,y = grid cell (0..19) * 10; facing "-1" = removed
 *     DONE  BEGIN  STOP  CLEAR         see bluetooth_bridge_node.cpp
 *     f b fl fr bl br                  manual drive buttons
 *
 *   RPi -> Android
 *     ROBOT,<x>,<y>,<N|E|S|W>   x,y = grid cell (0..18) of the robot's bottom-left
 *                               cell (the robot is 2x2 cells on the tablet map)
 *     TARGET,<n>,<id>  PLAN:<state>  RESET:<state>  STATUS:<text>
 */

#ifndef MDP_BRIDGE__BLUETOOTH_PROTOCOL_HPP_
#define MDP_BRIDGE__BLUETOOTH_PROTOCOL_HPP_

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace mdp_bridge
{
namespace bt
{

constexpr double kCellM = 0.10;   /* one grid cell */
constexpr int kGridCells = 20;
constexpr int kRobotMaxCell = 18; /* 20 cells - the robot's 2-cell footprint */

inline std::string trim(const std::string & s)
{
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {b++;}
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {e--;}
  return s.substr(b, e - b);
}

inline std::vector<std::string> split(const std::string & s, char sep)
{
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    size_t pos = s.find(sep, start);
    if (pos == std::string::npos) {
      out.push_back(trim(s.substr(start)));
      return out;
    }
    out.push_back(trim(s.substr(start, pos - start)));
    start = pos + 1;
  }
}

inline std::string upper(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {return std::toupper(c);});
  return s;
}

inline bool startsWith(const std::string & s, const std::string & prefix)
{
  return s.compare(0, prefix.size(), prefix) == 0;
}

/* Strict number parse: whole string must be consumed. */
inline bool parseDouble(const std::string & s, double & out)
{
  if (s.empty()) {return false;}
  char * end = nullptr;
  out = std::strtod(s.c_str(), &end);
  return end != nullptr && *end == '\0' && std::isfinite(out);
}

struct Obstacle
{
  int cx = 0;  /* grid cell 0..19, origin bottom-left */
  int cy = 0;
  char facing = 'N';
};

struct ObstacleLine
{
  int n = 0;
  bool removed = false;
  Obstacle obstacle;
};

/* 'OBSTACLE,n,x,y,FACING' -> ObstacleLine. The tablet's x/y are the cell index
 * * 10, so cell = x / 10 (0..19; anything else is rejected). Facing accepts
 * N/E/S/W or the full words; "-1" (or empty) means the obstacle was removed.
 * Returns false if malformed. */
inline bool parseObstacle(const std::string & line, ObstacleLine & out)
{
  const auto p = split(line, ',');
  if (p.size() < 5 || upper(p[0]) != "OBSTACLE") {return false;}
  double n = 0, x_cm = 0, y_cm = 0;
  if (!parseDouble(p[1], n) || !parseDouble(p[2], x_cm) || !parseDouble(p[3], y_cm)) {
    return false;
  }
  out.n = static_cast<int>(n);
  const long cx = std::lround(x_cm / 10.0);
  const long cy = std::lround(y_cm / 10.0);
  if (cx < 0 || cx > kGridCells - 1 || cy < 0 || cy > kGridCells - 1) {return false;}
  out.obstacle.cx = static_cast<int>(cx);
  out.obstacle.cy = static_cast<int>(cy);

  const std::string f = upper(p[4]);
  out.removed = f.empty() || f == "-1";
  if (!out.removed) {
    if (f[0] != 'N' && f[0] != 'E' && f[0] != 'S' && f[0] != 'W') {return false;}
    out.obstacle.facing = f[0];
  }
  return true;
}

/* {n: obstacle} -> the runner's '/obstacle_cells' payload 'id:cx,cy,F|...'
 * (grid cells 0..19, sorted by the tablet's obstacle number), e.g. '1:0,1,N|2:9,8,W'. */
inline std::string formatSetup(const std::map<int, Obstacle> & obstacles)
{
  std::string out;
  for (const auto & kv : obstacles) {
    char buf[48];
    std::snprintf(
      buf, sizeof(buf), "%d:%d,%d,%c", kv.first, kv.second.cx, kv.second.cy, kv.second.facing);
    if (!out.empty()) {out.push_back('|');}
    out += buf;
  }
  return out;
}

/* Robot centre coordinate (m) -> cell of the robot's bottom-left corner. */
inline int robotCell(double centre_m)
{
  const int cell = static_cast<int>(std::floor(centre_m / kCellM + 1e-9)) - 1;
  return std::min(std::max(cell, 0), kRobotMaxCell);
}

/* Arena heading (deg, 0 = +x = East, 90 = North) -> nearest N/E/S/W. */
inline char compass(double yaw_deg)
{
  double d = std::fmod(yaw_deg, 360.0);
  if (d < 0) {d += 360.0;}
  return "ENWS"[static_cast<int>(std::lround(d / 90.0)) % 4];
}

/* The runner's 'ROBOT,x_m,y_m,yaw_deg' -> the tablet's 'ROBOT,x,y,D'.
 * Empty string if the line isn't a well-formed runner pose. */
inline std::string convertRobot(const std::string & line)
{
  const auto p = split(line, ',');
  double x = 0, y = 0, yaw = 0;
  if (p.size() != 4 || p[0] != "ROBOT" || !parseDouble(p[1], x) || !parseDouble(p[2], y) ||
    !parseDouble(p[3], yaw))
  {
    return "";
  }
  char buf[48];
  std::snprintf(buf, sizeof(buf), "ROBOT,%d,%d,%c", robotCell(x), robotCell(y), compass(yaw));
  return buf;
}

/* Manual-drive buttons -> (linear sign, angular sign); left turn = +angular. */
inline bool driveButton(const std::string & line, int & lin, int & ang)
{
  const std::string k = upper(line);
  if (k == "F") {lin = 1; ang = 0;} else if (k == "B") {lin = -1; ang = 0;} else if (k == "FL") {
    lin = 1; ang = 1;
  } else if (k == "FR") {lin = 1; ang = -1;} else if (k == "BL") {lin = -1; ang = 1;} else if (k ==
    "BR")
  {
    lin = -1; ang = -1;
  } else {return false;}
  return true;
}

/* True while the runner owns /cmd_vel, so tablet drive buttons must be ignored. */
inline bool runnerBusy(const std::string & status)
{
  return startsWith(status, "STATUS:Going") || startsWith(status, "STATUS:Scanning") ||
         status == "STATUS:Stopped";
}

}  // namespace bt
}  // namespace mdp_bridge

#endif  // MDP_BRIDGE__BLUETOOTH_PROTOCOL_HPP_
