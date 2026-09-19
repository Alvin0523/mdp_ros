/**
 * @file bluetooth_bridge_node.cpp
 * @brief Placeholder for the Android tablet's Bluetooth RFCOMM link.
 *
 * Not yet implemented - see docs/android/index.md for the planned interface
 * contract (app -> robot movement/obstacle commands, robot -> app TARGET/ROBOT
 * status strings over /dev/rfcommN). This node currently does nothing but spin.
 */

#include "rclcpp/rclcpp.hpp"

namespace mdp_bridge
{

class BluetoothBridgeNode : public rclcpp::Node
{
public:
  BluetoothBridgeNode()
  : Node("bluetooth_bridge_node")
  {
    RCLCPP_WARN(get_logger(), "bluetooth_bridge_node is a placeholder - not yet implemented");
  }
};

}  // namespace mdp_bridge

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<mdp_bridge::BluetoothBridgeNode>());
  rclcpp::shutdown();
  return 0;
}
