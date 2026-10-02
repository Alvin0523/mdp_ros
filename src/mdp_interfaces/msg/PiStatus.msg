# The Pi's own load on /pi/status, 1 Hz (mdp_bringup pi_status, runs on the Pi).
# With a laptop the monitors run there, so this is how they see the Pi.
# Plot any number in Foxglove (e.g. /pi/status.cpu_percent).

std_msgs/Header header

float32 cpu_percent       # all cores, 0..100, over the last second
float32[] cpu_cores       # per core, 0..100
float32 mem_percent       # used (not available), 0..100
float32 temp_c            # SoC temperature
bool throttled            # slowed down right now (heat or low voltage)
bool throttled_since_boot # it has happened since boot
bool undervoltage         # supply below ~4.63 V right now
