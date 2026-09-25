# task1_runner's live state on /run_status, 2 Hz, every state.
# Plot any number in Foxglove (e.g. /run_status.dist_to_checkpoint).
# Positions are in the arena frame (map), metres / radians.

std_msgs/Header header

string state          # WAITING_FOR_SETUP PLANNING_PATH WAITING_FOR_GO NAVIGATING_TO_TARGET PAUSE_FOR_SCAN FINISHED STOPPED
string plan_state     # WAITING PLANNING DONE (same as the tablet's PLAN line)
bool reset_done       # pose verified at the start (the tablet's RESET line)

# Where it is going
string obstacle       # tablet obstacle number of the current target, '' when none
uint8 leg             # 1-based leg in visit order, 0 when none
uint8 leg_count
float64 checkpoint_x
float64 checkpoint_y
float64 checkpoint_yaw
float64 dist_to_checkpoint   # NaN when there is no current checkpoint

# The car
float64 x
float64 y
float64 yaw

# Path following (NAVIGATING_TO_TARGET)
uint16 wp_index       # 1-based waypoint the follower is chasing, 0 when not following
uint16 wp_count
float64 target_x
float64 target_y
bool reverse          # that waypoint is a reverse-gear segment
float64 cmd_v         # last /cmd_vel sent, m/s
float64 cmd_w         # rad/s

# Scanning (PAUSE_FOR_SCAN)
float64 scan_elapsed
float64 scan_duration
string[] yolo_ids     # every id YOLO reported during this scan...
uint16[] yolo_counts  # ...and how often
string detected_id    # the most frequent - what TARGET will report
