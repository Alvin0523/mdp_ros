#include <gtest/gtest.h>

#include <map>
#include <string>

#include "mdp_bridge/bluetooth_protocol.hpp"

using namespace mdp_bridge::bt;  // NOLINT

TEST(Obstacle, TabletValuesAreCellTimesTen)
{
  ObstacleLine o;
  ASSERT_TRUE(parseObstacle("OBSTACLE,2,0,10,N", o));
  EXPECT_EQ(o.n, 2);
  EXPECT_FALSE(o.removed);
  EXPECT_EQ(o.obstacle.cx, 0);
  EXPECT_EQ(o.obstacle.cy, 1);
}

TEST(Obstacle, TopRightCellAndFullWordFacing)
{
  ObstacleLine o;
  ASSERT_TRUE(parseObstacle("OBSTACLE, 3, 190, 190, west", o));
  EXPECT_EQ(o.obstacle.facing, 'W');
  EXPECT_EQ(o.obstacle.cx, 19);
}

TEST(Obstacle, MinusOneFacingMeansRemoved)
{
  ObstacleLine o;
  ASSERT_TRUE(parseObstacle("OBSTACLE,3,10,20,-1", o));
  EXPECT_TRUE(o.removed);
}

TEST(Obstacle, MalformedLinesRejected)
{
  ObstacleLine o;
  EXPECT_FALSE(parseObstacle("OBSTACLE,3,10,20,X", o));
  EXPECT_FALSE(parseObstacle("OBSTACLE,a,10,20,N", o));
  EXPECT_FALSE(parseObstacle("OBSTACLE,1,2", o));
  EXPECT_FALSE(parseObstacle("OBSTACLE,1,200,0,N", o));  /* cell 20: off the grid */
  EXPECT_FALSE(parseObstacle("OBSTACLE,1,-10,0,N", o));
  EXPECT_FALSE(parseObstacle("BEGIN", o));
}

TEST(Setup, SortedByTabletNumber)
{
  std::map<int, Obstacle> m;
  m[3] = {9, 8, 'W'};
  m[1] = {0, 1, 'N'};
  EXPECT_EQ(formatSetup(m), "1:0,1,N|3:9,8,W");
}

TEST(Robot, CellsAndCompass)
{
  EXPECT_EQ(convertRobot("ROBOT,0.150,0.150,90.0"), "ROBOT,0,0,N");
  EXPECT_EQ(convertRobot("ROBOT,0.300,0.200,0.0"), "ROBOT,2,1,E");
  EXPECT_EQ(convertRobot("ROBOT,1.95,1.95,-90"), "ROBOT,18,18,S");
  EXPECT_EQ(convertRobot("ROBOT,0.3,0.3,359"), "ROBOT,2,2,E");
  EXPECT_EQ(convertRobot("ROBOT,0.3,0.3,180"), "ROBOT,2,2,W");
}

TEST(Robot, ClampedAndMalformed)
{
  EXPECT_EQ(convertRobot("ROBOT,-0.5,3.0,90"), "ROBOT,0,18,N");
  EXPECT_EQ(convertRobot("ROBOT,x,1,1"), "");
  EXPECT_EQ(convertRobot("STATUS:x"), "");
}

TEST(Drive, ButtonsAndBusyStates)
{
  int lin = 0, ang = 0;
  ASSERT_TRUE(driveButton("fl", lin, ang));
  EXPECT_EQ(lin, 1);
  EXPECT_EQ(ang, 1);
  ASSERT_TRUE(driveButton("BR", lin, ang));
  EXPECT_EQ(lin, -1);
  EXPECT_EQ(ang, -1);
  EXPECT_FALSE(driveButton("go", lin, ang));
  EXPECT_TRUE(runnerBusy("STATUS:Going to obstacle 1"));
  EXPECT_TRUE(runnerBusy("STATUS:Stopped"));
  EXPECT_FALSE(runnerBusy("STATUS:Ready"));
  EXPECT_FALSE(runnerBusy(""));
}
