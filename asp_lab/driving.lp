#include <incmode>.

#program base.

% problem entities
distance(-8..8).
dir(n,s,e,w).
coord(0..8).

% possible starting positions
start(8,3).
start(8,4).
start(8,5).
start(0,3).
start(0,4).
start(0,5).
start(3,0).
start(4,0).
start(5,0).
start(3,8).
start(4,8).
start(5,8).


time(1..10).    

% Goal cell
goal_pos(4,4).

% Obstacles
obs_pos(1, 4).
obs_pos(3, 3).
obs_pos(4, 1).
obs_pos(5, 4).
obs_pos(4, 5).
obs_pos(5, 7).
obs_pos(7, 5).

% Unavailable cells
forbidden_cell(0,0).
forbidden_cell(0,1).
forbidden_cell(1,0).
forbidden_cell(1,1).
forbidden_cell(0,7).
forbidden_cell(0,8).
forbidden_cell(1,7).
forbidden_cell(1,8).
forbidden_cell(7,0).
forbidden_cell(8,0).
forbidden_cell(7,1).
forbidden_cell(8,1).
forbidden_cell(7,7).
forbidden_cell(7,8).
forbidden_cell(8,7).
forbidden_cell(8,8).

% Sidewalks
sidewalk(0,2).
sidewalk(0,6).
sidewalk(1,2).
sidewalk(1,6).
sidewalk(2,0).
sidewalk(2,1).
sidewalk(2,2).
sidewalk(2,6).
sidewalk(2,7).
sidewalk(2,8).
sidewalk(6,0).
sidewalk(6,1).
sidewalk(6,2).
sidewalk(6,6).
sidewalk(6,7).
sidewalk(6,8).
sidewalk(7,2).
sidewalk(7,6).
sidewalk(8,2).
sidewalk(8,6).

crossroad(2,3).
crossroad(2,4).
crossroad(2,5).
crossroad(6,3).
crossroad(6,4).
crossroad(6,5).
crossroad(3,2).
crossroad(4,2).
crossroad(5,2).
crossroad(3,6).
crossroad(4,6).
crossroad(5,6).

% pedestrians moving in the environment
ped_pos(0, 2, 1).
ped_pos(1, 2, 2).
ped_pos(1, 3, 3).
ped_pos(1, 4, 4).
ped_pos(1, 5, 5).
ped_pos(1, 6, 6).
ped_pos(2, 6, 7).
ped_pos(3, 6, 8).
ped_pos(4, 6, 9).
ped_pos(5, 6, 10).

ped_pos(0, 6, 1).
ped_pos(0, 7, 2).
ped_pos(0, 8, 3).
ped_pos(1, 8, 4).
ped_pos(2, 8, 5).
ped_pos(3, 8, 6).
ped_pos(4, 8, 7).
ped_pos(5, 8, 8).
ped_pos(6, 8, 9).
ped_pos(7, 8, 10).


% random starting position
1 { agent_pos(X, Y, 1) : coord(X), coord(Y), start(X, Y) } 1.

#program step(t).
%TO BE COMPLETED (action preconditions as aggregates, effects and constraints...)
%possible actions are move_slow(dir), move_fast(dir), moving the agent in direction dir of 1 or 2 cells respectively.

% HINT: use weak constraints to encourage the agent to preserve the safety of pedestrians and avoid obstacles!

at_goal(t) :- goal_pos(X,Y), agent_pos(X,Y,t).

#program check(t).
%:- query(t), t<3. %for debug purposes
:- not at_goal(t), query(t).

% output
#show agent_pos/3.
#show move_fast/2.
#show move_slow/2.