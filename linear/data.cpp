#include "linear.h"

/*
const vector<int> DualProblem::y(
	{1, 1, 1, -1, -1, -1}
);

const vector<vector<double>> DualProblem::x({
	{0,0},
	{1,1},
	{1,2},
	{5,5},
	{5,6},
	{6,6}
});
*/

const vector<int> DualProblem::y(
	{1, 1, 1, 1, 1, -1, -1, -1, -1, -1}
);

const vector<vector<double>> DualProblem::x({
	{1,2},
	{0,1},
	{-1,2},
	{2,1},
	{3,5},
	{0,-1},
	{-1,-2},
	{-2,0},
	{-4,0},
	{-3,-3}
});
