#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class Solver;
class DualProblem;
class SteepestDescent;

// Dual problem
class DualProblem {
public:
	DualProblem() {
		xSize = x.size();
		for (int i = 0 ; i < xSize; i++) {
			a.push_back(-1);
		}
	}

	// Quadratic programming
	double QP();
	// Differential function
	vector<double> dQP();

	// Constraint condition
	vector<double> h();
	vector<double> g();
	// Differential function
	vector<vector<double>> dh();
	vector<vector<double>> dg();

	// Kernel function
	double K(int i, int j);

	// Lagrange multiplier
	vector<double> a;

	// Training set
	static const vector<vector<double>> x;
	static const vector<int> y;

	// Size of variable
	int xSize;
	int hSize = 1;
};

// Examples
const vector<vector<double>> DualProblem::x({ {0,0}, {1,1}, {1,2}, {5,5}, {5,6}, {6,6} });
// Labels
const vector<int> DualProblem::y({ 1, 1, 1, -1, -1, -1 });

// Steepest Descent Method
class SteepestDescent {
public:
	SteepestDescent() {}
	SteepestDescent(Solver* s) {
		solv = s;
	}
	// Execute
	void run(const double& r1, const double& r2, const double& epsilon, DualProblem& dpk);
	bool check(DualProblem& dpk);

	// Step width
	double t = 0.01;
	// ε > 0
	double epsilon = 0.01;

	Solver* solv;
};

class Solver {
public:
	Solver() {
		SteepestDescent f(this);
		sdm = f;
	}
	// Execute
	void run();
	bool check();

	// Penalty
	double P1(DualProblem& dpk);
	double P2(DualProblem& dpk);
	// Differential function
	vector<double> dP1(DualProblem& dpk);
	vector<double> dP2(DualProblem& dpk);

	// Unconstrained optimization problem
	double E(DualProblem& dpk);
	// Differential function
	vector<double> dE(DualProblem& dpk);

	// Results
	void calResults();
	void priResults();
	// weight vector
	vector<double> w;
	// bias
	double b;

	// ε > 0
	double epsilon = 0.01;
	// Penalty factor
	double r1 = 1.1;
	double r2 = 2;

	// Dual problem
	DualProblem dp;
	DualProblem dp1; // 1 iteration before

	// Unconstrained optimization technique
	SteepestDescent sdm;
};

// QP(a) = Σa - 1/2 ΣΣaayyxx
double DualProblem::QP() {
	double suma = 0;
	for (int i = 0; i < xSize; i++) {
		suma += a[i];
	}

	double sumSumaayyxx = 0;
	for ( int i = 0; i < xSize; i++ ){
		for (int j = 0; j < xSize; j++) {
			sumSumaayyxx += a[i] * a[j] * y[i] * y[j] * K(i, j);
		}
	}

	double r = suma - 1.0 / 2.0 * sumSumaayyxx;

	// Min problem, so puts -(Minus) sign
	return -r;
}
// QP'(a) = 1 - Σayyxx
vector<double> DualProblem::dQP() {
	vector<double> r(xSize);

	for (int i = 0; i < xSize; i++) {
		double sumayyxx = 0;
		for (int j = 0; j < xSize; j++) {
			sumayyxx += a[j] * y[i] * y[j] * K(i, j);
		}
		// Min problem, so puts -(Minus) sign
		r[i] = -(1 - sumayyxx);
	}

	return r;
}


/**
 * Equality Constraints
 * h(a) = Σay = 0
 */
vector<double> DualProblem::h() {
	vector<double> r(hSize);

	for (int i = 0; i < hSize; i++) {
		for (int j = 0; j < xSize; j++) {
			r[i] += a[j] * y[j];
		}
	}

	return r;
}
// h'(a) = y
vector<vector<double>> DualProblem::dh() {
	vector<vector<double>> r(xSize);

	for (int i = 0; i < xSize; i++) {
		r[i].resize(hSize);
		for (int j = 0; j < hSize; j++) {
			r[i][j] = y[i];
		}
	}

	return r;
}

/**
 * Inequality Constraints
 * g(a) = -a <= 0
 */
vector<double> DualProblem::g() {
	vector<double> r(xSize);

	for (int i = 0; i < xSize; i++) {
		r[i] = -a[i];
	}

	return r;
}
// g'(a) = -1, 0
vector<vector<double>> DualProblem::dg() {
	vector<vector<double>> r(xSize);

	for (int i = 0; i < xSize; i++) {
		r[i].resize(xSize);
		for (int j = 0; j < xSize; j++) {
			if (i == j) {
				r[i][j] = -1;
			} else {
				r[i][j] = 0;
			}
		}
	}

	return r;
}

// Linear kernel
double DualProblem::K(int i, int j) {
	double r = 0;

	for (int k = 0; k < x[i].size(); k++) {
		r += x[i][k] * x[j][k];
	}
	return r;
}


/**
 * Penalty term (Equality Constraints)
 * P1(a) = (Σay)^2
 */
double Solver::P1(DualProblem& dpk) {
	double r = 0;

	vector<double> h = dpk.h();
	for (int i = 0; i < dpk.hSize; i++) {
		r += h[i] * h[i];
	}
	return r;
}
// P1'(a) = 2 * h(a) * h'(a)
vector<double> Solver::dP1(DualProblem& dpk) {
	vector<double> r(dpk.xSize);

	vector<double> h = dpk.h();
	vector<vector<double>> dh = dpk.dh();

	for (int i = 0; i < dpk.xSize; i++) {
		for (int j = 0; j < dpk.hSize; j++) {
			r[i] += 2 * h[j] * dh[i][j];
		}
	}
	return r;
}

/**
 * Penalty term (Inequality Constraints)
 * P2(a) = Σ{[min(a, 0)]^2}
 */
double Solver::P2(DualProblem& dpk) {
	double r = 0;

	vector<double> g = dpk.g();
	for (int i = 0; i < dpk.xSize; i++) {
		if (g[i] <= 0) {
			r += 0;
		} else {
			r += g[i] * g[i];
		}
	}
	return r;
}
// P2'(a) = 2 * g * g'
vector<double> Solver::dP2(DualProblem& dpk) {
	vector<double> r(dpk.xSize);

	vector<double> g = dpk.g();
	vector<vector<double>> dg = dpk.dg();

	for (int i = 0; i < dpk.xSize; i++) {
		for (int j = 0; j < dpk.xSize; j++) {
			if (g[j] <= 0) {
				r[i] += 0;
			} else {
				r[i] += 2 * g[j] * dg[i][j];
			}
		}
	}
	return r;
}

/**
 * Expansion objective function of the penalty method
 * E(a) = QP(a) + r1 * P1(a) + r2 * P2(a)
 */
double Solver::E(DualProblem& dpk) {
	return dpk.QP() + r1 * P1(dpk) + r2 * P2(dpk);
}
// E'(a) = QP'(a) + r1 * P1'(a) + r2 * P2'(a)
vector<double> Solver::dE(DualProblem& dpk) {
	vector<double> r(dpk.xSize);

	for (int i = 0; i < dpk.xSize; i++) {
		r[i] = dpk.dQP()[i] + r1 * dP1(dpk)[i] + r2 * dP2(dpk)[i];
	}
	return r;
}

bool SteepestDescent::check(DualProblem& dpk) {
	vector<double> deltaa = solv->dE(dpk);

	double norm = 0;
	for (int i = 0; i < deltaa.size(); i++) {
		norm += (deltaa[i]) * (deltaa[i]);
	}
	norm = sqrt(norm);

	if (norm > epsilon) {
		return true;
	} else {
		return false;
	}
}

void SteepestDescent::run(const double& r1, const double& r2, const double& epsilon, DualProblem& dpk) {
	// Initial points
	DualProblem dpkk = dpk;

	do {
		vector<double> deltaa = solv->dE(dpkk);
		for (int i = 0; i < dpkk.xSize; i++) {
			dpkk.a[i] = dpkk.a[i] - t * deltaa[i];
		}
	} while (check(dpkk));

	dpk = dpkk;
}


bool Solver::check() {
	double r = 0;
	for (int i = 0; i < dp.xSize; i++) {
		r += (dp.a[i] - dp1.a[i]) * (dp.a[i] - dp1.a[i]);
	}

	if (sqrt(r) > epsilon) {
		return true;
	} else {
		return false;
	}
}

void Solver::run() {
	do {
		dp1 = dp;
		// Solve the unconstrained optimization problem
		sdm.run(r1, r2, epsilon, dp);
	} while (check());

	// a = 0, if a < 0
	for (int i = 0; i < dp.xSize; i++) {
		if (dp.a[i] < 0) {
			dp.a[i] = 0;
		}
	}

	calResults();
	priResults();
}

void Solver::calResults() {
	w.resize(dp.x[0].size());
	for (int i = 0; i < w.size(); i++) {
		for (int j = 0; j < dp.xSize; j++) {
			w[i] += dp.a[j] * dp.y[j] * dp.x[j][i];
		}
	}

	vector<double> S;
	for (int i = 0; i < dp.a.size(); i++) {
		if (0 < dp.a[i]) {
			S.push_back(i);
		}
	}
	b = 0;
	for (int j = 0; j < S.size(); j++) {
		double sumayxx = 0;
		for (int i = 0; i < S.size(); i++) {
			sumayxx += dp.a[S[i]] * dp.y[S[i]] * dp.K(S[i], S[j]);
		}
		b += dp.y[S[j]] - sumayxx;
	}
	b = b / S.size();
}

void Solver::priResults() {
	cout << "ラグランジェ乗数：\t";
	for (int i = 0; i < dp.xSize; i++) {
		cout << "a" << i << "=" << dp.a[i] << " ";
	}
	cout << endl << "2次計画問題の最小値：\t" << "f(λ) = " << dp.QP() << endl;

	cout << "主問題の最適解：\t";
	for (int i = 0; i < w.size(); i++) {
		cout << "w" << i << "=" << w[i] << ", ";
	}
	cout << "b=" << b << endl;
}

int main() {
	Solver solv;
	solv.run();

	return 0;
}
