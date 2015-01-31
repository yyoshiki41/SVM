#include <iostream>
#include <cmath>
#include "linear.h"

class SUMT;
class FSDM;

// 最急降下法
class FSDM {
public:
	//メソッド
	FSDM() {}
	FSDM(SUMT* s) {
		sumt = s;
	}
	//実行する
	void run(const double& r1, const double& r3, const double& epsilon, DualProblem& dpk);
	void update(DualProblem& dpk);
	bool check(DualProblem& dpk);

	//変数
	//ステップ幅
	double t = 0.01;
	//結果の精度
	double epsilon = 0.01;
	//SUMT
	SUMT* sumt;
};

// SUMTクラス
class SUMT {
public:
	//メソッド
	SUMT() {
		FSDM f(this);
		fsdm = f;
	}
	//実行
	void run();
	bool check();

	//ペナルティ項を計算する
	double P1(DualProblem& dpk);
	double P2(DualProblem& dpk);
	//ペナルティ項をaで偏微分する
	vector<double> gradP1a(DualProblem& dpk);
	vector<double> gradP2a(DualProblem& dpk);

	//SUMTの最小化問題
	double theta(DualProblem& dpk);
	//theta()をaで偏微分したもの
	vector<double> gradThetaa(DualProblem& dpk);

	//結果を表示する
	void printResult();

	// 変数
	// ε > 0
	double epsilon = 0.01;
	// ペナルティ項
	double r1 = 1.1;
	double r3 = 2;
	// 双対問題クラス
	DualProblem dp;
	// １ターン前の双対問題クラス
	DualProblem dp1;
	// 最適化問題を解くクラス
	FSDM fsdm;
};

// f(a) = Σa - 1/2ΣΣaayyxx
double DualProblem::f() {
	//結果を格納する変数
	double r = 0;

	//Σaを計算する
	double suma = 0;
	for (int i = 0; i < aSize; i++) {
		suma += a[i];
	}

	//ΣΣaayyxxを計算する
	double sumSumaayyxx = 0;
	for ( int i = 0; i < aSize; i++ ){
		for (int j = 0; j < aSize; j++) {
			sumSumaayyxx += a[i]*a[j]*y[i]*y[j]*K(i, j);
		}
	}

	r = suma - 1.0 / 2.0 * sumSumaayyxx;

	//最小化問題なので-をつける
	return -r;
}

// f'(a) = 1 - Σayyxx
vector<double> DualProblem::fDasha() {
	//結果を格納する変数
	vector<double> r(aSize);

	// f'(w)を計算する
	for (int i = 0; i < aSize; i++) {
		double sumayyxx = 0;
		for (int j = 0; j < aSize; j++) {
			sumayyxx += a[j] * y[i] * y[j] * K(i, j);
		}
		// 最小化問題なので-をつける
		r[i] = -(1 - sumayyxx);
	}

	return r;
}


// 制約条件g1: g1(a) = Σay = 0
vector<double> DualProblem::g1() {
	//結果を格納する変数
	vector<double> r(g1Size);

	// g(a)を計算する
	for (int i = 0; i < aSize; i++) {
		r[0] += a[i] * y[i];
	}

	return r;
}

// g1'(a) = y
vector<vector<double>> DualProblem::gradg1a() {
	//結果を格納する変数
	vector<vector<double>> r(aSize);

	// g'(a)を計算する
	for (int i = 0; i < aSize; i++) {
		r[i].resize(g1Size);
		for (int j = 0; j < g1Size; j++) {
			r[i][j] = y[i];
		}
	}

	return r;
}

// 制約条件g2: g2(a) = -a <= 0
vector<double> DualProblem::g2() {
	//結果を格納する変数
	vector<double> r(g2Size);

	// g(a)を計算する
	for (int i = 0; i < g2Size; i++) {
		r[i] = -a[i];
	}

	return r;
}

// g2'(a) = -1, 0
vector<vector<double>> DualProblem::gradg2a() {
	// 結果を格納する変数
	vector<vector<double>> r(aSize);

	// g'(a)を計算する
	for (int i = 0; i < aSize; i++) {
		r[i].resize(g2Size);
		for (int j = 0; j < g2Size; j++) {
			if (i == j) {
				r[i][j] = -1;
			} else {
				r[i][j] = 0;
			}
		}
	}

	return r;
}


// 線形カーネル
double DualProblem::K(int i, int j) {
	// 結果を格納する変数
	double r = 0;
	for (int k = 0; k < x[i].size(); k++) {
		r += x[i][k] * x[j][k];
	}
	return r;
}


// ペナルティ項 P1(a) = (Σay)^2
double SUMT::P1(DualProblem& dpk) {
	//結果を格納する変数
	double r = 0;
	vector<double> g1 = dpk.g1();

	for (int i = 0; i < dpk.g1Size; i++) {
		r += g1[i] * g1[i];
	}
	return r;
}

// ペナルティ項 P2(a) = Σ{[min(a, 0)]^2}
double SUMT::P2(DualProblem& dpk) {
	//結果を格納する変数
	double r = 0;
	vector<double> g2 = dpk.g2();

	for (int i = 0; i < dpk.g2Size; i++) {
		if (g2[i] <= 0) {
			r += 0;
		} else {
			r += g2[i] * g2[i];
		}
	}

	return r;
}

// P1'(a) = 2 * g1 * g1'
vector<double> SUMT::gradP1a(DualProblem& dpk) {
	//結果を格納する変数
	vector<double> r(dpk.aSize);
	//g1(a)
	vector<double> g1 = dpk.g1();
	//g1'(a)
	vector<vector<double>> gg1a = dpk.gradg1a();

	//P1'(w)を計算する
	for (int i = 0; i < dpk.aSize; i++) {
		//制約1の分
		for (int j = 0; j < dpk.g1Size; j++) {
			r[i] += 2 * g1[j] * gg1a[i][j];
		}
	}

	return r;
}
// P2'(a) = 2 * g2 * g2'
vector<double> SUMT::gradP2a(DualProblem& dpk) {
	//結果を格納する変数
	vector<double> r(dpk.aSize);
	//g2(a)
	vector<double> g2 = dpk.g2();
	//g2'(a)
	vector<vector<double>> gg2a = dpk.gradg2a();

	//P2'(w)を計算する
	for (int i = 0; i < dpk.aSize; i++) {
		for (int j = 0; j < dpk.g2Size; j++) {
			if (g2[j] <= 0) {
				r[i] += 0;
			} else {
				r[i] += 2 * g2[j] * gg2a[i][j];
			}
		}
	}

	return r;
}

// 最小化する関数
// θ(a) = f(a) + r1 * P1(a) + r3 * P2(a)
double SUMT::theta(DualProblem& dpk) {
	return dpk.f() + r1 * P1(dpk) + r3 * P2(dpk);
}

// θ'(a) = f'(a) + r1 * P1'(a) + r3 * P2'(a)
vector<double> SUMT::gradThetaa(DualProblem& dpk) {
	// 結果を格納する変数
	vector<double> r(dpk.aSize);

	for (int i = 0; i < dpk.aSize; i++) {
		r[i] = dpk.fDasha()[i] + r1 * gradP1a(dpk)[i] + r3 * gradP2a(dpk)[i];
	}

	return r;
}

void FSDM::update(DualProblem& dpk) {
	vector<double> deltaa = sumt->gradThetaa(dpk);
	for (int i = 0; i < dpk.aSize; i++) {
		dpk.a[i] = dpk.a[i] - t * deltaa[i];
	}
}

bool FSDM::check(DualProblem& dpk) {
	vector<double> deltaa = sumt->gradThetaa(dpk);

	double norm = 0;
	for (int i = 0; i < deltaa.size(); i++) {
		norm += (deltaa[i]) * (deltaa[i]);
	}
	norm = sqrt(norm);

	//距離がepsilonより大きければ続ける
	if (norm > epsilon) {
		return true;
	} else {
		return false;
	}
}

// 最急降下法もどき
void FSDM::run(const double& r1, const double& r3, const double& epsilon, DualProblem& dpk) {
	// 初期地点
	DualProblem dpkk = dpk;

	// 最適解と最小値を求める
	do {
		update(dpkk);
	} while (check(dpkk));

	// 結果を代入する
	dpk = dpkk;
}

bool SUMT::check() {
	//距離を計算する
	double r = 0;
	for (int i = 0; i < dp.aSize; i++) {
		r += (dp.a[i] - dp1.a[i]) * (dp.a[i] - dp1.a[i]);
	}

	if (sqrt(r) > epsilon) {
		return true;
	} else {
		return false;
	}
}

// SUMT
void SUMT::run() {
	// Iterative Step
	do {
		dp1 = dp;
		// 無制約最適化問題を解く
		fsdm.run(r1, r3, epsilon, dp);
	} while (check());

	// a <= 0 なら0とする
	for (int i = 0; i < dp.aSize; i++) {
		if (dp.a[i] <= 0) {
			dp.a[i] = 0;
		}
	}

	printResult();
}

void SUMT::printResult() {
	cout << "最適解：";
	for (int i = 0; i < dp.aSize; i++) {
		cout << "a" << i << "=" << dp.a[i] << " ";
	}
	cout << "\n最小値：" << dp.f() << endl;

	for (int i = 0; i < dp.x[0].size(); i++) {
		double w = 0;
		for (int j = 0; j < dp.xSize; j++) {
			w += dp.a[j] * dp.y[j] * dp.x[j][i];
		}
		cout << "w" << i << "=" << w << " ";
	}
	double b = 0;
	vector<double> S;
	for (int i = 0; i < dp.a.size(); i++) {
		if (0 < dp.a[i]) {
			S.push_back(i);
		}
	}

	for (int j = 0; j < S.size(); j++) {
		double sumayxx = 0;
		for (int i = 0; i < S.size(); i++) {
			sumayxx += dp.a[S[i]] * dp.y[S[i]] * dp.K(S[i], S[j]);
		}
		b += dp.y[S[j]] - sumayxx;
	}
	b /= S.size();
	cout << "b=" << b << endl;
}

//メインループ
int main() {
	SUMT sumt;
	sumt.run();

	return 0;
}
