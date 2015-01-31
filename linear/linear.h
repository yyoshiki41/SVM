#include <vector>

using namespace std;

//双対問題クラス
class DualProblem {
public:
	DualProblem() {
		xSize = x.size();
		for (int i = 0 ; i < xSize; i++) {
			a.push_back(-1);
		}
		aSize = a.size();
		g1Size = 1;
		g2Size = a.size();
	}
	//メソッド
	//最小化問題
	double f();
	//最小化問題fをaで偏微分したもの
	vector<double> fDasha();

	//制約1
	vector<double> g1();
	//制約g1をaで偏微分したもの
	vector< vector<double> > gradg1a();

	//制約3
	vector<double> g2();
	//制約g2をaで偏微分したもの
	vector< vector<double> > gradg2a();

	//カーネル
	double K(int i, int j);

	//変数
	//特徴ベクトル
	static const vector<vector<double>> x;
	//教師データ
	static const vector<int> y;
	//ラグランジュ未定乗数
	vector<double> a;

	//変数のサイズ
	int xSize;
	int aSize;
	int g1Size;
	int g2Size;
};
