// libsvm
#include "svm.h"

// その他STLとか
#include <ctime>
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
//#include <random>
#include <math.h>
#include <utility>


    // サンプルデータを表すクラス
    // 二次元ベクトル：ラベル（ノードが属するクラスのカテゴリ）を持つ
class node
{
public : 
    node( int label, double x, double y )
        : label_( label ), x_( x ), y_( y )
    {
    }
        // member variables
    int label_;
    double x_, y_;
};


int main( int, char*[] )
{
    using namespace std;

    // サンプルデータを適当に作る
        // 今回は２クラスで，いずれも正規分布に従います
    std::vector< node > samples;

            // それぞれのクラスで作るサンプルデータの数です
    const auto kClass1NodeNum = 100;
    const auto kClass2NodeNum = 100;

            // それぞれのクラスのラベルです，今回はクラス１は「1」，クラス２は「-1」としました
    const auto kClass1Label = 1;
    const auto kClass2Label = -1;

            // それぞれのクラスのx,y軸のMeanとSigmaを入れて，その正規分布に従う乱数生成器を作ります
    auto class1_x_generator = std::normal_distribution<double>( 0.3, 0.2 );
    auto class1_y_generator = std::normal_distribution<double>( 0.3, 0.2 );

    auto class2_x_generator = std::normal_distribution<double>( 0.7, 0.2 );
    auto class2_y_generator = std::normal_distribution<double>( 0.7, 0.2 );

            // 実際にサンプルデータを作ります
    std::mt19937 eng( static_cast<unsigned int>( time( nullptr ) ) );
    for( std::size_t i=0; i<kClass1NodeNum; ++i )
    {
        samples.push_back( node( kClass1Label, class1_x_generator( eng ), class1_y_generator( eng ) ) );
    }
    for( std::size_t i=0; i<kClass2NodeNum; ++i )
    {
        samples.push_back( node( kClass2Label, class2_x_generator( eng ), class2_y_generator( eng ) ) );
    }


    // libsvmに学習してもらいます
        // libsvm用の学習データを作ります
    svm_problem prob;
    svm_node* prob_vec;
            // 学習データの数
    prob.l = samples.size();
            // 各学習データのラベル
    prob.y = new double[ prob.l ];
    for( std::size_t i=0; i<samples.size(); ++i )
    {
        prob.y[i] = samples[i].label_;
    }
            // 各学習データのベクトル
            /*
                libsvmではスパースな特徴量も効率的に扱えるよう，一つの学習データ（ベクトル）はsvm_nodeの配列で表されます
                svm_nodeはベクトル内の一つの軸の値を保持するもので，そのベクトルの軸の番号「index」（1から始まる）と値「value」をメンバに持ちます
                二次元ベクトルの場合，軸はXとYで二つなので，二次元ベクトルを表現するのにsvm_node[2]という配列が必要です
                しかし，これだとベクトルの終端がどこだか分からないため（libsvmではスパースなベクトルも扱えるため），
                実際には最後に”終端”を表すsvm_nodeが必要です（この”終端”を表すsvm_nodeは，indexを「-1」とすることで処理されます）
                よって，二次元ベクトル一つを表現するのに必要なのはsvm_node[3]の配列です
            */
    prob_vec = new svm_node[ prob.l *(2+1) ];// 全てのベクトルが2次元，かつベクトルの句切れに1個必要なので，学習データ＊３のsvm_nodeが必要
    prob.x = new svm_node*[ prob.l ];
    for( std::size_t i=0; i<samples.size(); ++i )
    {
        prob.x[i] = prob_vec+i*3;
        prob.x[i][0].index = 1;
        prob.x[i][0].value = samples[i].x_;
        prob.x[i][1].index = 2;
        prob.x[i][1].value = samples[i].y_;
        prob.x[i][2].index = -1;
    }
        // 学習する識別器のパラメータ
    svm_parameter param;
        param.svm_type = C_SVC;// SVCとかSVRとか
        param.kernel_type = LINEAR;// RBF（放射基底関数）カーネルとかLINEAR（線形）カーネルとかPOLY（多項式）カーネルとか
        param.C = 8096;// SVMにおけるコスト：大きいほどハードマージン
        param.gamma = 0.1;// カーネルとかで使われるパラメータ

            // その他
        param.coef0 = 0;
        param.cache_size = 100;
        param.eps = 1e-3;
        param.shrinking = 1;
        param.probability = 0;

            // その他状況（svm_typeやカーネル）に応じて必要なもの
        param.degree = 3;
        param.nu = 0.5;
        param.p = 0.1;
        param.nr_weight = 0;
        param.weight_label = nullptr;
        param.weight = nullptr;

        // 学習！
    cout << "Ready to train ..." << endl;
    svm_model* model = svm_train( &prob, &param );
    cout << "Finished ..." << endl;


        // predit training samples
    int correct_count =0, wrong_count =0;
    cout << "predict training samples ..." << endl;
    for( std::size_t i=0; i<samples.size(); ++i )
    {
        svm_node test[3];
        test[0].index = 1;
        test[0].value = samples[i].x_;
        test[1].index = 2;
        test[1].value = samples[i].y_;
        test[2].index = -1;
            // libsvmにpredictしてもらう
        const auto kPredictedLabel = static_cast<int>( svm_predict( model, test ) );
            // 結果カウント
        if ( kPredictedLabel == samples[i].label_ )
        {
            ++correct_count;
        }
        else
        {
            ++wrong_count;
        }
    }
    cout << "done" << endl;
    cout << "RESULT : correct=" << correct_count << " : wrong=" << wrong_count << endl;
    cout << "Accuracy[%]=" << ( static_cast<double>(correct_count)/static_cast<double>(correct_count+wrong_count)*100.0 ) << endl;

	// 後始末
    svm_free_and_destroy_model( &model );
    delete[] prob.y;
    delete[] prob.x;
    delete[] prob_vec;

    return 0;
}
