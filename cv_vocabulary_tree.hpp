#ifndef cv_vocabulary_tree_cpp
#define cv_vocabulary_tree_cpp

// vocabulary tree using kmeans cluster

#include <vector>
#include <iostream>
#include <iomanip>

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

using namespace std;
struct cv_vocabulary_tree_parameter
{
    int k_;          // k clusters
    int max_depth_;  // maximum depth of tree
    int nLabel_;     // label number
    int min_leaf_node_;

    cv_vocabulary_tree_parameter()
    {
        k_ = 8;
        max_depth_ = 3;
        min_leaf_node_ = 40;
    }
};

struct distributionData {
    size_t index;
    float singleHistogram;
};

struct by_number {
    bool operator()(distributionData const &left, distributionData const &right) {
        return left.singleHistogram < right.singleHistogram;
    }
};


class cv_vocabulary_tree_node;
class cv_vocabulary_tree
{
    cv_vocabulary_tree_node *root_;
    int dim_;      // feature dimension
    int nLabel_;   // number of labels

public:
    cv_vocabulary_tree();
    ~cv_vocabulary_tree();

    // data CV_32F
    // each row is one feature
    void buildTree(const cv::Mat & data,
                   const vector<unsigned int> & labels,
                   const cv_vocabulary_tree_parameter & para);

    // query the distribution of multiple feature, each vector in a row
    void query(const cv::Mat & features, vector<float> & distribution) const;

    void sortDistribution(vector<float> distribution,  vector<distributionData> & sortedDistribution, int num);


private:
    void buildTree(cv_vocabulary_tree_node * node,
                   const cv::Mat & data,
                   const vector<unsigned int> & labels,
                   const cv_vocabulary_tree_parameter & para, int depth);

    void queryOneFeature(const cv_vocabulary_tree_node * node,
                         const cv::Mat & feature,
                         vector<float> & distribution) const;

};

class cv_vocabulary_tree_node
{
public:
    vector<cv_vocabulary_tree_node * > subnodes_;
    cv::Mat cluster_center_;  // float
    int depth_;
    bool isLeaf_;

    vector<float> histgram_;  // only leaf has

    cv_vocabulary_tree_node()
    {
        depth_ = 0;
        isLeaf_ = false;
    }
};



#endif /* cv_vocabulary_tree_cpp */
