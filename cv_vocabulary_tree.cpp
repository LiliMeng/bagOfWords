#include "cv_vocabulary_tree.hpp"
#include <algorithm>


cv_vocabulary_tree::cv_vocabulary_tree()
{

}

cv_vocabulary_tree::~cv_vocabulary_tree()
{

}

void cv_vocabulary_tree::buildTree(const cv::Mat & data,
                                   const vector<unsigned int> & labels,
                                   const cv_vocabulary_tree_parameter & para)
{
    assert(data.type() == CV_32F);
    assert(para.k_ >= 2);
    assert(data.rows >= para.k_);
    assert(data.rows == labels.size());

    root_ = new cv_vocabulary_tree_node();
    root_->depth_ = 0;
    this->dim_ = data.cols;
    this->nLabel_ = para.nLabel_;
    this->buildTree(root_, data, labels, para, 0);
}



void cv_vocabulary_tree::buildTree(cv_vocabulary_tree_node * node,
                                    const cv::Mat & data,
                                    const vector<unsigned int> & labels,
                                    const cv_vocabulary_tree_parameter & para, int depth)
{
    assert(node);
    assert(data.rows == labels.size());

    if (depth > para.max_depth_ || data.rows <= para.min_leaf_node_) {
        node->isLeaf_ = true;
        node->histgram_ = vector<float>(para.nLabel_, 0);
        for (int i = 0; i<labels.size(); i++) {
            node->histgram_[labels[i]] += 1.0;
        }
        for (int i = 0; i<node->histgram_.size(); i++) {
            node->histgram_[i] /= labels.size();
        }
        //   printf("leaf node size %lu\n", labels.size());
        return;
    }

    // run k mean
    cv::Mat partition; // int
    cv::Mat cluster_centers; // float?
    cv::kmeans(data, para.k_, partition, cv::TermCriteria( cv::TermCriteria::EPS+ cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_RANDOM_CENTERS, cluster_centers);

    const unsigned int K = (unsigned int) cluster_centers.rows;
    //printf("acutal k is %d\n", K);

    for (int i = 0; i<K; i++) {
        cv_vocabulary_tree_node *subNode = new cv_vocabulary_tree_node();
        assert(subNode);
        subNode->depth_ = depth + 1;
        subNode->cluster_center_ = cluster_centers.row(i);
        node->subnodes_.push_back(subNode);
    }

    // split data
    vector<cv::Mat> splitted_datas(K);
    vector<vector<unsigned int> > splitted_labels(K);
    for (int i = 0; i<partition.rows; i++) {
        int idx = partition.at<int>(i); // kmeans index
        assert(idx < K);
        assert(idx >= 0);

        splitted_datas[idx].push_back(data.row(i)); // add a row
        splitted_labels[idx].push_back(labels[i]);
    }

    // subdivide
    for (int i = 0; i<node->subnodes_.size() && i<splitted_datas.size() && i<splitted_labels.size(); i++) {
        this->buildTree(node->subnodes_[i], splitted_datas[i], splitted_labels[i], para, depth + 1);
    }
}


void cv_vocabulary_tree::query(const cv::Mat & features, vector<float> & distribution) const
{
    assert(root_);
    assert(features.cols == dim_);
    assert(features.rows >= 1);
    assert(features.type() == CV_32F);

    distribution = vector<float>(nLabel_, 0.0f);
    for (int i = 0; i<features.rows; i++) {
        vector<float> dist;
        this->queryOneFeature(root_, features.row(i), dist);

        assert(dist.size() == distribution.size());
        for (int j = 0; j<dist.size(); j++) {
            distribution[j] += dist[j];
        }
    }

    // average
    for (int i = 0; i<distribution.size(); i++) {
        distribution[i] /= features.rows;
    }
}
void cv_vocabulary_tree::queryOneFeature(const cv_vocabulary_tree_node * node,
                                         const cv::Mat & feature,
                                         vector<float> & distribution) const
{
    assert(feature.rows == 1);

    if (node->isLeaf_) {
        distribution = node->histgram_;
        return;
    }

    // compare with every cluster center
    double min_dis = INT_MAX;
    int min_index = 0;
    for (int i =0 ; i<node->subnodes_.size(); i++) {
        cv::Mat dif = node->subnodes_[i]->cluster_center_ - feature;  // vector (one row matrix) distance
        double dis = cv::norm(dif);
        if (dis < min_dis) {
            min_dis = dis;
            min_index = i;
        }
    }
    assert(min_index < node->subnodes_.size());

    this->queryOneFeature(node->subnodes_[min_index], feature, distribution);
}








