#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ctime>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/legacy/legacy.hpp>

#include "cv_vocabulary_tree.hpp"

using namespace std;
using namespace cv;

void printDistribution(ofstream &printName,  vector<float> &distribution)
{
    for(int i=0; i<distribution.size(); i++)
    {

        printName<<distribution[i]<<" ";

    }
    printName<<endl;

}

int main()
{
        int imgNum=300;
        int databaseNum=300;

        vector<Mat>  imgVec;
        imgVec.resize(imgNum);

        vector<string> nameVec;
        nameVec.resize(imgNum);

        vector<vector<KeyPoint> > keyPointsVec;
        keyPointsVec.resize(imgNum);

        vector<Mat> descriptorsVec;
        descriptorsVec.resize(imgNum);

        for(int i=0; i<imgNum; i++)
        {
            char fileName[1024] ={NULL};

            sprintf(fileName, "/home/lili/workspace/SLAM/vocabTree/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_%06d.ppm", i);

            nameVec[i]=string(fileName);

            imgVec[i]=imread(nameVec[i], CV_LOAD_IMAGE_GRAYSCALE);
        }

        //-- Step 1: Detect the keypoints using SURF Detector
        int minHessian = 400;

        SurfFeatureDetector detector(minHessian);

        SurfDescriptorExtractor extractor;

        vector<unsigned int> labels;
        for(int i=0; i<imgNum; i++)
        {
            detector.detect(imgVec[i], keyPointsVec[i]);

            extractor.compute(imgVec[i], keyPointsVec[i], descriptorsVec[i]);
            for(int j = 0; j<descriptorsVec[i].rows; j++)
            {
                labels.push_back(i);
            }
        }



        Mat all_descriptors;

        for(int i = 0; i<descriptorsVec.size(); i++)
        {
            all_descriptors.push_back(descriptorsVec[i]);
        }

        assert(labels.size() == all_descriptors.rows);


        /*
        cv_vocabulary_tree::buildTree(const cv::Mat & data,
                                   const vector<unsigned int> & labels,
                                   const cv_vocabulary_tree_parameter & para)
         */
         cv_vocabulary_tree vocTree;
         cv_vocabulary_tree_parameter para;
         para.nLabel_ = imgNum;

         clock_t begin1 = clock();
         vocTree.buildTree(all_descriptors, labels, para);
         clock_t end1 = clock();
         double buildTree_time = double(end1 - begin1) / CLOCKS_PER_SEC;
         cout.precision(5);
         cout<<"buildTree time "<<buildTree_time<<endl;

         vector<KeyPoint> queryKeypoints;
         Mat queryDescriptors;

         ///QueryImage
         {
            string queryImageName="/home/lili/workspace/SLAM/vocabTree/Lip6IndoorDataSet/Images/lip6kennedy_bigdoubleloop_000381.ppm";
            Mat queryImg=imread(queryImageName, CV_LOAD_IMAGE_GRAYSCALE);
            detector.detect(queryImg, queryKeypoints);
            extractor.compute(queryImg, queryKeypoints, queryDescriptors);
         }

         vector<float> distribution;

         clock_t begin2 = clock();
         vocTree.query(queryDescriptors, distribution);
         clock_t end2 = clock();
         double query_time = double(end2 - begin2) / CLOCKS_PER_SEC;
         cout.precision(5);
         cout<<"query time "<<query_time<<endl;

         cout<<"distribution.size() "<<distribution.size()<<endl;

         ofstream fout1("/home/lili/workspace/SLAM/vocabTree/distribution.txt");

         printDistribution(fout1,  distribution);


        return 0;
}
