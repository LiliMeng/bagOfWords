#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

int main( int argc, char** argv )
{
  Mat src = imread( argv[1], 1 );
  Mat samples(src.rows * src.cols, 3, CV_32F);
  for( int y = 0; y < src.rows; y++ )
    for( int x = 0; x < src.cols; x++ )
      for( int z = 0; z < 3; z++)
        samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];


  int clusterCount = 15;
  Mat labels;
  int attempts = 5;
  Mat centers;
  kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );


  Mat new_image( src.size(), src.type() );
  for( int y = 0; y < src.rows; y++ )
    for( int x = 0; x < src.cols; x++ )
    { 
      int cluster_idx = labels.at<int>(y + x*src.rows,0);
      new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
      new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
      new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
    }
  imshow( "clustered image", new_image );
  waitKey( 0 );
}
