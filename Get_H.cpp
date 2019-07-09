#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main()
{
	float mul = 10.0;
    Mat birdImage;
    int board_w = 7;
    int board_h = 6;
    int board_n = board_w*board_h;
    Size board_sz = Size(7, 6);
	//Size image_size = Size(1280,720);

    Mat src = imread("12.jpg");

	//imshow("srcImage",src2);
	//Rect rect(0,150,1280, 570);
	//Mat src(src2, rect);



	imwrite("remap_src.jpg",src);

	Size image_size = src.size();
	Size size = Size(int(1280*mul),int(720*mul));

	Mat mapx = Mat(size, CV_32FC1);
	Mat mapy = Mat(size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);

	cv::Mat intrinsic_matrix(3, 3, cv::DataType<float>::type); // Intrisic matrix
	intrinsic_matrix.at<float>(0, 0) = 526.0460621686325;
	intrinsic_matrix.at<float>(1, 0) = 0;
	intrinsic_matrix.at<float>(2, 0) = 0;
	
	intrinsic_matrix.at<float>(0, 1) = 0;
	intrinsic_matrix.at<float>(1, 1) = 527.7225009042863;
	intrinsic_matrix.at<float>(2, 1) = 0;
	
	intrinsic_matrix.at<float>(0, 2) = 654.1771804245265;
	intrinsic_matrix.at<float>(1, 2) = 320.8740042532168;
	intrinsic_matrix.at<float>(2, 2) = 1;
	
	cv::Mat distortion_coeffs(4, 1, cv::DataType<float>::type);   // Distortion vector
	distortion_coeffs.at<float>(0) = -0.0467926;
	distortion_coeffs.at<float>(1) = 0.00453962;
	distortion_coeffs.at<float>(2) = 0.000105393;
	distortion_coeffs.at<float>(3) = -0.00195177;




	Mat intrinsic_mat(intrinsic_matrix), new_intrinsic_mat;

	intrinsic_mat.copyTo(new_intrinsic_mat);
	//调节视场大小,乘的系数越小视场越大
	new_intrinsic_mat.at<float>(0, 0) *= 0.35*mul;
	new_intrinsic_mat.at<float>(1, 1) *= 0.35*mul;
	//调节校正图中心，建议置于校正图中心
	new_intrinsic_mat.at<float>(0, 2) = 0.5 *mul * src.cols;
	new_intrinsic_mat.at<float>(1, 2) = 0.5 *mul * src.rows;

	Mat src1;
	//fisheye::undistortImage(src, src1, intrinsic_matrix, distortion_coeffs, new_intrinsic_mat, size);

	fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,new_intrinsic_mat, size,CV_32FC1,mapx,mapy);
	//fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R,
		//getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, image_size, 1, size, 0), image_size, CV_32FC1, mapx, mapy);
	imshow("src", src);

	cv::remap(src, src1, mapx, mapy, INTER_LINEAR);
	imshow("remap", src1);
	imwrite("remap.jpg", src1);
	//Mat srcImage = src1.clone();
	Mat grayImage;
	cvtColor(src1, grayImage, CV_BGR2GRAY);


    waitKey(10);


	Rect rect(0,260*mul,1280*mul, 460*mul);
	Mat temp(src1, rect);
	imshow("roi", temp);

	Size temp_size = temp.size();



    vector<Point2f> corners;

    //寻找4组对应点坐标
    
    bool found = findChessboardCorners(temp, board_sz, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

	cout << "found == 1:" << found << endl;



        if (!found)
        {

            waitKey(30);
            cout << "找不到角点，需删除图片文件,重新排列文件名，再次标定" << endl;
            getchar();
            
            exit(1);
        }
        else
        {
            // 亚像素精确化
            cornerSubPix(grayImage, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
          
        }

    cout << corners << endl;
    drawChessboardCorners(temp, board_sz, corners, found);


    namedWindow("chess");
    imshow("chess", temp);
	imwrite("chess.jpg",temp);
    Point2f imagePoints[4], objectPoints[4];
    objectPoints[0].x = 0;  objectPoints[0].y = 0;
    objectPoints[1].x = board_w - 1;    objectPoints[1].y = 0;
    objectPoints[2].x = 0;  objectPoints[2].y = board_h-1;
    objectPoints[3].x = board_w-1;  objectPoints[3].y = board_h-1;


	/***
    imagePoints[0] = corners.at(0);
    imagePoints[1] = corners.at(board_w - 1);
    imagePoints[2] = corners.at(board_w*(board_h - 1));
    imagePoints[3] = corners.at(board_n - 1);
	***/

	
    imagePoints[3] = corners.at(0) ;
    imagePoints[2] = corners.at(board_w - 1) ;
    imagePoints[1] = corners.at(board_w*(board_h - 1)) ;
    imagePoints[0] = corners.at(board_n - 1) ;

    circle(temp, imagePoints[0], 5, Scalar(0, 0, 255),-1);
    circle(temp, imagePoints[1], 5, Scalar(0, 255, 0), -1);
    circle(temp, imagePoints[2], 5, Scalar(255, 0, 0), -1);
    circle(temp, imagePoints[3], 5, Scalar(255, 255, 255), -1);
    cout << objectPoints[0] << imagePoints[0] << endl;
    cout << objectPoints[1] << imagePoints[1] << endl;
    cout << objectPoints[2] << imagePoints[2] << endl;
    cout << objectPoints[3] << imagePoints[3] << endl;





    Mat H;
				
	cv::Mat H(3, 3, cv::DataType<double>::type); // Intrisic matrix




    //计算单应矩阵
    //计算obj到img的单应矩阵而不是直接计算img到obj单应矩阵的主要原因是可控转换后输入图像的大小
    //其控制参数即为h33，经过MATLAB的运算可知h33与坐标成线性单增关系，从而能保证图像不失真


    H = getPerspectiveTransform(imagePoints, objectPoints);




    warpPerspective(temp, birdImage, H, temp_size , WARP_INVERSE_MAP+INTER_LINEAR);
  
    namedWindow("birdImage");
    imshow("birdImage", birdImage);

    waitKey(0);
    return 0;
}


