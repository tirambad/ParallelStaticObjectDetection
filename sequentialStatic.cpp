#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#include <math.h>

using namespace cv;

int main(int argc, char** argv )
{
    int i;
    int counter = 0;
    Mat temp1, temp1Gray; 
    Mat temp2, temp2Gray; 
    Mat channel1[3];
    Mat channel2[3];
    int j, k;
    unsigned char *recvMatrix;
    Vec3b pixel1, pixel1Gray, pixel1Blue, pixel1Green, pixel1Red;
    Vec3b pixel2, pixel2Gray, pixel2Blue, pixel2Green, pixel2Red;
    Vec3b pixelRef, pixelRefGray, pixelRefBlue, pixelRefGreen, pixelRefRed;
    int counter = 0; // To ignore first receive
    long int threshold = 100;
    long int threshold2 = 50;
    long int thresholdGray = 100;
    long int thresholdGray2 = 50;
    long int thresholdBlue = 100;
    long int thresholdBlue2 = 50;
    long int thresholdGreen = 100;
    long int thresholdGreen2 = 50;
    long int thresholdRed = 100;
    long int thresholdRed2 = 50;
    long int ref, val1, val2;

    VideoWriter videoNormal, videoGray, videoBlue, videoGreen, videoRed;
    
    VideoCapture cap(argv[1]); // Open File
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat edges;
    Mat frame, backgroundImage, backgroundGray;
    Mat channelBackground[3], channelBackground[3];
    //index each frame to send to different nodes
    int index = 0;
    //namedWindow("edges",1);

    cap >> frame; // get a new frame from camera/video
    //make a continuous frame
    if (!frame.isContinuous())
    { 
        frame = frame.clone();
    }
    
    //Declare background ; First Frame = Background 
    //Declare Background Frame
    backgroundImage = frame.clone();
    //BackgroundGrayScale
    vtColor( backgroundImage, backgroundGray, CV_BGR2GRAY );
    //Split into BGR channel
    split (backgroundImage, channelBackground);
    //Define videowriter
    VideoWriter videoNormal("out_Normal.avi",CV_FOURCC('M','J','P','G'),10, Size(frame.cols, frame.rows),true);
    VideoWriter videoGray("out_Normal.avi",CV_FOURCC('M','J','P','G'),10, Size(frame.cols, frame.rows),true);
    VideoWriter videoBlue("out_Normal.avi",CV_FOURCC('M','J','P','G'),10, Size(frame.cols, frame.rows),true);
    VideoWriter videoGreen("out_Normal.avi",CV_FOURCC('M','J','P','G'),10, Size(frame.cols, frame.rows),true);
    VideoWriter videoRed("out_Normal.avi",CV_FOURCC('M','J','P','G'),10, Size(frame.cols, frame.rows),true);
    }
    for(;;)
    {
        if(index > 0)
            cap >> frame; // get a new frame from camera/video
        //make a continuous frame
        if (!frame.isContinuous())
        { 
            frame = frame.clone();
        }
        
        if(counter > 0){
            temp2 = newImage.clone();
            //ConvertToGrayScale
            cvtColor( temp2, temp2Gray, CV_BGR2GRAY );
            
            //Split into BGR
            split (temp2, channel2);
            if (!temp2Gray.isContinuous())
            { 
                temp2Gray = temp2Gray.clone();
            }

            //Start checking the difference
            for(j = 0; j < temp2.cols; j++){
                for(k = 0; k < temp2.rows; k++){
                    if(j == 0 && k == 0){
                        pixelRef = backgroundImage.at<Vec3b>(Point(j, k));
                        pixelRefGray = backgroundGray.at<uchar>(Point(j, k));
                        pixelRefBlue = channelBackground[0].at<uchar>(Point(j, k));
                        pixelRefGreen = channelBackground[1].at<uchar>(Point(j, k));
                        pixelRefRed = channelBackground[2].at<uchar>(Point(j, k));
                    }

                    pixel1 = temp1.at<Vec3b>(Point(j, k));
                    pixel2 = temp2.at<Vec3b>(Point(j, k));

                    pixel1Gray = temp1Gray.at<uchar>(Point(j, k)); 
                    pixel2Gray = temp2Gray.at<uchar>(Point(j, k));
                    
                    pixel1Blue = channel1[0].at<uchar>(Point(j, k)); 
                    pixel2Blue = channel2[0].at<uchar>(Point(j, k));

                    pixel1Green = channel1[1].at<uchar>(Point(j, k)); 
                    pixel2Green = channel2[1].at<uchar>(Point(j, k));

                    pixel1Red = channel1[2].at<uchar>(Point(j, k)); 
                    pixel2Red = channel2[2].at<uchar>(Point(j, k));

                    //Check difference for normal image
                    ref = (pixelRef[0] + pixelRef[1] + pixelRef[2]);
                    val1 = (pixel1[0] + pixel1[1] + pixel1[2]);
                    val2 = (pixel2[0] + pixel2[1] + pixel2[2]);
                    if(abs(ref - val1) > threshold && abs(val1 - val2) < threshold2){
                        //New static pixel detected
                        temp1.at<Vec3b>(Point(j,k)) = 0;
                    }

                    //Check difference for grayscale image
                    ref = (pixelRefGray[0]);
                    val1 = (pixel1Gray[0]);
                    val2 = (pixel2Gray[0]);
                    if(abs(ref - val1) > thresholdGray && abs(val1 - val2) < thresholdGray2){
                        //New static pixel detected
                        temp1.at<uchar>(Point(j,k)) = 0;
                    }

                    //Check difference for blue image
                    ref = pixelRefBlue[0];
                    val1 = (pixel1Blue[0]);
                    val2 = (pixel2Blue[0]);
                    //printf("%d: Ref - val = %ld Val1 - Val2 = %ld\n", rank, abs(ref - val1), abs(val1 - val2));
                    if(abs(ref - val1) > thresholdBlue && abs(val1 - val2) < thresholdBlue2){
                        //New static pixel detected
                        channel1[0].at<uchar>(Point(j,k)) = 0;
                    }

                    //Check difference for Green image
                    ref = (pixelRefGreen[0]);
                    val1 = (pixel1Green[0]);
                    val2 = (pixel2Green[0]);
                    if(abs(ref - val1) > thresholdGreen && abs(val1 - val2) < thresholdGreen2){
                        //New static pixel detected
                        channel1[1].at<uchar>(Point(j,k)) = 0;
                    }

                    //Check difference for Red image
                    ref = (pixelRefRed[0]);
                    val1 = (pixel1Red[0]);
                    val2 = (pixel2Red[0]);
                    if(abs(ref - val1) > thresholdRed && abs(val1 - val2) < thresholdRed2){
                        //New static pixel detected
                        channel1[2].at<uchar>(Point(j,k)) = 0;
                    }
                }
            }

            //Write Normal Image
            video.write(temp1);
            //Write Grayscale Image
            video.write(temp1Gray);
            //Write Blue Image
            video.write(channel1[0]);
            //Write Green Image
            video.write(channel1[1]);
            //Write Red Image
            video.write(channel1[2]);

            //Put Second frame data to first Frame to calculate next static objects
            temp1 = temp2.clone();
            temp1Gray = temp2Gray.clone();
            channel1[0] = channel2[0].clone();
            channel1[1] = channel2[1].clone();
            channel1[2] = channel2[2].clone();
        }
        else{
            temp1 = frame.clone(); 
            //Convert to GrayScale
            cvtColor( temp1, temp1Gray, CV_BGR2GRAY );
            
            //Split into BGR
            split (temp1, channel1);
            if (!temp1Gray.isContinuous())
            { 
                temp1Gray = temp1Gray.clone();
            }
            counter = 1;
        }
    

        index = 1;
        if(waitKey(30) >= 0) break;
    }
    MPI_Finalize();
    return 0;
}